#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE BACKEND'S CUDA KERNELS - fused, because P0 measured kernel count as the
lever and precision as a red herring.

WHAT P0 ESTABLISHED, AND WHY THIS FILE LOOKS LIKE THIS
    * the bounded backend is **LAUNCH-BOUND**: 128x the work costs 1.33x the
      time, the launch floor is 9.59 us, and a naive cupy expression chain
      issues ~226 launches per LM iteration.  So every stage here is ONE
      kernel, not an expression;
    * **float32 is required for CORRECTNESS, not speed**: the FP32 solve of the
      normal equations returns 2.9e-1 relative error at cond 1e8 and NaN at
      1e10, where the FP32 square-root form holds 1.2e-6.  Hence
      `chol_solve` factors in float32 but keeps the *factor* as the working
      object, and the caller may hand it a pre-conditioned system;
    * **atomics ship** (the pre-committed decision of plan 5.3): fixed-order
      segmented reduction measured +245 % / +563 %, far past the 15 % limit.
      Where atomics are live, **byte hashes are the wrong control** and
      comparison uses a tolerance;
    * **cuSOLVER is 91 % of the composed floor**: `potrf` on 225x225 costs
      0.437 ms for ~3.8 MFLOP, and doubling the matrix costs only 1.17x - that
      is host overhead, not factorisation.  `chol_solve` below is the answer to
      that, and the number to beat is **0.529 -> <=0.1 ms**.

WHY A SINGLE-BLOCK CHOLESKY IS THE RIGHT SHAPE HERE
    The reduced camera system of a BOUNDED local BA is small by construction -
    a 10-30 keyframe window is 60-450 unknowns.  cuSOLVER is built for matrices
    three orders of magnitude larger and pays a host round trip per call.  One
    block, one launch, factor and both triangular solves in the same kernel,
    is not a clever optimisation: it is the shape the problem actually has.
"""
from __future__ import annotations

import cupy as cp

# --------------------------------------------------------------------------
_SRC = r"""
extern "C" {

/* ---- ONE KERNEL: residual, Jacobians, landmark blocks -----------------
   One thread per observation.  Replaces six cupy expressions and their
   temporaries; P0 measured the chain form at ~226 launches per iteration on a
   problem whose arithmetic is free.                                        */
__global__ void build_blocks(const float* __restrict__ Tcw,   /* (nk,12) */
                             const float* __restrict__ pts,   /* (nl,3)  */
                             const int* __restrict__ okf,
                             const int* __restrict__ olm,
                             const float* __restrict__ ouv,
                             float* __restrict__ Jpose,       /* (N,2,6) */
                             float* __restrict__ Jlm,         /* (N,2,3) */
                             float* __restrict__ resw,        /* (N,2)   */
                             float* __restrict__ Hll,         /* (nl,9)  */
                             float* __restrict__ bl,          /* (nl,3)  */
                             float* __restrict__ Hpl,         /* (N,6,3) */
                             const int N, const float fx, const float cx,
                             const float cy, const float huber)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= N) return;
    const int k = okf[o], l = olm[o];
    const float* T = Tcw + (long long)k * 12;
    const float* P = pts + (long long)l * 3;

    float px = T[0]*P[0] + T[1]*P[1] + T[2]*P[2]  + T[3];
    float py = T[4]*P[0] + T[5]*P[1] + T[6]*P[2]  + T[7];
    float pz = T[8]*P[0] + T[9]*P[1] + T[10]*P[2] + T[11];
    if (pz < 1e-6f) pz = 1e-6f;
    const float iz = 1.f / pz, iz2 = iz * iz;
    float ru = fx * px * iz + cx - ouv[o * 2];
    float rv = fx * py * iz + cy - ouv[o * 2 + 1];

    float e = sqrtf(ru * ru + rv * rv);
    float w = (e <= huber) ? 1.f : (huber / fmaxf(e, 1e-9f));
    float sw = sqrtf(w);

    float A[2][3];
    A[0][0] = fx * iz;  A[0][1] = 0.f;      A[0][2] = -fx * px * iz2;
    A[1][0] = 0.f;      A[1][1] = fx * iz;  A[1][2] = -fx * py * iz2;

    float dP[3][6] = {{1,0,0, 0, pz,-py},
                      {0,1,0,-pz, 0, px},
                      {0,0,1, py,-px, 0}};

    float jp[2][6], jl[2][3];
    for (int q = 0; q < 2; ++q) {
        for (int c = 0; c < 6; ++c) {
            float v = 0.f;
            for (int m2 = 0; m2 < 3; ++m2) v += A[q][m2] * dP[m2][c];
            jp[q][c] = v * sw;
            Jpose[(long long)o * 12 + q * 6 + c] = jp[q][c];
        }
        for (int c = 0; c < 3; ++c) {
            float v = 0.f;
            for (int m2 = 0; m2 < 3; ++m2) v += A[q][m2] * T[m2 * 4 + c];
            jl[q][c] = v * sw;
            Jlm[(long long)o * 6 + q * 3 + c] = jl[q][c];
        }
    }
    float rwu = ru * sw, rwv = rv * sw;
    resw[o * 2] = rwu;  resw[o * 2 + 1] = rwv;

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b)
            atomicAdd(&Hll[(long long)l * 9 + a * 3 + b],
                      jl[0][a] * jl[0][b] + jl[1][a] * jl[1][b]);
        atomicAdd(&bl[(long long)l * 3 + a],
                  -(jl[0][a] * rwu + jl[1][a] * rwv));
    }
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 3; ++b)
            Hpl[(long long)o * 18 + a * 3 + b] =
                jp[0][a] * jl[0][b] + jp[1][a] * jl[1][b];
}

/* ---- THE STEREO SCALE PRIOR -------------------------------------------
   Reprojection residuals alone leave SCALE unobservable: fixing one keyframe
   removes 6 DoF of gauge, and a bundle of bearings is invariant to the 7th.
   Measured consequence on MH_01 - local BA with reprojection only drifted
   sim3 scale to 1.0285 and took the ATE from 0.0649 (odometry) to 0.2180.

   The stereo pair already measured the depth of every landmark in its HOST
   keyframe, so that measurement enters as a prior on inverse depth:

       r     = 1/||P_w - c_host||  -  rho_measured
       sigma = sigma_disparity / (f*B)                  <- CONSTANT, Class B

   The sigma is constant because rho = d/(f*B) is LINEAR in disparity, so a
   0.5 px disparity noise is a fixed inverse-depth noise no matter how far
   away the point is - which is the same fact that makes inverse depth the
   right parameterisation for far points (plan 5.4).

   It touches only Hll and bl, so it costs one kernel and nothing in the Schur
   complement.                                                            */
__global__ void depth_prior(const float* __restrict__ pts,
                            const float* __restrict__ chost,   /* (nl,3) */
                            const float* __restrict__ rho0,    /* (nl,)  */
                            const float* __restrict__ invsig,  /* (nl,)  */
                            const float* __restrict__ valid,   /* (nl,)  */
                            float* __restrict__ Hll,
                            float* __restrict__ bl,
                            const int nl, const float huber)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nl || valid[l] < 0.5f) return;
    const float* P = pts + (long long)l * 3;
    const float* c = chost + (long long)l * 3;
    float dx = P[0] - c[0], dy = P[1] - c[1], dz = P[2] - c[2];
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < 1e-12f) return;
    float dist = sqrtf(d2);
    float rho = 1.f / dist;
    const float inv_sigma = invsig[l];
    float r = (rho - rho0[l]) * inv_sigma;
    float w = (fabsf(r) <= huber) ? 1.f : (huber / fmaxf(fabsf(r), 1e-9f));
    float sw = sqrtf(w);
    /* d(rho)/dP = -(P-c)/dist^3 */
    float g = -rho * rho / dist * inv_sigma * sw;
    float J[3] = {g * dx, g * dy, g * dz};
    float rw = r * sw;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b)
            atomicAdd(&Hll[(long long)l * 9 + a * 3 + b], J[a] * J[b]);
        atomicAdd(&bl[(long long)l * 3 + a], -J[a] * rw);
    }
}

/* ---- batched 3x3 inverse with LM damping ------------------------------ */
__global__ void inv3_damped(const float* __restrict__ Hll,
                            float* __restrict__ Hlli,
                            const int nl, const float lam)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nl) return;
    const float* H = Hll + (long long)l * 9;
    float tr = (H[0] + H[4] + H[8]) / 3.f;
    float dmp = lam * fmaxf(tr, 1e-9f);
    float a = H[0] + dmp, b = H[1], c = H[2];
    float d2 = H[3], e = H[4] + dmp, f = H[5];
    float g = H[6], h = H[7], i = H[8] + dmp;
    float A =  (e * i - f * h), B = -(d2 * i - f * g), C =  (d2 * h - e * g);
    float det = a * A + b * B + c * C;
    if (fabsf(det) < 1e-30f) det = (det < 0 ? -1e-30f : 1e-30f);
    float id = 1.f / det;
    float* O = Hlli + (long long)l * 9;
    O[0] = A * id;                    O[1] = -(b * i - c * h) * id;
    O[2] =  (b * f - c * e) * id;     O[3] = B * id;
    O[4] =  (a * i - c * g) * id;     O[5] = -(a * f - c * d2) * id;
    O[6] = C * id;                    O[7] = -(a * h - b * g) * id;
    O[8] =  (a * e - b * d2) * id;
}

/* ---- the Schur assembly, over a PAIR LIST -----------------------------
   Observations are sorted by landmark; pa/pb enumerate every ordered pair
   within a landmark segment.  One thread per (pair, row, col).  This is the
   kernel micro/fused.py measured at 7.3x the cupy chain, generalised to a
   variable observation count per landmark.

   ATOMICS SHIP - the pre-committed decision of plan 5.3, measured: fixed-order
   segmented reduction costs +245 percent (fp32) and +563 percent (fp64)
   against a 15 percent limit.  Where atomics are live, BYTE HASHES ARE THE
   WRONG CONTROL and comparison uses a tolerance.                           */
__global__ void schur_pairs(const float* __restrict__ Hpl,
                            const float* __restrict__ Hlli,
                            const int* __restrict__ okf,
                            const int* __restrict__ olm,
                            const int* __restrict__ pa,
                            const int* __restrict__ pb,
                            float* __restrict__ Hred,
                            const long long npair, const int d, const int n)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = npair * d * d;
    if (idx >= total) return;
    int b = (int)(idx % d);  long long t = idx / d;
    int a = (int)(t % d);    t /= d;
    long long p = t;

    const int oi = pa[p], oj = pb[p];
    const int ki = okf[oi], kj = okf[oj];
    const float* hi = Hpl + (long long)oi * d * 3;
    const float* hj = Hpl + (long long)oj * d * 3;
    const float* hl = Hlli + (long long)olm[oi] * 9;

    float acc = 0.f;
    for (int k = 0; k < 3; ++k) {
        float tk = 0.f;
        for (int q = 0; q < 3; ++q) tk += hi[a * 3 + q] * hl[q * 3 + k];
        acc += tk * hj[b * 3 + k];
    }
    atomicAdd(&Hred[(long long)(ki * d + a) * n + (kj * d + b)], -acc);
}

/* ---- the Schur term on the gradient ----------------------------------- */
__global__ void schur_grad(const float* __restrict__ Hpl,
                           const float* __restrict__ Hlli,
                           const float* __restrict__ bl,
                           const int* __restrict__ okf,
                           const int* __restrict__ olm,
                           float* __restrict__ g,
                           const int N, const int d)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)N * d) return;
    int a = (int)(idx % d);
    int o = (int)(idx / d);
    const int k = okf[o], l = olm[o];
    const float* h = Hpl + (long long)o * d * 3;
    const float* hl = Hlli + (long long)l * 9;
    const float* bb = bl + (long long)l * 3;
    float acc = 0.f;
    for (int c = 0; c < 3; ++c) {
        float tc = 0.f;
        for (int q = 0; q < 3; ++q) tc += h[a * 3 + q] * hl[q * 3 + c];
        acc += tc * bb[c];
    }
    atomicAdd(&g[k * d + a], -acc);
}

/* ---- the pose-block accumulation and the gradient --------------------- */
__global__ void pose_blocks(const float* __restrict__ Jp,
                            const float* __restrict__ res,
                            const int* __restrict__ kf,
                            float* __restrict__ Hred, float* __restrict__ g,
                            const int N, const int r, const int d, const int n)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * d * d;
    if (idx >= total) return;
    int b = (int)(idx % d);  long long t = idx / d;
    int a = (int)(t % d);    t /= d;
    int o = (int)t;
    int k = kf[o];
    if (k < 0) return;
    const float* J = Jp + (long long)o * r * d;
    float acc = 0.f;
    for (int q = 0; q < r; ++q) acc += J[q * d + a] * J[q * d + b];
    atomicAdd(&Hred[(long long)(k * d + a) * n + (k * d + b)], acc);
    if (b == 0) {
        float gg = 0.f;
        const float* rr = res + (long long)o * r;
        for (int q = 0; q < r; ++q) gg += J[q * d + a] * rr[q];
        atomicAdd(&g[k * d + a], -gg);
    }
}

/* ---- ONE KERNEL: Cholesky + forward + back substitution ---------------
   One block.  H is overwritten by its lower Cholesky factor; x receives the
   solution.  n <= 1024.  This is the replacement for the cuSOLVER call P0
   measured at 91 % of the composed floor.                                */
__global__ void chol_solve(float* __restrict__ H, const float* __restrict__ b,
                           float* __restrict__ x, const int n,
                           int* __restrict__ status)
{
    const int tid = threadIdx.x, nt = blockDim.x;

    for (int k = 0; k < n; ++k) {
        if (tid == 0) {
            float dkk = H[(long long)k * n + k];
            if (!(dkk > 0.f)) { *status = k + 1; dkk = 1e-8f; }
            H[(long long)k * n + k] = sqrtf(dkk);
        }
        __syncthreads();
        const float dk = H[(long long)k * n + k];
        const float inv = 1.f / dk;
        for (int i = k + 1 + tid; i < n; i += nt)
            H[(long long)i * n + k] *= inv;
        __syncthreads();
        const int mm = n - k - 1;
        for (long long p = tid; p < (long long)mm * mm; p += nt) {
            int i = k + 1 + (int)(p / mm), j = k + 1 + (int)(p % mm);
            if (j <= i)
                H[(long long)i * n + j] -= H[(long long)i * n + k]
                                         * H[(long long)j * n + k];
        }
        __syncthreads();
    }

    /* ---- forward solve  L y = b  --------------------------------------
       COLUMN-oriented, not row-oriented, and the difference is the whole
       cost.  The row form computes a dot product per row, which needs a
       cross-thread reduction: with 256 threads that is 256 SERIAL adds by
       thread 0 per row, and at n=90 it measured 0.371 ms - no better than the
       cuSOLVER call it was written to replace.  The column form updates the
       remaining entries in parallel and syncs twice per column, with no
       reduction at all.                                                    */
    for (int i = tid; i < n; i += nt) x[i] = b[i];
    __syncthreads();
    for (int j = 0; j < n; ++j) {
        if (tid == 0) x[j] /= H[(long long)j * n + j];
        __syncthreads();
        const float xj = x[j];
        for (int i = j + 1 + tid; i < n; i += nt)
            x[i] -= H[(long long)i * n + j] * xj;
        __syncthreads();
    }
    /* ---- back solve  L^T x = y  --------------------------------------- */
    for (int j = n - 1; j >= 0; --j) {
        if (tid == 0) x[j] /= H[(long long)j * n + j];
        __syncthreads();
        const float xj = x[j];
        for (int i = tid; i < j; i += nt)
            x[i] -= H[(long long)j * n + i] * xj;
        __syncthreads();
    }
}

/* ---- landmark back-substitution, CSR layout ---------------------------- */
__global__ void backsub(const float* __restrict__ Hpl,
                        const float* __restrict__ Hlli,
                        const float* __restrict__ bl,
                        const float* __restrict__ dxp,
                        const int* __restrict__ okf,
                        const int* __restrict__ offs,
                        float* __restrict__ dxl,
                        const int nl, const int d)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nl) return;
    float rhs[3] = {bl[l * 3], bl[l * 3 + 1], bl[l * 3 + 2]};
    for (int s2 = offs[l]; s2 < offs[l + 1]; ++s2) {
        const int k = okf[s2];
        const float* h = Hpl + (long long)s2 * d * 3;
        for (int c = 0; c < 3; ++c) {
            float acc = 0.f;
            for (int a = 0; a < d; ++a) acc += h[a * 3 + c] * dxp[k * d + a];
            rhs[c] -= acc;
        }
    }
    const float* hl = Hlli + (long long)l * 9;
    for (int c = 0; c < 3; ++c)
        dxl[l * 3 + c] = hl[c * 3] * rhs[0] + hl[c * 3 + 1] * rhs[1]
                       + hl[c * 3 + 2] * rhs[2];
}

/* ======================================================================
   INVERSE-DEPTH LANDMARKS (plan 5.4).  One scalar per landmark, anchored
   to a HOST keyframe by a fixed normalised-plane bearing.

       P_host  = m / rho          m = (x_n, y_n, 1), CONSTANT
       P_world = T_wc_host * P_host
       P_cam   = T_cw_obs  * P_world

   WHY THIS SHAPE, AND NOT THREE FREE NUMBERS
     * `rho = d / (f*B)` is LINEAR in disparity, so the stereo measurement's
       information is the SAME at 2 m and at 200 m.  A Euclidean 3x3 landmark
       block degenerates along the ray as 1/z^2: at EuRoC's far-bearing
       ceiling (48 m) against a window baseline of ~0.1 m the conditioning
       ratio is ~1e-6, and in float32 (eps 1.2e-7) the explicit cofactor
       inverse in `inv3_damped` returns noise.  That is the mechanism behind
       "the far band is refuted for now": it was never the band, it was the
       parameterisation - exactly as plan 5.4 predicted;
     * here `Hll` is a SCALAR, and the stereo prior adds a positive constant
       `1/sigma_rho^2` to it, so it is bounded away from zero by construction
       and no damping heuristic is load-bearing;
     * the prior becomes LINEAR IN THE STATE - residual `(rho - rho_meas) *
       invsig`, Jacobian `invsig`.  The inverse-RANGE-versus-inverse-DEPTH
       units defect that cost MH_01 a path ratio of 0.8906 CANNOT BE WRITTEN
       in this form: there is no conversion left to get wrong.

   * THE HOST POSE IS A VARIABLE TOO, AND OMITTING IT WOULD BE A LIE
     A landmark is rigidly attached to its host, so when the host keyframe is
     free in the window its motion MOVES the landmark.  Every observation
     therefore carries TWO pose blocks - observer and host - and `Hpl` holds
     both.  Dropping the host block would make the linearisation inconsistent
     with the update it produces, which surfaces as steps the monotone guard
     rejects rather than as a wrong answer, which is worse: it would look like
     a slow solver instead of a wrong one.  When the host sits outside the
     window it is a constant and its slot index is -1.

   NOTE the host's OWN observation is structurally zero: with obs == host,
   d(P_c)/d(xi_obs) + d(P_c)/d(xi_host) = [I | -hat(P)] + [-I | hat(P)] = 0,
   and m/rho reprojects to the host keypoint by construction.  Those rows are
   dropped at assembly, so they cost no threads and cannot deflate the
   reported RMS with exact zeros.
   ====================================================================== */
__global__ void build_blocks_id(const float* __restrict__ Tcw,   /* (nk,12) */
                                const float* __restrict__ hpose, /* (nl,12) */
                                const int* __restrict__ hidx,    /* (nl,)   */
                                const float* __restrict__ mbear, /* (nl,3)  */
                                const float* __restrict__ rhov,  /* (nl,)   */
                                const int* __restrict__ okf,
                                const int* __restrict__ olm,
                                const float* __restrict__ ouv,
                                float* __restrict__ Jp,      /* (N,2,2,6) */
                                float* __restrict__ resw,    /* (N,2)     */
                                int*   __restrict__ kslot,   /* (N,2)     */
                                float* __restrict__ Hll,     /* (nl,)     */
                                float* __restrict__ bl,      /* (nl,)     */
                                float* __restrict__ Hpl,     /* (N,2,6)   */
                                const int N, const float fx, const float cx,
                                const float cy, const float huber)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= N) return;
    const int k = okf[o], l = olm[o];
    const int hk = hidx[l];
    const float* T = Tcw + (long long)k * 12;
    const float rho = rhov[l];
    const float* mm = mbear + (long long)l * 3;

    /* ---- T_wc of the host -----------------------------------------------
       An in-window host is INVERTED HERE from the live T_cw rather than
       mirrored from a second host-side array.  Maintaining `poses = inv(T_cw)`
       on the host cost a batched inverse, a copy and a `.set()` per iteration
       - and P0's lesson is that at these sizes the arithmetic is free and the
       host round trip is everything.  Out-of-window hosts are constants and
       arrive already as T_wc.                                              */
    float Hw[12];
    if (hk >= 0) {
        const float* C = Tcw + (long long)hk * 12;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) Hw[a * 4 + b] = C[b * 4 + a];
            Hw[a * 4 + 3] = -(C[a] * C[3] + C[4 + a] * C[7]
                              + C[8 + a] * C[11]);
        }
    } else {
        const float* E = hpose + (long long)l * 12;
        for (int a = 0; a < 12; ++a) Hw[a] = E[a];
    }

    const float irho = 1.f / rho;
    float Ph[3] = {mm[0] * irho, mm[1] * irho, mm[2] * irho};

    float Pw[3];
    Pw[0] = Hw[0]*Ph[0] + Hw[1]*Ph[1] + Hw[2]*Ph[2]  + Hw[3];
    Pw[1] = Hw[4]*Ph[0] + Hw[5]*Ph[1] + Hw[6]*Ph[2]  + Hw[7];
    Pw[2] = Hw[8]*Ph[0] + Hw[9]*Ph[1] + Hw[10]*Ph[2] + Hw[11];

    float px = T[0]*Pw[0] + T[1]*Pw[1] + T[2]*Pw[2]  + T[3];
    float py = T[4]*Pw[0] + T[5]*Pw[1] + T[6]*Pw[2]  + T[7];
    float pz = T[8]*Pw[0] + T[9]*Pw[1] + T[10]*Pw[2] + T[11];
    if (pz < 1e-6f) pz = 1e-6f;
    const float iz = 1.f / pz, iz2 = iz * iz;
    float ru = fx * px * iz + cx - ouv[o * 2];
    float rv = fx * py * iz + cy - ouv[o * 2 + 1];

    float e = sqrtf(ru * ru + rv * rv);
    float w = (e <= huber) ? 1.f : (huber / fmaxf(e, 1e-9f));
    float sw = sqrtf(w);

    float A[2][3];
    A[0][0] = fx * iz;  A[0][1] = 0.f;      A[0][2] = -fx * px * iz2;
    A[1][0] = 0.f;      A[1][1] = fx * iz;  A[1][2] = -fx * py * iz2;

    /* ---- observer pose block: A @ [I | -hat(Pc)] ---------------------- */
    float dPo[3][6] = {{1,0,0,  0, pz,-py},
                       {0,1,0,-pz,  0, px},
                       {0,0,1, py,-px,  0}};
    for (int q = 0; q < 2; ++q)
        for (int c = 0; c < 6; ++c) {
            float v = 0.f;
            for (int m2 = 0; m2 < 3; ++m2) v += A[q][m2] * dPo[m2][c];
            Jp[(long long)o * 24 + q * 6 + c] = v * sw;
        }

    /* ---- R_ch = R_cw_obs * R_wc_host ---------------------------------- */
    float Rch[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            float v = 0.f;
            for (int q = 0; q < 3; ++q) v += T[a * 4 + q] * Hw[q * 4 + b];
            Rch[a][b] = v;
        }
    float ARch[2][3];
    for (int q = 0; q < 2; ++q)
        for (int b = 0; b < 3; ++b) {
            float v = 0.f;
            for (int m2 = 0; m2 < 3; ++m2) v += A[q][m2] * Rch[m2][b];
            ARch[q][b] = v;
        }

    /* ---- host pose block: A @ R_ch @ [-I | +hat(Ph)] -------------------
       T_cw_host <- exp(d) T_cw_host  =>  T_wc_host <- T_wc_host exp(-d),
       so d(P_h)/d(delta) = [-I | +hat(P_h)], in the HOST camera frame.   */
    if (hk >= 0) {
        float dPh[3][6] = {{-1, 0, 0,      0, -Ph[2],  Ph[1]},
                           { 0,-1, 0,  Ph[2],      0, -Ph[0]},
                           { 0, 0,-1, -Ph[1],  Ph[0],      0}};
        for (int q = 0; q < 2; ++q)
            for (int c = 0; c < 6; ++c) {
                float v = 0.f;
                for (int m2 = 0; m2 < 3; ++m2) v += ARch[q][m2] * dPh[m2][c];
                Jp[(long long)o * 24 + 12 + q * 6 + c] = v * sw;
            }
    } else {
        for (int c = 0; c < 12; ++c) Jp[(long long)o * 24 + 12 + c] = 0.f;
    }
    kslot[o * 2] = k;
    kslot[o * 2 + 1] = hk;

    /* ---- the LANDMARK block: (x_n, y_n, rho), three parameters ---------
       P_h = (a, b, 1) / rho, so

           d(P_h)/da = e1/rho ,  d(P_h)/db = e2/rho ,  d(P_h)/drho = -P_h/rho

       * WHY THE BEARING IS A VARIABLE AND NOT A CONSTANT.
       Freezing it at the host keypoint costs the host's ~1 px of localisation
       noise PERMANENTLY: no adjustment of rho can move a landmark off a wrong
       ray, so the error is locked and - because every landmark hosted by one
       keyframe shares that keyframe's noise - it is CORRELATED rather than
       independent, and correlated structure error integrates into a bend
       instead of averaging out.  Measured directly: the global BA's final
       residual rose a uniform 40 % on all four R0 rows against the Euclidean
       path, and a controlled experiment reproduced it exactly by varying only
       the host bearing's noise (0 px -> 1.19x, 0.5 px -> 1.48x, 1 px ->
       2.11x, while the Euclidean solver stayed flat at 0.543 px because it
       AVERAGES the host measurement in with every other view).  On v2_01 that
       became a 63 % ATE regression with the RPE and drift UNCHANGED - the
       signature of correlated, not larger, error.

       Three parameters is the same DoF as a Euclidean point, but in a chart
       where the ill-conditioned direction is the well-conditioned one, and
       where the record stays host-relative so a loop correction is exact.  */
    float jl[2][3];
    for (int q = 0; q < 2; ++q) {
        float dr = 0.f;
        for (int c = 0; c < 3; ++c) dr += ARch[q][c] * Ph[c];
        jl[q][0] = ARch[q][0] * irho * sw;
        jl[q][1] = ARch[q][1] * irho * sw;
        jl[q][2] = -dr * irho * sw;
    }
    float rwu = ru * sw, rwv = rv * sw;
    resw[o * 2] = rwu;  resw[o * 2 + 1] = rwv;

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b)
            atomicAdd(&Hll[(long long)l * 9 + a * 3 + b],
                      jl[0][a] * jl[0][b] + jl[1][a] * jl[1][b]);
        atomicAdd(&bl[(long long)l * 3 + a],
                  -(jl[0][a] * rwu + jl[1][a] * rwv));
    }
    for (int s2 = 0; s2 < 2; ++s2) {
        const float* J = Jp + (long long)o * 24 + s2 * 12;
        for (int a = 0; a < 6; ++a)
            for (int c = 0; c < 3; ++c)
                Hpl[(long long)o * 36 + s2 * 18 + a * 3 + c] =
                    J[a] * jl[0][c] + J[6 + a] * jl[1][c];
    }
}

/* ---- the LANDMARK prior, LINEAR in all three parameters ----------------
   The measurement is the host keyframe's own observation:

       (x_n, y_n)  the keypoint, information (fx / sigma_px)^2   <- DETECTOR
       rho         the disparity, information (f*B / sigma_d)^2  <- STEREO

   Carrying it as a PRIOR rather than as a residual row is exactly equivalent
   - the host's reprojection residual IS `fx * (a - a0, b - b0)` because
   `m/rho` projects to the keypoint that defined `m` - but it also works when
   the host keyframe sits outside the window, which is the common case.  So
   the structurally zero host rows stay dropped and this carries their
   information instead.

   Every term is a positive constant, so `Hll` is bounded away from
   singularity by construction: the transverse directions get (fx/sigma_px)^2
   and the depth direction (f*B/sigma_d)^2, a condition number of order 20
   against the ~1.6e4 a Euclidean 3x3 block reaches on far structure.       */
__global__ void landmark_prior_id(const float* __restrict__ mb,   /* (nl,3) */
                                  const float* __restrict__ rhov,
                                  const float* __restrict__ m0,   /* (nl,2) */
                                  const float* __restrict__ rho0,
                                  const float* __restrict__ invsig,
                                  const float* __restrict__ wbear,
                                  const float* __restrict__ valid,
                                  float* __restrict__ Hll,
                                  float* __restrict__ bl,
                                  const int nl, const float huber)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nl || valid[l] < 0.5f) return;
    const float wb = wbear[l];
    float d[3] = {(mb[l * 3] - m0[l * 2]) * wb,
                  (mb[l * 3 + 1] - m0[l * 2 + 1]) * wb,
                  (rhov[l] - rho0[l]) * invsig[l]};
    const float J0[3] = {wb, wb, invsig[l]};
    /* * THE BEARING PRIOR IS NOT ROBUSTIFIED, AND THAT IS THE POINT.
       A Huber kernel belongs on a MEASUREMENT that might be an outlier. The
       bearing prior is not that: it is the host keypoint, and it is the ONLY
       thing holding (x_n, y_n), which the reprojections cannot pin on their
       own for a landmark with few views. Down-weighting it as 1/|r| past
       1.345 px therefore RELEASES exactly the bearings that have started to
       drift, and they run away - which showed up as V1_01 going bimodal the
       moment the bearing became a variable: draws of 0.0389 / 0.0413 against
       0.0743 / 0.0771, with the global BA entering at 3.8-5.0 px in the good
       mode and 26.9-27.7 px in the bad one. The one-parameter form never had
       that spread (1.13x over three draws).
       rho keeps its Huber: a disparity genuinely can be an outlier, and rho
       is additionally constrained by multi-view reprojection, so it cannot
       run away when the prior steps back. */
    for (int a = 0; a < 3; ++a) {
        float r = d[a];
        float sw = 1.f;
        if (a == 2) {
            float w = (fabsf(r) <= huber) ? 1.f
                                          : (huber / fmaxf(fabsf(r), 1e-9f));
            sw = sqrtf(w);
        }
        float J = J0[a] * sw;
        atomicAdd(&Hll[(long long)l * 9 + a * 3 + a], J * J);
        atomicAdd(&bl[(long long)l * 3 + a], -J * r * sw);
    }
}

/* ---- pose blocks over TWO slots (observer, host) ------------------------ */
__global__ void pose_blocks_id(const float* __restrict__ Jp,   /* (N,2,2,6) */
                               const float* __restrict__ res,  /* (N,2)     */
                               const int* __restrict__ kslot,  /* (N,2)     */
                               float* __restrict__ Hred,
                               float* __restrict__ g,
                               const int N, const int d, const int n)
{
    /* * ONE THREAD PER (observation, a, b), LOOPING OVER THE SLOT PAIRS.
       The obvious shape is one thread per (o, sa, sb, a, b), which is 4x the
       threads - and in a real run about 82 % of landmarks are hosted by a
       keyframe the window has already slid past, so three quarters of those
       threads launch only to hit `kslot < 0` and return.  On a problem P0
       measured as LAUNCH-BOUND that is the expensive kind of waste: the
       atomics are unchanged, only the launches go away.                    */
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)N * d * d) return;
    int b = (int)(idx % d);  long long t = idx / d;
    int a = (int)(t % d);
    int o = (int)(t / d);
    const float* rr = res + (long long)o * 2;
    for (int sa = 0; sa < 2; ++sa) {
        const int ka = kslot[o * 2 + sa];
        if (ka < 0) continue;
        const float* Ja = Jp + (long long)o * 24 + sa * 12;
        for (int sb = 0; sb < 2; ++sb) {
            const int kb = kslot[o * 2 + sb];
            if (kb < 0) continue;
            const float* Jb = Jp + (long long)o * 24 + sb * 12;
            atomicAdd(&Hred[(long long)(ka * d + a) * n + (kb * d + b)],
                      Ja[a] * Jb[b] + Ja[6 + a] * Jb[6 + b]);
        }
        if (b == 0)
            atomicAdd(&g[ka * d + a], -(Ja[a] * rr[0] + Ja[6 + a] * rr[1]));
    }
}

/* ---- Schur assembly over pairs x slot pairs ----------------------------- */
__global__ void schur_pairs_id(const float* __restrict__ Hpl,
                               const float* __restrict__ Hlli,
                               const int* __restrict__ kslot,
                               const int* __restrict__ olm,
                               const int* __restrict__ pa,
                               const int* __restrict__ pb,
                               float* __restrict__ Hred,
                               const long long npair, const int d, const int n)
{
    /* Same shape as `pose_blocks_id`: one thread per (pair, a, b), looping
       over the slot pairs, so an out-of-window host costs a branch rather
       than three dead launches.                                            */
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= npair * d * d) return;
    int b = (int)(idx % d);  long long t = idx / d;
    int a = (int)(t % d);
    long long p = t / d;
    const int oi = pa[p], oj = pb[p];
    const float* hl = Hlli + (long long)olm[oi] * 9;
    for (int si = 0; si < 2; ++si) {
        const int ki = kslot[oi * 2 + si];
        if (ki < 0) continue;
        const float* hi = Hpl + (long long)oi * 36 + si * 18 + a * 3;
        float t3[3];
        for (int k = 0; k < 3; ++k)
            t3[k] = hi[0] * hl[k] + hi[1] * hl[3 + k] + hi[2] * hl[6 + k];
        for (int sj = 0; sj < 2; ++sj) {
            const int kj = kslot[oj * 2 + sj];
            if (kj < 0) continue;
            const float* hj = Hpl + (long long)oj * 36 + sj * 18 + b * 3;
            float acc = t3[0] * hj[0] + t3[1] * hj[1] + t3[2] * hj[2];
            atomicAdd(&Hred[(long long)(ki * d + a) * n + (kj * d + b)], -acc);
        }
    }
}

/* ---- Schur term on the gradient ---------------------------------------- */
__global__ void schur_grad_id(const float* __restrict__ Hpl,
                              const float* __restrict__ Hlli,
                              const float* __restrict__ bl,
                              const int* __restrict__ kslot,
                              const int* __restrict__ olm,
                              float* __restrict__ g,
                              const int N, const int d)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)N * 2 * d) return;
    int a = (int)(idx % d);  long long t = idx / d;
    int s2 = (int)(t % 2);   t /= 2;
    int o = (int)t;
    const int k = kslot[o * 2 + s2];
    if (k < 0) return;
    const int l = olm[o];
    const float* h = Hpl + (long long)o * 36 + s2 * 18 + a * 3;
    const float* hl = Hlli + (long long)l * 9;
    const float* bb = bl + (long long)l * 3;
    float acc = 0.f;
    for (int c = 0; c < 3; ++c) {
        float tc = h[0] * hl[c] + h[1] * hl[3 + c] + h[2] * hl[6 + c];
        acc += tc * bb[c];
    }
    atomicAdd(&g[k * d + a], -acc);
}

/* ---- landmark back-substitution, scalar, CSR layout --------------------- */
__global__ void backsub_id(const float* __restrict__ Hpl,
                           const float* __restrict__ Hlli,
                           const float* __restrict__ bl,
                           const float* __restrict__ dxp,
                           const int* __restrict__ kslot,
                           const int* __restrict__ offs,
                           float* __restrict__ dxl,
                           const int nl, const int d)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nl) return;
    float rhs[3] = {bl[l * 3], bl[l * 3 + 1], bl[l * 3 + 2]};
    for (int s2 = offs[l]; s2 < offs[l + 1]; ++s2)
        for (int sl = 0; sl < 2; ++sl) {
            const int k = kslot[s2 * 2 + sl];
            if (k < 0) continue;
            const float* h = Hpl + (long long)s2 * 36 + sl * 18;
            for (int c = 0; c < 3; ++c) {
                float acc = 0.f;
                for (int a = 0; a < d; ++a) acc += h[a * 3 + c] * dxp[k * d + a];
                rhs[c] -= acc;
            }
        }
    const float* hl = Hlli + (long long)l * 9;
    for (int c = 0; c < 3; ++c)
        dxl[l * 3 + c] = hl[c * 3] * rhs[0] + hl[c * 3 + 1] * rhs[1]
                       + hl[c * 3 + 2] * rhs[2];
}

}   /* extern "C" */
"""

_MOD = None


def module():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=("-std=c++14",))
    return _MOD


def _grid(total, block=256):
    return (int((total + block - 1) // block),), (block,)


def build_blocks(Tcw, pts, okf, olm, ouv, Jpose, Jlm, resw, Hll, bl, Hpl,
                 N, fx, cx, cy, huber):
    g, b = _grid(N)
    module().get_function("build_blocks")(
        g, b, (Tcw, pts, okf, olm, ouv, Jpose, Jlm, resw, Hll, bl, Hpl,
               int(N), cp.float32(fx), cp.float32(cx), cp.float32(cy),
               cp.float32(huber)))


def depth_prior(pts, chost, rho0, invsig, valid, Hll, bl, nl, huber):
    g, b = _grid(nl)
    module().get_function("depth_prior")(
        g, b, (pts, chost, rho0, invsig, valid, Hll, bl, int(nl),
               cp.float32(huber)))


def inv3_damped(Hll, Hlli, nl, lam):
    g, b = _grid(nl)
    module().get_function("inv3_damped")(g, b, (Hll, Hlli, int(nl),
                                                cp.float32(lam)))


def schur_pairs(Hpl, Hlli, okf, olm, pa, pb, Hred, npair, d, n):
    g, b = _grid(npair * d * d)
    module().get_function("schur_pairs")(g, b, (Hpl, Hlli, okf, olm, pa, pb,
                                                Hred, int(npair), d, n))


def schur_grad(Hpl, Hlli, bl, okf, olm, g_, N, d):
    g, b = _grid(N * d)
    module().get_function("schur_grad")(g, b, (Hpl, Hlli, bl, okf, olm, g_,
                                               int(N), d))


def pose_blocks(Jp, res, kf, Hred, grad, N, r, d, n):
    g, b = _grid(N * d * d)
    module().get_function("pose_blocks")(g, b, (Jp, res, kf, Hred, grad,
                                                N, r, d, n))


_STATUS = None


def chol_solve(H, b, x, n, nthreads=256):
    """Factor and solve in ONE launch.  H is destroyed.

    One block, so the whole factorisation and both triangular solves happen
    without a host round trip.  The status buffer is reused rather than
    allocated per call - at these sizes an allocation is a measurable fraction
    of the kernel.
    """
    global _STATUS
    if _STATUS is None:
        _STATUS = cp.zeros(1, dtype=cp.int32)
    module().get_function("chol_solve")(
        (1,), (nthreads,), (H, b, x, int(n), _STATUS))
    return _STATUS


def backsub(Hpl, Hlli, bl, dxp, okf, offs, dxl, nl, d):
    g, b = _grid(nl)
    module().get_function("backsub")(g, b, (Hpl, Hlli, bl, dxp, okf, offs,
                                            dxl, int(nl), d))


# --------------------------------------------------------------------------
# INVERSE-DEPTH WRAPPERS (plan 5.4).  Same launch discipline as above - one
# kernel per stage, because P0 measured the chain as launch-bound, not
# FLOP-bound.
# --------------------------------------------------------------------------


def build_blocks_id(Tcw, hpose, hidx, mbear, rhov, okf, olm, ouv,
                    Jp, resw, kslot, Hll, bl, Hpl, N, fx, cx, cy, huber):
    g, b = _grid(N)
    module().get_function("build_blocks_id")(
        g, b, (Tcw, hpose, hidx, mbear, rhov, okf, olm, ouv,
               Jp, resw, kslot, Hll, bl, Hpl, int(N), cp.float32(fx),
               cp.float32(cx), cp.float32(cy), cp.float32(huber)))


def landmark_prior_id(mb, rhov, m0, rho0, invsig, wbear, valid, Hll, bl, nl,
                      huber):
    g, b = _grid(nl)
    module().get_function("landmark_prior_id")(
        g, b, (mb, rhov, m0, rho0, invsig, wbear, valid, Hll, bl, int(nl),
               cp.float32(huber)))


def pose_blocks_id(Jp, res, kslot, Hred, grad, N, d, n):
    g, b = _grid(N * d * d)
    module().get_function("pose_blocks_id")(g, b, (Jp, res, kslot, Hred,
                                                   grad, int(N), d, n))


def schur_pairs_id(Hpl, Hlli, kslot, olm, pa, pb, Hred, npair, d, n):
    g, b = _grid(npair * d * d)
    module().get_function("schur_pairs_id")(
        g, b, (Hpl, Hlli, kslot, olm, pa, pb, Hred, int(npair), d, n))


def schur_grad_id(Hpl, Hlli, bl, kslot, olm, g_, N, d):
    g, b = _grid(N * 2 * d)
    module().get_function("schur_grad_id")(g, b, (Hpl, Hlli, bl, kslot, olm,
                                                  g_, int(N), d))


def backsub_id(Hpl, Hlli, bl, dxp, kslot, offs, dxl, nl, d):
    g, b = _grid(nl)
    module().get_function("backsub_id")(g, b, (Hpl, Hlli, bl, dxp, kslot,
                                               offs, dxl, int(nl), d))

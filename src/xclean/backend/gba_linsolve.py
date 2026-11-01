# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""XTEN_GBA_SOLVER: a CONVERGED step for the run-end global BA above the size SuperLU is validated on.

    dx, info = solve(H, bp, T_cw, vel, free, nk, n_pose, mode="dpcg", rtol=1e-8, deadline=None)

THE DEFECT THIS REPAIRS
    global_ba_id hands every damped reduced system above 30,000 unknowns (madmax_B0, madmax_C2, rosario_01 / 06,
    rosariov2_01 and every 4Seasons drive) to scipy's CG with a plain Jacobi preconditioner, maxiter 2000, and uses the
    result WHETHER OR NOT CG CONVERGED (cg_info is only checked together with a non-finite output).  On 4s_120 (90k
    unknowns) CG stops at the cap on 9 of 10 pass-1 iterations; the GBA then moves the 4 km trajectory by 0.12 m rms and
    its Sim(3) scale by 1e-5 while the reprojection rms goes 6.27 -> 0.91 px.  A
    Jacobi-preconditioned CG resolves a mode at a rate set by its eigenvalue relative to the diagonal, and the
    long-wavelength modes of a keyframe chain - world scale, slow bending, heading and tilt drift - have the smallest
    eigenvalues; 2000 iterations leave them where the online estimator put them.  Measured on rosario_01 (pass 1, LM
    iteration 5, 32.9k unknowns): the landed CG ends at relative residual 1.2e-2 and needs 7804 Jacobi iterations to reach
    1e-6; per-keyframe block-Jacobi PCG stagnates at 1.1e-3 and the segment smoother without the coarse space at 2.7e-4;
    this two-level solve reaches 6.6e-9 in 196 iterations.

THE FIX (tier 1: the SAME damped SPD system, solved to a convergence criterion; no model change, no new weight)
    Deflated preconditioned conjugate gradients (Saad, Yeung, Erhel & Guyomarc'h, SIAM J. Sci. Comput. 21 (2000);
    Tang, Nabben, Vuik & Erlangga, J. Sci. Comput. 39 (2009)), two-level in the domain-decomposition sense:
      * the keyframes are cut into contiguous SEGMENTS of m keyframes;
      * preconditioner M: the exact inverse of every segment's own diagonal block of H (all its pose unknowns and, with
        the inertial term, its velocities) and of the block of the global unknowns (biases, gravity direction, beta):
        everything inside a segment is solved exactly, given its neighbours;
      * deflation space Z: what is left are the segments' FLOATING modes - every segment may move as a rigid body in the
        world frame (6), change its scale (1) and, with the inertial term, offset its world velocities (3); every global
        unknown is its own coarse unknown.  The world scale and the global rigid motions lie in span(Z) exactly; slow
        bending / heading / tilt drift to within one segment.  The modes are the linearised world motions in the
        solver's own chart (T_cw <- exp(d) T_cw, d = [rho, phi]):
            rigid  xi:  d_k = -Ad(T_cw_k) xi ,   dv_k = phi x v_k
            scale  s :  d_k = s [t_cw_k ; 0] ,   dv_k = s v_k
      * the coarse operator E = Z^T H Z is dense and factorised exactly (LAPACK Cholesky, Jacobi-scaled);
      * the segment length is not tuned: m = (2 c^2 nk / b^2)^(1/3) minimises the preconditioner's stored numbers
        b^2 m nk (segment blocks, b unknowns per keyframe) + c^2 (nk/m)^2 (coarse matrix, c modes per segment), which is
        also its per-iteration work; at least 2 keyframes so a segment's scale is independent of its translation;
      * stopping rule: ||bp - H dx|| <= rtol ||bp|| on the TRUE residual, rtol = the landed 1e-8.  A safety cap on the
        iteration count (MAXITER, a resource bound far above every measured solve) and the GBA's own wall-clock deadline
        stop it otherwise; the step is then REPORTED unconverged (info["converged"] False with the reason) and the caller
        does not use it.  The true residual is checked every CHECK_EVERY iterations; stagnation (no new smallest true
        residual over STALL_CHECKS checks) or divergence (a decade above the smallest) stop it at once, reported.

    Memory: H is read only; the extras are the segment blocks (b^2 m nk doubles), H @ Z (n x nc sparse), E (nc^2 doubles)
    and O(n) vectors - no fill-in, nothing that grows with a factorisation of H itself.
"""
from __future__ import annotations

import os
import time

import numpy as np

GBA_SOLVER_ENV = "XTEN_GBA_SOLVER"
GBA_SOLVER_MODES = ("dpcg",)
D = 6
# the landed CG's tolerance, kept: the fix is to REACH it, not to move it
RTOL = 1e-8
# a resource bound, not a tuning knob: the landed Jacobi CG's own cap, about 9x above the largest DPCG count measured on
# the CG-path dumps; hitting it is REPORTED and the step is NOT used (the caller
# raises the damping x10, which shortens the next solve) - so the cap bounds the run-end wall time of a pathological
# solve instead of leaving it open (20,000 in the 2026-10-03 draft)
MAXITER = 2_000
# the true residual is recomputed every CHECK_EVERY iterations (one extra mat-vec in 50: 2 % of the work) - the
# recursion alone can drift below the truth, or run on after the attainable accuracy is reached
CHECK_EVERY = 50
# stagnation: this many consecutive true-residual checks (STALL_CHECKS x CHECK_EVERY iterations) without a new smallest
# true residual - the attainable accuracy of the floating-point recursion is reached (INFERENCE; the 1e-11 reference
# solves on 4s_20210107_120 stop this way, by divergence of the true residual, at 1.4e-10 / 1.7e-10, in our solver study)
STALL_CHECKS = 4
# P6r3: a block (or the coarse matrix) whose Cholesky fails is accepted as positive SEMI-definite to rounding when its
# most negative eigenvalue is within PSD_TOL of its largest - sqrt(eps), the relative accuracy to which a computed
# eigenvalue near zero of a rounded symmetric matrix can be told from zero in double precision (not tuned); anything
# more negative is indefinite and raises (reported at setup, the step is not used)
PSD_TOL = float(np.sqrt(np.finfo(float).eps))


def _psd_eig(A, what):
    w, V = np.linalg.eigh(A)
    top = float(w.max())
    if not (top > 0.0) or float(w.min()) < -PSD_TOL * top:
        raise np.linalg.LinAlgError("%s is not positive semi-definite (eigenvalues %.3e .. %.3e)"
                                    % (what, float(w.min()), top))
    return w, V, top


def _inv_spd_stack(B):
    """(k, s, s) symmetric blocks -> (their inverses, how many went through the eigen route).  Batched Cholesky first
    (the 2026-10-03 arithmetic, unchanged when every block is positive definite); otherwise block by block, and a block
    whose Cholesky fails is inverted through its eigen-decomposition with the eigenvalues floored at PSD_TOL x the
    largest - an SPD preconditioner block - provided it is PSD to rounding (_psd_eig raises otherwise)."""
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        L = None
    if L is not None:
        Li = np.linalg.inv(L)
        del L
        return np.matmul(np.swapaxes(Li, 1, 2), Li), 0
    out = np.empty_like(B)
    nf = 0
    for i in range(B.shape[0]):
        try:
            Li = np.linalg.inv(np.linalg.cholesky(B[i]))
            out[i] = Li.T @ Li
        except np.linalg.LinAlgError:
            w, V, top = _psd_eig(B[i], "block %d" % i)
            out[i] = (V / np.maximum(w, PSD_TOL * top)) @ V.T
            nf += 1
    return out, nf


def gba_solver_mode(env=None):
    """XTEN_GBA_SOLVER: unset / "" / "0" / "off" -> None (the landed CG path); "dpcg" -> the converged two-level
    deflated PCG; anything else raises, so a misspelt queue flag fails at construction rather than after a run."""
    v = (os.environ if env is None else env).get(GBA_SOLVER_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off"):
        return None
    if v in GBA_SOLVER_MODES:
        return v
    raise RuntimeError("%s=%r is not a mode: unset/0/off or %s"
                       % (GBA_SOLVER_ENV, v, " / ".join(GBA_SOLVER_MODES)))


# ---------------------------------------------------------------------------------------------------------------
# layout and segments
# ---------------------------------------------------------------------------------------------------------------
def layout(n, nk, n_pose, inertial):
    """-> (kf_idx (nk, b) the unknowns of every keyframe, glob (ng,) the global unknowns).

    Pose of keyframe k: 6k..6k+5; with the inertial term its velocity: n_pose + 3k..+2; everything after the
    per-keyframe unknowns (gyro / accel bias, gravity direction, beta, any further nuisance) is global."""
    pose = np.arange(nk)[:, None] * D + np.arange(D)[None, :]
    if inertial:
        velo = n_pose + np.arange(nk)[:, None] * 3 + np.arange(3)[None, :]
        kf_idx = np.concatenate([pose, velo], 1)
        g0 = n_pose + 3 * nk
    else:
        kf_idx = pose
        g0 = n_pose
    if g0 > n:
        raise ValueError("layout does not fit the system (%d > %d)" % (g0, n))
    return kf_idx.astype(np.int64), np.arange(g0, n, dtype=np.int64)


def segment_length(nk, b, c):
    """keyframes per segment: argmin over m of b^2 m nk + c^2 (nk / m)^2, i.e. m = (2 c^2 nk / b^2)^(1/3), at least 2."""
    return max(2, int(round((2.0 * c * c * nk / float(b * b)) ** (1.0 / 3.0))))


def segment_bounds(nk, m):
    nseg = max(1, int(round(nk / float(m))))
    return np.unique(np.linspace(0, nk, nseg + 1).round().astype(np.int64))


# ---------------------------------------------------------------------------------------------------------------
# the smoother: exact inverses of the segments' diagonal blocks
# ---------------------------------------------------------------------------------------------------------------
class SegmentJacobi:
    """z = M^-1 r, M = blockdiag(segment blocks of H, global block of H), each inverted exactly through its Cholesky
    factor (M^-1 = L^-T L^-1, symmetric positive definite by construction; a block that is not numerically positive
    definite raises LinAlgError, which the caller reports)."""

    def __init__(self, H, kf_idx, glob, bounds):
        n = H.shape[0]
        nk, b = kf_idx.shape
        nseg = len(bounds) - 1
        sizes = (bounds[1:] - bounds[:-1]) * b
        smax = int(sizes.max())
        blk = np.full(n, -1, np.int64)
        pos = np.full(n, -1, np.int64)
        idx_pad = np.full((nseg, smax), n, np.int64)       # n = the zero slot of the extended vector
        for s in range(nseg):
            idx = kf_idx[bounds[s]:bounds[s + 1]].reshape(-1)
            blk[idx] = s
            pos[idx] = np.arange(len(idx))
            idx_pad[s, :len(idx)] = idx
        ng = len(glob)
        blk[glob] = nseg
        pos[glob] = np.arange(ng)
        if (blk < 0).any():
            raise ValueError("unknowns outside every block: layout does not match the system")
        B = np.zeros((nseg, smax, smax))
        Bg = np.zeros((ng, ng))
        ip, ix, dt = H.indptr, H.indices, H.data
        per_row = max(1, int(H.nnz // max(n, 1)))
        step = max(1, 4_000_000 // per_row)
        for r0 in range(0, n, step):
            r1 = min(n, r0 + step)
            a, e = int(ip[r0]), int(ip[r1])
            rows = np.repeat(np.arange(r0, r1, dtype=np.int64), np.diff(ip[r0:r1 + 1]).astype(np.int64))
            cols = ix[a:e].astype(np.int64)
            br = blk[rows]
            m = br == blk[cols]
            if not m.any():
                continue
            rr, cc, vv, bb = rows[m], cols[m], dt[a:e][m], br[m]
            g = bb == nseg
            k = ~g
            # canonical CSR: every (row, col) pair once, so assignment equals accumulation
            B[bb[k], pos[rr[k]], pos[cc[k]]] = vv[k]
            if g.any():
                Bg[pos[rr[g]], pos[cc[g]]] = vv[g]
        for s in range(nseg):                               # padding rows of a shorter segment: identity
            if sizes[s] < smax:
                j = np.arange(sizes[s], smax)
                B[s, j, j] = 1.0
        B = 0.5 * (B + np.swapaxes(B, 1, 2))
        self.Binv, nf = _inv_spd_stack(B)            # P6r3: PSD-to-rounding blocks floored, indefinite raises
        del B
        if ng:
            Bg = 0.5 * (Bg + Bg.T)
            Bginv, nfg = _inv_spd_stack(Bg[None])
            self.Bginv = Bginv[0]
            nf += nfg
        else:
            self.Bginv = np.zeros((0, 0))
        self.n_floored = int(nf)
        self.idx_pad, self.glob, self.n = idx_pad, glob, n
        self.valid = idx_pad < n

    def __call__(self, r):
        re = np.empty(self.n + 1)
        re[:self.n] = r
        re[self.n] = 0.0
        zp = np.matmul(self.Binv, re[self.idx_pad][:, :, None])[:, :, 0]
        z = np.empty(self.n)
        z[self.idx_pad[self.valid]] = zp[self.valid]
        if len(self.glob):
            z[self.glob] = self.Bginv @ r[self.glob]
        return z


# ---------------------------------------------------------------------------------------------------------------
# the deflation space: the segments' floating modes
# ---------------------------------------------------------------------------------------------------------------
def _hat(v):
    z = np.zeros(v.shape[:-1] + (3, 3))
    z[..., 0, 1] = -v[..., 2]
    z[..., 0, 2] = v[..., 1]
    z[..., 1, 0] = v[..., 2]
    z[..., 1, 2] = -v[..., 0]
    z[..., 2, 0] = -v[..., 1]
    z[..., 2, 1] = v[..., 0]
    return z


def mode_vectors(T_cw, vel, free, inertial):
    """per keyframe, the world modes in the solver's chart: (nk, b, c) with c = 7 (+3 velocity offsets with the
    inertial term): columns [translation x3, rotation x3, scale, (velocity offset x3)], rows [pose 6, (velocity 3)].
    Rows of fixed poses are zero (their unknowns are the identity rows of the gauge fix)."""
    nk = len(T_cw)
    R = T_cw[:, :3, :3]
    t = T_cw[:, :3, 3]
    c = 10 if inertial else 7
    b = 9 if inertial else 6
    M = np.zeros((nk, b, c))
    # d_k = -Ad(T_cw_k) xi,  Ad = [[R, [t]x R], [0, R]]  ([rho, phi] order, core/lie.adjoint)
    M[:, 0:3, 0:3] = -R
    M[:, 0:3, 3:6] = -np.einsum("kij,kjl->kil", _hat(t), R)
    M[:, 3:6, 3:6] = -R
    M[:, 0:3, 6] = t
    M[~np.asarray(free, bool), 0:6, :] = 0.0
    if inertial:
        v = np.asarray(vel, float)
        M[:, 6:9, 3:6] = -_hat(v)          # dv = phi x v = -[v]x phi
        M[:, 6:9, 6] = v
        M[:, 6:9, 7:10] = np.eye(3)
    return M


def deflation_space(T_cw, vel, free, kf_idx, glob, n, inertial, bounds):
    """Z (n x nc, CSR): per segment the orthonormalised span of its keyframes' mode vectors (rank-revealing SVD: a
    direction below the matrix's own rounding level is dropped), plus one unit column per global unknown."""
    import scipy.sparse as sp
    M = mode_vectors(T_cw, vel, free, inertial)
    rows, cols, vals = [], [], []
    col0 = 0
    eps = np.finfo(float).eps
    for s in range(len(bounds) - 1):
        k0, k1 = int(bounds[s]), int(bounds[s + 1])
        Lm = M[k0:k1].reshape(-1, M.shape[2])
        ridx = kf_idx[k0:k1].reshape(-1)
        nzr = np.abs(Lm).sum(1) > 0
        if not nzr.any():
            continue
        U, S, _ = np.linalg.svd(Lm[nzr], full_matrices=False)
        r = int(np.sum(S > S[0] * max(Lm[nzr].shape) * eps)) if S.size and S[0] > 0 else 0
        if r == 0:
            continue
        rows.append(np.repeat(ridx[nzr], r))
        cols.append(np.tile(np.arange(col0, col0 + r), int(nzr.sum())))
        vals.append(U[:, :r].reshape(-1))
        col0 += r
    ng = len(glob)
    if ng:
        rows.append(glob)
        cols.append(np.arange(col0, col0 + ng))
        vals.append(np.ones(ng))
        col0 += ng
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, col0))


class Coarse:
    """E = Z^T H Z (dense) and its exact Cholesky factor (Jacobi-scaled); keeps Z^T H = (H Z)^T for the projections."""

    def __init__(self, H, Z):
        import scipy.linalg as sla
        HZ = (H @ Z).tocsr()
        self.Z = Z
        self.Zt = Z.T.tocsr()
        self.ZtH = HZ.T.tocsr()
        E = (self.Zt @ HZ).toarray()
        del HZ
        E = 0.5 * (E + E.T)
        d = np.sqrt(np.maximum(np.diag(E), 1e-300))
        E /= d[:, None]
        E /= d[None, :]
        self.d = d
        self.nc = E.shape[0]
        self.V = None
        self.n_dropped = 0
        try:
            self.cf = sla.cho_factor(E, lower=True, overwrite_a=False, check_finite=False)
        except np.linalg.LinAlgError:
            # P6r3: singular to rounding (a mode of a keyframe without vision rows): the pseudo-inverse, which keeps
            # Z E^+ Z^T H a projection onto the resolvable part of span Z; indefinite raises
            w, V, top = _psd_eig(E, "the coarse matrix")
            keep = w > PSD_TOL * top
            self.V, self.winv = V[:, keep], 1.0 / w[keep]
            self.n_dropped = int((~keep).sum())
            self.cf = None

    def solve(self, y):
        import scipy.linalg as sla
        if self.V is not None:
            return (self.V @ (self.winv * (self.V.T @ (y / self.d)))) / self.d
        return sla.cho_solve(self.cf, y / self.d, check_finite=False) / self.d

    def project(self, z):
        """z - Z E^-1 Z^T H z  (H-orthogonal to span Z)."""
        return z - self.Z @ self.solve(self.ZtH @ z)

    def coarse_part(self, r):
        """Z E^-1 Z^T r."""
        return self.Z @ self.solve(self.Zt @ r)


# ---------------------------------------------------------------------------------------------------------------
# the iteration
# ---------------------------------------------------------------------------------------------------------------
def _pcg(mv, b, Minv, C, rtol, maxiter, deadline):
    """deflated preconditioned CG on H x = b to ||b - H x|| <= rtol ||b|| on the TRUE residual.

    x0 = Z E^-1 Z^T b makes r0 orthogonal to Z, and every search direction is made H-orthogonal to Z
    (p = P z + beta p, P z = z - Z E^-1 Z^T H z), so the residuals stay orthogonal to Z and the coarse modes never have
    to be found by the iteration (Saad et al. 2000, preconditioned).

    The recursive residual is never trusted on its own: the TRUE residual is computed every CHECK_EVERY iterations,
    whenever the recursion falls a decade below the smallest true residual so far, and when it reaches the tolerance;
    convergence is declared on the true residual alone.
      * a recursion more than a decade below the truth (or at the tolerance while the truth is not) is restarted from
        the true residual (residual replacement, re-projected onto Z's complement);
      * the iterate with the smallest true residual is kept; STALL_CHECKS consecutive checks without a new smallest
        true residual is stagnation (the attainable accuracy of the floating-point recursion), a true residual a
        decade above the smallest is divergence - both stop at once and return the best iterate, REPORTED
        unconverged (the engine does not use it)."""
    nb = float(np.linalg.norm(b))
    info = {"iters": 0, "matvecs": 0, "restarts": 0, "reason": None}
    if nb == 0.0 or not np.isfinite(nb):
        ok = nb == 0.0
        info.update(converged=ok, relres=0.0 if ok else float("nan"), reason="zero_rhs" if ok else "nonfinite_rhs")
        return np.zeros_like(b), info
    tol = rtol * nb
    x = C.coarse_part(b) if C is not None else np.zeros_like(b)
    r = b - mv(x)
    info["matvecs"] += 1
    best = float(np.linalg.norm(r))
    if best <= tol:
        info.update(converged=True, relres=best / nb, reason="tol")
        return x, info
    x_best = x.copy()
    stall = 0
    it = 0
    z = Minv(r)
    p = C.project(z) if C is not None else z.copy()
    rz = float(r @ z)
    while True:
        if it >= maxiter:
            why = "maxiter"
        elif deadline is not None and time.perf_counter() > deadline:
            why = "deadline"
        else:
            why = None
        if why is not None:
            info.update(iters=it, converged=False, relres=best / nb, reason=why)
            return x_best, info
        w = mv(p)
        info["matvecs"] += 1
        it += 1
        pw = float(p @ w)
        if not (pw > 0.0) or not np.isfinite(pw):
            info.update(iters=it, converged=False, relres=best / nb, reason="breakdown")
            return x_best, info
        alpha = rz / pw
        x += alpha * p
        r -= alpha * w
        rn = float(np.linalg.norm(r))
        if rn <= tol or rn <= 0.1 * best or it % CHECK_EVERY == 0:
            rt = b - mv(x)
            info["matvecs"] += 1
            true = float(np.linalg.norm(rt))
            if true <= tol:
                info.update(iters=it, converged=True, relres=true / nb, reason="tol")
                return x, info
            if true < best:
                best, stall = true, 0
                x_best = x.copy()
            else:
                stall += 1
            if true > 10.0 * best:
                info.update(iters=it, converged=False, relres=best / nb, reason="divergence")
                return x_best, info
            if stall >= STALL_CHECKS:
                info.update(iters=it, converged=False, relres=best / nb, reason="stagnation")
                return x_best, info
            if rn <= tol or true > 10.0 * rn:
                # residual replacement: restart from the true residual of the current iterate
                info["restarts"] += 1
                r = rt
                if C is not None:
                    x = x + C.coarse_part(r)
                    r = b - mv(x)
                    info["matvecs"] += 1
                z = Minv(r)
                p = C.project(z) if C is not None else z.copy()
                rz = float(r @ z)
                continue
        z = Minv(r)
        rz_new = float(r @ z)
        beta = rz_new / rz
        rz = rz_new
        p = (C.project(z) if C is not None else z) + beta * p


def solve(H, bp, T_cw, vel, free, nk, n_pose, mode="dpcg", rtol=RTOL, maxiter=None, deadline=None, matvec=None,
          seg_len=None):
    """The step of the damped reduced system H dx = bp (H: the CSR matrix global_ba_id hands its solver, gauge rows
    already identity).  -> (dx, info).  info["converged"] False means dx must NOT be used (info["reason"] says why).
    `seg_len` overrides the derived segment length (diagnostics only; the engine never passes it)."""
    t0 = time.perf_counter()
    maxiter = MAXITER if maxiter is None else int(maxiter)
    n = H.shape[0]
    inertial = vel is not None
    mv = matvec if matvec is not None else (lambda x: H @ x)
    info = {"mode": mode, "n": int(n), "nnz": int(H.nnz)}
    try:
        if mode != "dpcg":
            raise ValueError("unknown solver mode %r" % (mode,))
        kf_idx, glob = layout(n, nk, n_pose, inertial)
        b, c = kf_idx.shape[1], (10 if inertial else 7)
        m = int(seg_len) if seg_len else segment_length(nk, b, c)
        bounds = segment_bounds(nk, m)
        Minv = SegmentJacobi(H, kf_idx, glob, bounds)
        Z = deflation_space(T_cw, vel, free, kf_idx, glob, n, inertial, bounds)
        C = Coarse(H, Z)
        info.update(seg_len=int(m), segments=int(len(bounds) - 1), nc=int(C.nc),
                    floored_blocks=int(Minv.n_floored), coarse_dropped=int(C.n_dropped))
    except (np.linalg.LinAlgError, ValueError, MemoryError) as e:
        info.update(converged=False, reason="setup %s: %s" % (type(e).__name__, e), iters=0, matvecs=0, restarts=0,
                    relres=float("nan"), t_setup=time.perf_counter() - t0, t_iter=0.0)
        return np.zeros(n), info
    t1 = time.perf_counter()
    x, it = _pcg(mv, np.asarray(bp, float), Minv, C, rtol, maxiter, deadline)
    info.update(it)
    info.update(t_setup=t1 - t0, t_iter=time.perf_counter() - t1)
    return x, info

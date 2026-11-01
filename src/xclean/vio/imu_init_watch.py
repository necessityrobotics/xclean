#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.1: IMU-FIRST INITIALISATION (D1).  Pure functions, numpy /
scipy, float64, CPU.  vi_init.py is read, never edited (its own track).

D1 STATIONARITY.  A platform is stationary when the MEAN of its
IMU signals does not move.  Over the last W frame intervals (W = the window length, class B), the samples of each
interval form a sub-window (the camera period: class A); per axis (3 gyro + 3 accelerometer) the one-way ANOVA
statistic
        F = [sum_k n_k (m_k - m)^2 / (K - 1)] / [sum_k sum_i (x_ki - m_k)^2 / (N - K)]
compares the scatter of the sub-window means with what the within-sub-window scatter (MEASURED, class C) predicts.
Stationary when every axis passes F <= F_{K-1, N-K}(1 - (1 - P_GATE) / 6) (class-D level, Bonferroni over 6 axes).
  Why not the declared white density: on v2_03 the motors run while the MAV sits; the accelerometer scatter is 4-19x and
  the gyro 3-30x the declared density, yet the means hold (gz 0.0816 +- 0.0005 rad/s over 4.4 s).
  Caveat (stated): autocorrelated vibration inside a sub-window can make the F test anti-conservative; on the dev rows
  it is not (v2_03 stationary F 0.38-0.97 vs >= 4.8 at take-off).
The state at admission (the IMU's own estimate; v = 0 while stationary):
  bg = the mean gyro;  g_w = -G_STD * R_wb a_mean / |a_mean|  (|g| = G_STD, vi_init's constant);
  ba = (|a_mean| - G_STD) a_mean / |a_mean|  (the component along gravity; the horizontal components are 0 given g_w,
       because with g_w fixed ba = a_mean + R^T g_w exactly at rest);
  covariances from the MEASURED frame-interval noise MSB (the ANOVA mean square between sub-windows = the white-
  equivalent variance that survives integration over a frame interval): Cov(bg) = diag(MSB_g) / N, Cov(ba) =
  diag(MSB_a) / N (the grand mean of N samples), Cov(v) = diag(MSB_a) T / f (the velocity random walk over the window
  span T, density^2 = MSB / f), all class C.

D3 BIAS-RATE TEST: DROPPED and REMOVED (D3 rule, arm A: it demoted mh_04 and v1_02, rows where VI beats
VO).  D3 keeps only the window's whitened IMU chi2 test (engine_live._watchdog).
"""
from __future__ import annotations

import math

import numpy as np

P_GATE = 0.999
G_STD = 9.80665


def f_ppf(p, dfn, dfd):
    from scipy.special import fdtri
    return float(fdtri(float(dfn), float(dfd), float(p)))


def chi2_ppf(p, dof):
    from scipy.special import gammaincinv
    return float(2.0 * gammaincinv(0.5 * dof, p))


def stationarity(t, gyro, acc, horizon, frame_t, p=P_GATE):
    """One-way ANOVA stationarity test over the frame intervals frame_t[k] < t <= frame_t[k+1] (causal: samples with
    t > horizon are never read).  -> dict(ok, F (6,), thr, K, N, mean_g, mean_a, s2_g, s2_a, T) or ok False with
    a reason."""
    ft = np.asarray(frame_t, float)
    if len(ft) < 3:
        return {"ok": False, "why": "fewer than 2 frame intervals"}
    t = np.asarray(t, float)
    lo = int(np.searchsorted(t, ft[0], side="right"))
    hi = int(np.searchsorted(t, min(float(ft[-1]), float(horizon)), side="right"))
    if hi - lo < 4:
        return {"ok": False, "why": "too few samples"}
    ts = t[lo:hi]
    X = np.concatenate([np.asarray(gyro, float)[lo:hi], np.asarray(acc, float)[lo:hi]], axis=1)   # (N, 6)
    lab = np.searchsorted(ft, ts, side="left") - 1          # interval index: ft[k] < t <= ft[k+1]
    K = len(ft) - 1
    ok_lab = (lab >= 0) & (lab < K)
    X = X[ok_lab]
    lab = lab[ok_lab]
    n_k = np.bincount(lab, minlength=K).astype(float)
    use = n_k >= 2
    if use.sum() < 2:
        return {"ok": False, "why": "fewer than 2 sub-windows with >= 2 samples"}
    keep = use[lab]
    X = X[keep]
    lab = lab[keep]
    # re-index the kept sub-windows
    kidx = np.cumsum(use) - 1
    lab = kidx[lab]
    Kk = int(use.sum())
    n_k = np.bincount(lab, minlength=Kk).astype(float)
    N = float(len(X))
    sums = np.zeros((Kk, 6))
    np.add.at(sums, lab, X)
    m_k = sums / n_k[:, None]
    m = X.mean(axis=0)
    ssb = (n_k[:, None] * (m_k - m) ** 2).sum(axis=0)
    ssw = ((X - m_k[lab]) ** 2).sum(axis=0)
    dfn, dfd = Kk - 1, N - Kk
    if dfd < 1:
        return {"ok": False, "why": "no within-sub-window dof"}
    msw = ssw / dfd
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(msw > 0, (ssb / dfn) / msw, np.inf)
    thr = f_ppf(1.0 - (1.0 - p) / 6.0, dfn, dfd)
    T = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    msb = ssb / dfn
    return {"ok": bool(np.all(F <= thr)), "F": F, "thr": thr, "K": Kk, "N": int(N), "mean_g": m[:3].copy(),
            "mean_a": m[3:].copy(), "s2_g": msw[:3].copy(), "s2_a": msw[3:].copy(), "s2b_g": msb[:3].copy(),
            "s2b_a": msb[3:].copy(), "T": T, "rate": (N - 1.0) / T if T > 0 else float("nan")}


def imu_first_state(R_wb, st, g_mag=G_STD):
    """The admission state from a stationary window (module doc).  R_wb: the body attitude in the window's world
    (constant while stationary).  -> dict(g_w, v, bg, ba, Hb (15x15 information on [dphi dp dv dbg dba]; pose rows 0))."""
    a = np.asarray(st["mean_a"], float)
    na = float(np.linalg.norm(a))
    u = a / na
    g_w = -g_mag * (np.asarray(R_wb, float) @ u)
    ba = (na - g_mag) * u
    bg = np.asarray(st["mean_g"], float).copy()
    N = float(st["N"])
    f = float(st["rate"]) if np.isfinite(st["rate"]) and st["rate"] > 0 else N / max(st["T"], 1e-9)
    # the noise that survives integration over a frame interval is what the sub-window MEANS show: the white-equivalent
    # variance n * var(m_k) = the ANOVA mean square between (MSB); within-interval vibration that averages out (F < 1
    # on v2_03) does not reach the preintegrated factors.  Var(grand mean) = MSB / N.
    var_bg = np.asarray(st["s2b_g"], float) / N
    var_ba = np.asarray(st["s2b_a"], float) / N
    var_v = np.asarray(st["s2b_a"], float) * max(float(st["T"]), 1.0 / f) / f
    Hb = np.zeros((15, 15))
    Hb[6:9, 6:9] = np.diag(1.0 / np.maximum(var_v, 1e-300))
    Hb[9:12, 9:12] = np.diag(1.0 / np.maximum(var_bg, 1e-300))
    Hb[12:15, 12:15] = np.diag(1.0 / np.maximum(var_ba, 1e-300))
    return {"g_w": g_w, "v": np.zeros(3), "bg": bg, "ba": ba, "Hb": Hb,
            "sd_v": np.sqrt(var_v), "sd_bg": np.sqrt(var_bg), "sd_ba": np.sqrt(var_ba)}

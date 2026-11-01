# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The offline scorer (xclean/eval/score.py) on synthetic trajectories with known answers."""
import json

import numpy as np
import pytest

from xclean.core.lie import so3_exp
from xclean.eval.score import associate, load_gt, quat_to_R, score, umeyama

T0_NS = 1_600_000_000_000_000_000


def _rot_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.RandomState(0)
    src = rng.standard_normal((200, 3)) * 3.0
    R = so3_exp(np.array([0.3, -1.1, 0.7]))
    t = np.array([1.0, -2.0, 0.5])
    s = 1.37
    dst = s * src @ R.T + t
    R2, t2, s2 = umeyama(src, dst, with_scale=True)
    assert np.abs(R2 - R).max() < 1e-12 and np.abs(t2 - t).max() < 1e-12 and abs(s2 - s) < 1e-12
    R3, t3, s3 = umeyama(src, (dst - t) / s + t, with_scale=False)    # rigid only
    assert s3 == 1.0 and np.abs(R3 - R).max() < 1e-12


def test_umeyama_keeps_a_right_handed_rotation_on_a_planar_set():
    rng = np.random.RandomState(1)
    src = np.c_[rng.standard_normal((50, 2)), np.zeros(50)]
    dst = src * np.array([-1.0, 1.0, 1.0])        # mirrored in its plane = a rotation by pi about the y axis
    R, t, _s = umeyama(src, dst, with_scale=False)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12    # a proper rotation, never the reflection diag(-1, 1, 1)
    assert np.abs(src @ R.T + t - dst).max() < 1e-12


def test_quat_to_R_matches_scipy():
    from scipy.spatial.transform import Rotation
    rng = np.random.RandomState(2)
    q = rng.standard_normal((20, 4))                                  # w, x, y, z, not normalised
    R = quat_to_R(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    Rs = Rotation.from_quat(np.c_[q[:, 1:], q[:, 0]]).as_matrix()
    assert np.abs(R - Rs).max() < 1e-12


def test_associate_respects_max_dt():
    ia, ib, dt = associate(np.array([0.0, 1.0, 2.0, 3.5]), np.array([3.0, 0.005, 1.03, 2.0]), 0.02)
    assert ia.tolist() == [0, 2] and ib.tolist() == [1, 3] and float(dt.max()) <= 0.02


def _write_sequence(tmp_path, t_bc):
    """GT body poses on a helix (rotating body), a calib with a lever arm t_bc; returns (seq dir, t, p_cam, p_body)."""
    seq = tmp_path / "seq"
    seq.mkdir()
    T_bc = np.eye(4)
    T_bc[:3, 3] = t_bc
    (seq / "calib.json").write_text(json.dumps({"t0_ns": T0_NS, "T_body_cam0": T_bc.tolist(), "dataset": "synthetic"}))
    t = np.arange(0.0, 30.0, 0.05)                                   # 600 poses, 20 Hz
    p_body = np.c_[3 * np.cos(0.3 * t), 3 * np.sin(0.3 * t), 0.1 * t]
    R_body = np.stack([so3_exp(np.array([0.0, 0.1 * np.sin(0.2 * ti), 0.3 * ti])) for ti in t])
    q = np.stack([_rot_to_quat_wxyz(R) for R in R_body])
    with open(seq / "gt.csv", "w") as f:
        f.write("#t_ns,x,y,z,qw,qx,qy,qz\n")
        for ti, pi, qi in zip(t, p_body, q):
            f.write("%d,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f\n" % ((T0_NS + int(round(ti * 1e9))), *pi, *qi))
    p_cam = p_body + np.einsum("nij,j->ni", R_body, np.asarray(t_bc, float))
    return seq, t, p_cam, p_body


def _write_traj(path, t, p):
    with open(path, "w") as f:
        f.write("# t,x,y,z,qx,qy,qz,qw\n")
        for ti, pi in zip(t, p):
            f.write("%.9f,%.12f,%.12f,%.12f,0,0,0,1\n" % (ti, *pi))


def test_score_rigidly_moved_truth_has_zero_ate(tmp_path):
    seq, t, p_cam, _ = _write_sequence(tmp_path, [0.1, -0.05, 0.02])
    R = so3_exp(np.array([0.2, 0.4, -1.0]))
    est = p_cam @ R.T + np.array([5.0, -3.0, 1.0])                   # the estimator's own world frame
    _write_traj(tmp_path / "traj.csv", t, est)
    r = score(str(tmp_path / "traj.csv"), str(seq), tag="synthetic")
    assert r.n_assoc == len(t) and r.coverage == 1.0
    assert r.ate_se3.rmse < 1e-8 and r.ate_sim3.rmse < 1e-8 and abs(r.sim3_scale - 1.0) < 1e-9
    assert abs(r.path_ratio - 1.0) < 1e-9 and r.verdict_blockers() == []


def test_score_scale_error_is_seen_by_se3_and_removed_by_sim3(tmp_path):
    seq, t, p_cam, _ = _write_sequence(tmp_path, [0.0, 0.0, 0.0])
    _write_traj(tmp_path / "traj.csv", t, 1.10 * p_cam)              # 10 % too long
    r = score(str(tmp_path / "traj.csv"), str(seq))
    assert r.ate_se3.rmse > 0.1
    assert r.ate_sim3.rmse < 1e-8
    assert r.sim3_scale == pytest.approx(1 / 1.10, abs=1e-9)
    assert r.path_ratio == pytest.approx(1.10, abs=1e-9)


def test_score_known_constant_offset_with_lever_arm(tmp_path):
    """frame='cam0' moves GT through T_body_cam0; frame='body' must then see exactly the lever-arm error."""
    t_bc = [0.3, 0.0, 0.0]
    seq, t, p_cam, p_body = _write_sequence(tmp_path, t_bc)
    _write_traj(tmp_path / "traj.csv", t, p_cam)
    assert score(str(tmp_path / "traj.csv"), str(seq), frame="cam0").ate_se3.rmse < 1e-8
    r_body = score(str(tmp_path / "traj.csv"), str(seq), frame="body")
    assert 0.0 < r_body.ate_se3.rmse <= 0.3 + 1e-9                   # alignment can absorb part of a 0.3 m lever arm


def test_score_refuses_a_short_overlap(tmp_path):
    seq, t, p_cam, _ = _write_sequence(tmp_path, [0.0, 0.0, 0.0])
    _write_traj(tmp_path / "traj.csv", t[:10], p_cam[:10])
    r = score(str(tmp_path / "traj.csv"), str(seq))
    assert any("associations" in b for b in r.verdict_blockers())


def test_load_gt_reads_w_first_quaternions(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("# header\n1000,1,2,3,1,0,0,0\n\n2000,4,5,6,0,1,0,0\n")
    t_ns, pos, q = load_gt(str(p))
    assert t_ns.tolist() == [1000, 2000] and pos[1].tolist() == [4, 5, 6] and q[1].tolist() == [0, 1, 0, 0]

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE MAP BUILD KIT - seven artifacts, written off the frame path.

    kit = MapKit(out_dir, dataset, rig, manifest)
    kit.frame(i, t, T_wc, state)             # every frame
    kit.keyframe(kf_id, i, t, T_wc, cov)     # every keyframe
    kit.retire(landmarks)                    # on the way OUT of the working set
    kit.close(final_kf_poses)

WHY THIS EXISTS AT ALL (plan 1b)
    the engine RUNS no map - no fusion, no render, no densify, no TSDF, no 3DGS -
    because a map coupled into the estimator's frame path costs the estimator
    its budget and gets no verification in return.  A predecessor's TSDF ate
    732 MiB steady with 2.3 GiB rebuild spikes, its Gaussian map hit an
    SH-degree-3 cliff with torch reserving 30 GB while 1.6 GB was live, and the
    audit's verdict on the whole thing was **write-only**.

    But deferring the map is only legitimate if deferring costs NOTHING later.
    So the artifacts are a specified deliverable, and P2's exit is that a real
    3DGS build consumes them.

★ THE TENSION, AND HOW IT RESOLVES
    Culling keeps the backend bounded; a 3DGS build wants the WHOLE sparse
    cloud.  **Culled landmarks are RETIRED, not deleted** - they leave the
    estimator's working set and are appended here on the way out.  The working
    set is bounded by the window; the RECORD is bounded by the trajectory; they
    are two different objects.

    And retirement is LOSSLESS under later loop corrections because landmarks
    are stored **host-relative** - inverse depth in the host keyframe's frame -
    so the post-process re-derives world positions from the FINAL keyframe
    trajectory.  A world-position record would go stale the instant loop
    closure moved its host.

★ THE FRAME CONVENTION IS DECLARED, PROMINENTLY
    Getting it wrong is worth a factor of **2.3-2.6**: scoring a camera-frame
    trajectory as though it were body-frame moved MH_02 from 0.0174 to 0.0404
    and V1_01 from 0.0332 to 0.0848.  The lever arm is 68.9 mm - four times
    MH_02's entire ATE.  A map built against the wrong frame is wrong by that
    much everywhere, and it will look plausible.

Budget: <= 0.3 ms/frame, sequential writes only, never in the frame path's
critical section.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np

LM_STRUCT = struct.Struct("<iifff fff i i")   # see `retire` for the fields
FRAME_CONVENTION = "T_world_cam0_rectified"


def _q_from_R(R):
    """[qx,qy,qz,qw] - TUM order, which is what the scorer reads."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, \
            (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    return x / n, y / n, z / n, w / n


def _evidence_cols(ev):
    """LOG (tier-1 programme, 2026-09-29): the per-frame evidence that the
    keyframe policy and the feature-budget controller acted on, as three
    integer columns `inliers,budget_at_cap,budget` appended after
    `frame_index,state`.  -1 = not applicable (the bootstrap frame has no
    PnP; a front end without a budget controller has no cap).

    `budget_at_cap` is 1 when the SuperPoint budget stood at its cap when
    this frame's keyframe decision was taken - the gate T1-b needs, and the
    per-frame record that proves where that gate is a structural no-op."""
    if ev is None:
        return "-1,-1,-1"
    inl, at_cap, bud = ev
    return "%d,%d,%d" % (-1 if inl is None else int(inl),
                         -1 if at_cap is None else int(at_cap),
                         -1 if bud is None else int(bud))


class MapKit:
    def __init__(self, out_dir, dataset, rig, manifest=None, enabled=True):
        self.dir = out_dir
        self.enabled = enabled
        self.rig = rig
        self.n_retired = 0
        if not enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        self._kf = open(os.path.join(out_dir, "trajectory_kf.csv"), "w",
                        encoding="utf-8", buffering=1 << 16)
        self._kf.write("# frame convention: %s\n" % FRAME_CONVENTION)
        self._kf.write("# t,x,y,z,qx,qy,qz,qw,kf_id,frame_index\n")
        self._fr = open(os.path.join(out_dir, "trajectory_frames.csv"), "w",
                        encoding="utf-8", buffering=1 << 16)
        self._fr.write("# frame convention: %s\n" % FRAME_CONVENTION)
        self._fr.write("# t,x,y,z,qx,qy,qz,qw,frame_index,state,"
                       "inliers,budget_at_cap,budget\n")
        # (t, frame_index, state, anchor_kf_id, T_kf_frame) per frame, so the
        # file can be re-derived from the FINAL trajectory - see `frame`
        self._fr_anchor = []
        self._lm = open(os.path.join(out_dir, "landmarks.bin"), "wb",
                        buffering=1 << 16)
        self._kfj = open(os.path.join(out_dir, "keyframes.jsonl"), "w",
                         encoding="utf-8", buffering=1 << 16)
        self._manifest = dict(manifest or {})
        self._manifest.update(
            frame_convention=FRAME_CONVENTION,
            frame_convention_note=(
                "poses are the RECTIFIED LEFT CAMERA optical centre in world. "
                "Scoring or splatting this as a body-frame trajectory costs a "
                "factor of 2.3-2.6 on EuRoC (68.9 mm lever arm)."),
            dataset=str(dataset), rig=rig.as_dict())
        cal = os.path.join(str(dataset), "calib.json")
        if os.path.isfile(cal):
            blob = open(cal, "rb").read()
            open(os.path.join(out_dir, "calib.json"), "wb").write(blob)
            self._manifest["calib_sha1"] = hashlib.sha1(blob).hexdigest()

    # ---- per frame / per keyframe ---------------------------------------
    def frame(self, i, t, T_wc, state="tracking", anchor=None,
              evidence=None):
        """`anchor` is `(kf_id, T_world_kf_AS_IT_STOOD_NOW)`.

        WITHOUT IT THIS FILE IS ODOMETRY WEARING THE KIT'S NAME.
        The pose streamed here is what the tracker believed before the loop
        correction, the pose graph and the global bundle ever ran - exactly
        the defect `rewrite_keyframes` exists to repair for keyframes, left
        standing for the 2-3x more numerous frames the build order declares
        as the extra 3DGS supervision.  Measured at a keyframe's OWN
        timestamp, where the two files describe the same camera at the same
        instant and so must agree: median disagreement 0.0912 m on MH_01,
        0.0781 on V1_01, 0.0896 on V2_01 - bigger than the whole ATE budget.

        So each frame is kept HOST-RELATIVE to the keyframe that was current
        when it was tracked, the same way a landmark is kept relative to its
        host, and re-derived at close from the FINAL trajectory.  `T_rel` is
        correction-invariant, so it is computed once, here, and never touched
        again.
        """
        if not self.enabled:
            return
        x, y, z = T_wc[:3, 3]
        qx, qy, qz, qw = _q_from_R(T_wc[:3, :3])
        # LOG (tier-1 programme, 2026-09-29): three provenance columns AFTER
        # frame_index,state - see _evidence_cols.  Every consumer of the
        # scorer-facing files goes through run._strip (8 columns), and every
        # kit reader indexes columns 0-9 by position, so the columns are
        # additive.
        ev = _evidence_cols(evidence)
        self._fr.write("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%s,%s\n"
                       % (t, x, y, z, qx, qy, qz, qw, i, state, ev))
        if anchor is not None:
            kid, T_kf = anchor
            T_kf = np.asarray(T_kf, dtype=np.float64)
            R = T_kf[:3, :3]
            T_rel = np.eye(4)
            T_rel[:3, :3] = R.T @ T_wc[:3, :3]
            T_rel[:3, 3] = R.T @ (T_wc[:3, 3] - T_kf[:3, 3])
            self._fr_anchor.append((float(t), int(i), str(state), int(kid),
                                    T_rel, ev))

    def keyframe(self, kf_id, i, t, T_wc, left="", right="", rect_digest=""):
        if not self.enabled:
            return
        x, y, z = T_wc[:3, 3]
        qx, qy, qz, qw = _q_from_R(T_wc[:3, :3])
        self._kf.write("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%d\n"
                       % (t, x, y, z, qx, qy, qz, qw, kf_id, i))
        self._kfj.write(json.dumps({"kf_id": int(kf_id), "frame_index": int(i),
                                    "t": float(t), "left": left, "right": right,
                                    "rect_digest": rect_digest}) + "\n")

    def retire(self, host_kf, bearing, inv_depth, inv_var, n_obs, reason):
        """Append a landmark on its way OUT of the estimator's working set.

        Host-relative: `host_kf_id`, unit bearing in the host camera frame,
        inverse depth and its variance.  World position is NOT stored, so a
        later loop correction of the host keyframe cannot stale it.
        """
        if not self.enabled:
            return
        b = np.asarray(bearing, dtype=np.float64).reshape(3)
        self._lm.write(LM_STRUCT.pack(
            int(host_kf), 0, float(b[0]), float(b[1]), float(b[2]),
            float(inv_depth), float(inv_var), 0.0,
            int(n_obs), int(reason)))
        self.n_retired += 1

    def retire_many(self, rows):
        for r in rows:
            self.retire(*r)

    # ---- close -----------------------------------------------------------
    def rewrite_keyframes(self, records):
        """★ The keyframe trajectory is the OPTIMISED one.

        Streaming a keyframe pose at creation time records what the tracker
        believed before the backend ran - which is the odometry estimate with
        extra steps.  Measured: on MH_01 the streamed file scored 0.2180 while
        the run was carrying corrections it never wrote down.  The optimised
        poses are written once, at the end, from the estimator state.
        """
        if not self.enabled:
            return
        self._kf.close()
        import os as _os
        import shutil as _sh
        path = _os.path.join(self.dir, "trajectory_kf.csv")
        # ★ THE CAUSAL COLUMN (paper-1): the streamed file IS what a live
        # consumer received - poses as produced, before PGO/GBA/rewrite.
        # Preserved beside the optimised file so every run yields both
        # columns from the same draw.  Note ORB-SLAM3's published numbers
        # are also post-corrected; this column is stricter than anything
        # they publish.
        if _os.path.isfile(path):
            _sh.copy2(path, path.replace(".csv", "_causal.csv"))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("# frame convention: %s\n" % FRAME_CONVENTION)
            f.write("# t,x,y,z,qx,qy,qz,qw,kf_id,frame_index (OPTIMISED)\n")
            for kid, idx, t, T in records:
                x, y, z = T[:3, 3]
                qx, qy, qz, qw = _q_from_R(T[:3, :3])
                f.write("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%d\n"
                        % (t, x, y, z, qx, qy, qz, qw, kid, idx))
        _os.replace(tmp, path)
        self._kf = open(path, "a", encoding="utf-8")

    def rewrite_frames(self, records):
        """The per-frame trajectory is the OPTIMISED one too.

        `records` is the tracker's `all_kf` - `[kf_id, frame_index, t, T_wc]`
        with the FINAL poses.  Each frame is re-derived as
        `T_world_kf_final @ T_kf_frame`, which is exact: `T_kf_frame` was a
        rigid relationship at tracking time and no correction changes it.
        """
        if not self.enabled or not self._fr_anchor:
            return 0
        import os as _os
        import shutil as _sh
        final = {int(r[0]): np.asarray(r[3], dtype=np.float64)
                 for r in records}
        self._fr.close()
        path = _os.path.join(self.dir, "trajectory_frames.csv")
        # the causal column, per-frame (see rewrite_keyframes)
        if _os.path.isfile(path):
            _sh.copy2(path, path.replace(".csv", "_causal.csv"))
        tmp = path + ".tmp"
        n_fixed = 0
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("# frame convention: %s\n" % FRAME_CONVENTION)
            f.write("# t,x,y,z,qx,qy,qz,qw,frame_index,state,inliers,"
                    "budget_at_cap,budget (OPTIMISED: re-derived from the "
                    "final keyframe trajectory)\n")
            for t, i, state, kid, T_rel, ev in self._fr_anchor:
                T_kf = final.get(kid)
                if T_kf is None:
                    continue
                T = T_kf @ T_rel
                x, y, z = T[:3, 3]
                qx, qy, qz, qw = _q_from_R(T[:3, :3])
                f.write("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%s,%s\n"
                        % (t, x, y, z, qx, qy, qz, qw, i, state, ev))
                n_fixed += 1
        _os.replace(tmp, path)
        self._fr = open(path, "a", encoding="utf-8")
        return n_fixed

    def close(self, extra=None, depth_index=None):
        if not self.enabled:
            return
        for f in (self._kf, self._fr, self._lm, self._kfj):
            f.close()
        if depth_index is not None:
            json.dump(depth_index, open(os.path.join(self.dir,
                                                     "depth_index.json"), "w"),
                      indent=1)
        self._manifest.update(extra or {})
        self._manifest["landmarks_retired"] = self.n_retired
        self._manifest["landmark_record"] = {
            "struct": "<i i fff ff f i i",
            "fields": ["host_kf_id", "_pad", "bearing_x", "bearing_y",
                       "bearing_z", "inv_depth", "inv_depth_var", "_reserved",
                       "n_obs", "retirement_reason"],
            "note": "HOST-RELATIVE. world = T_world_host * (bearing / inv_depth)"
                    " using the FINAL keyframe trajectory."}
        json.dump(self._manifest,
                  open(os.path.join(self.dir, "run_manifest.json"), "w"),
                  indent=1, default=str)


def load_landmarks(path):
    """Read `landmarks.bin` back - the post-process side of the contract."""
    raw = open(path, "rb").read()
    n = len(raw) // LM_STRUCT.size
    out = []
    for i in range(n):
        out.append(LM_STRUCT.unpack_from(raw, i * LM_STRUCT.size))
    return out

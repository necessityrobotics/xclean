#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE DERIVATIONS - every Class-B quantity as a FUNCTION of calibration, and
every Class-C quantity as a function of what the run reports.

    rig = Rig.from_calib("…/MH_01_easy/calib.json")
    print(rig.table())          # every derived value with its formula

WHY THIS FILE IS THE WHOLE A/B/C/D DOCTRINE IN CODE (plan section 6)

    | class | meaning | rule |
    |-------|---------|------|
    | A | a measurement of the hardware | must be an INPUT (calib/EEPROM/self-cal) |
    | B | derivable from Class A | a FUNCTION of calibration, never a constant |
    | C | derivable from run-time statistics | a FUNCTION of what the run reports |
    | D | genuinely universal | one value, everywhere, forever |

    "One set of defaults" is the CONSEQUENCE; the rule is **zero hand-set
    constants** - a config DERIVATION, not a config FILE.

★ THE RECEIPT THAT MAKES THIS NON-NEGOTIABLE
    `max_depth_m = 9.0` was EuRoC's own rig value (0.2*f*B = 9.6) frozen as if
    it were a property of the world.  It caused **total** failure on all four
    trifecta sequences, and deriving it turned Malaga from 26.9 m to 1.32 m.
    **A Class-B bug does not look like a tuning problem; it looks like the
    dataset being broken.**  That is why they are dangerous and why they live
    in one file with their formulas attached.

★ AND THE OTHER HALF OF THE DOCTRINE, WHICH IS EASY TO GET BACKWARDS
    A pixel threshold on FEATURE LOCALISATION is Class D and must NOT be
    converted to angular units: SuperPoint localises to ~1 px whether the focal
    length is 190 or 795, so it is a property of the DETECTOR, not of the rig.
    A pixel threshold on a REPROJECTION GATE is Class C and must be: there the
    pixel is standing in for a geometric uncertainty that really does scale.
    Getting this backwards is not a tuning error, it is a category error, and
    both directions have already cost this project a week.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

# ---- Class D: genuinely universal.  One value, everywhere, forever. --------
HUBER_K = 1.345                 # 95 % asymptotic efficiency at the Gaussian
MIN_PNP_INLIERS = 25            # 6-DoF conditioning, NOT a fraction of the
MIN_LOOP_INLIERS = 30           # feature budget - it must not scale with it
MIN_RELOC_INLIERS = 60          # 2x the loop floor: a weld is further vetted
                                # by a soft gate + robust kernel + the bundle;
                                # a reloc replaces the live pose on its
                                # inlier set ALONE
NUM_DISPARITIES = 128           # the search extent the depth stage covers
COLD_MATCH_RATE = 0.5           # Class-C COLD START only: replaced by the
COLD_INLIER_RATE = 0.5          # measured rates from the second frame onward
NEAR_CLIP_MARGIN = 1.10         # how far above the saturation disparity the
                                # nearest REPRESENTABLE depth is placed
DEPTH_ERR_FRACTION = 0.10       # the 10 % depth error that defines max_depth
DISPARITY_NOISE_PX = 0.5        # the disparity noise max_depth is defined at
FAR_MIN_DISP_PX = 1.0           # below this a point is a BEARING, not a depth
FEATURE_SIGMA_PX = 1.0          # Class D: SuperPoint localises to ~1 px at any
                                # focal length - a DETECTOR property, and the
                                # information the host keypoint contributes


@dataclass
class Rig:
    """Class A in, Class B out.  Nothing here may be overridden per dataset."""
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    baseline_m: float
    fx_rect: float              # the focal of the RECTIFIED pair - see fB
    fB_m_px: float              # -P2[0,3]: f*B as the disparity actually means
    fps: float
    imu_rate_hz: float
    has_imu_noise: bool
    td_s: float = 0.0           # camera-IMU time offset: a DECLARED Class-A
                                # input.  Measured blindness: a 1 ms offset is
                                # invisible to all 15 core rows, and 20 ms
                                # moves mh02 by 44.3 % while moving kitti07 by
                                # 0.8 %.  It is not modelled anywhere today.

    # ---- Class B: functions of the above, never constants ---------------
    @property
    def fB(self) -> float:
        """f*B in metre-pixels, taken from the RECTIFIED projection matrix.

        ★ NOT `fx * baseline`.  Disparity is measured on the RECTIFIED images,
        so the only focal length that converts it to depth is the rectified
        one.  On EuRoC MH_01 the raw form gives 458.654 * 0.11008 = 50.49 and
        the rectified form gives -P2[0,3] = 48.02 - a **5.1 % error**, which
        would inflate `max_depth_m` from 9.60 to 10.10 m and the far-bearing
        ceiling from 48.0 to 50.5 m.

        This was caught by the derived value disagreeing with the plan's own
        table (9.6 m / 48 m), and it is exactly the shape of the Class-B bug
        the plan warns about: it does not look like a tuning problem, it looks
        like the dataset being slightly wrong everywhere.
        """
        return self.fB_m_px

    @property
    def max_depth_m(self) -> float:
        """Where a `DISPARITY_NOISE_PX` error becomes a 10 % depth error.

        z = f*B/d and dz/z = dd/d, so a 10 % depth error at 0.5 px of
        disparity noise is d = 5 px, i.e. z = f*B/5 = 0.2*f*B.
        EuRoC: 9.6 m.  The frozen 9.0 was this number, for this rig, printed.
        """
        return (DEPTH_ERR_FRACTION / DISPARITY_NOISE_PX) * self.fB

    @property
    def near_clip_m(self) -> float:
        """Below f*B/num_disparities no depth is REPRESENTABLE - the search
        saturates - so a nearer surface is reported AT the near plane.  That is
        a biased value, not a missing one, and the margin keeps the estimator
        off the bias."""
        return NEAR_CLIP_MARGIN * self.fB / NUM_DISPARITIES

    @property
    def far_bearing_m(self) -> float:
        """Beyond f*B/min_disp a landmark carries direction but no usable
        range, so it enters the graph as a BEARING.  EuRoC 48 m, Malaga 95 m -
        and the fact that those differ by 2x is the whole argument against a
        constant."""
        return self.fB / FAR_MIN_DISP_PX

    @property
    def rho_sigma(self) -> float:
        """sigma of INVERSE DEPTH, in 1/m.  Class B, and CONSTANT.

        `rho = d / (f*B)` is LINEAR in disparity, so a fixed disparity noise is
        a fixed inverse-depth noise at every range.  That single fact is the
        whole argument for inverse depth as the landmark parameterisation: the
        measurement's information is range-independent, so the state that
        carries it has a well-conditioned Hessian block at 2 m and at 200 m,
        where a Euclidean XYZ block degenerates along the ray as 1/z^2.

        It is also what makes the stereo prior LINEAR in the state - residual
        `(rho - rho_meas)/sigma_rho`, Jacobian `1/sigma_rho`, no units
        conversion.  The range-versus-depth defect that cost MH_01 a path ratio
        of 0.8906 cannot be written in this form.
        """
        return DISPARITY_NOISE_PX / self.fB

    @property
    def bearing_sigma(self) -> float:
        """sigma of the normalised-plane bearing, in radians-ish (x_n units).

        `x_n = (u - cx)/fx`, so a keypoint localisation of `sigma_px` is a
        bearing uncertainty of `sigma_px / fx` - Class B from a Class-D
        detector property.  This is the information the HOST keyframe's own
        observation carries, and it is what makes the bearing a variable with
        a prior rather than a constant.
        """
        return FEATURE_SIGMA_PX / self.fx_rect

    @property
    def rho_floor(self) -> float:
        """The smallest inverse depth the estimator may hold - a POSITIVITY
        guard, not a band.

        The band is `far_bearing_m` (where disparity falls under a pixel).
        A solve is allowed to push a landmark PAST the band - that is the point
        of inverse depth - but not through infinity and out the other side,
        which is what a negative rho means.  Twice the far-bearing range is the
        derived place to stop: it is outside anything the sensor can measure
        and still finite, so the stereo prior can pull the landmark back.
        """
        return 1.0 / (2.0 * self.far_bearing_m)

    def depth_sigma_m(self, z, disp_noise_px=DISPARITY_NOISE_PX) -> float:
        """sigma_z = z^2 * sigma_d / (f*B) - the physics bound of plan 8.4.

        This is what lets the project say something no competitor ATE can: not
        "someone achieved 0.013" but "0.009 is the floor for THIS rig on THIS
        trajectory and you are 2x off it."
        """
        return (z * z) * disp_noise_px / self.fB

    def conditioning_floor(self, match_rate=None, inlier_rate=None) -> int:
        """The fewest keypoints that can still produce a conditioned solve.

        Derived, not chosen: a 6-DoF solve needs `MIN_PNP_INLIERS` INLIERS, and
        a keypoint becomes an inlier only after it is matched and survives
        rejection.  So the floor is `min_inliers / (match_rate * inlier_rate)`,
        where both rates are things the run MEASURES.  The cold-start values
        are declared above and are replaced from the second frame onward -
        that is what makes this Class C rather than a constant with a story.
        """
        m = COLD_MATCH_RATE if match_rate is None else max(match_rate, 1e-3)
        i = COLD_INLIER_RATE if inlier_rate is None else max(inlier_rate, 1e-3)
        return int(math.ceil(MIN_PNP_INLIERS / (m * i)))

    def feature_budget(self, density_per_px=2.84e-4, match_rate=None,
                       inlier_rate=None) -> int:
        """Class C, at its starting value only.

        The budget is a CONTROL LOOP, not a constant (plan 4.1): it starts here
        and grows only when the MEASURED inlier count approaches the floor,
        shrinking when it is far above.  The seed is `density x area` - an
        order below an earlier engine's 1024 on 752x480, which is 2.84e-3/px - and it may
        never go under `conditioning_floor`.
        """
        return max(int(density_per_px * self.width * self.height),
                   self.conditioning_floor(match_rate, inlier_rate))

    def keyframe_parallax_px(self, median_flow_px, median_z_m) -> float:
        """Class C: the gate's parallax is MEASURED, never assumed
        (`median(flow_px) * median(z) / fx`)."""
        return median_flow_px * median_z_m / self.fx

    # ---- io -------------------------------------------------------------
    @classmethod
    def from_calib(cls, path, name=None, raw_principal_point=False):
        """Class A in.  Rectification is performed HERE, once, because the
        rectified focal is the only one the derivations may use."""
        d = json.load(open(path, encoding="utf-8"))
        return cls.from_calib_dict(d, name=name or d.get("dataset") or os.path.basename(
            os.path.dirname(os.path.abspath(path))), raw_principal_point=raw_principal_point)

    @classmethod
    def from_calib_dict(cls, d, name=None, raw_principal_point=False):
        """The same derivation from an already-loaded calibration (xclean live mode: the engine receives the
        calibration, never a dataset path)."""
        import numpy as np
        path = "<calib dict>"
        M1 = np.array(d["M1"], dtype=float)
        w, h = int(d["camera_width"]), int(d["camera_height"])
        B = abs(float(d["baseline_m"]))
        fx_rect, fB = float(M1[0][0]), float(M1[0][0]) * B
        # ★ DEFECT 18 - THE CAMERA MODEL WAS MIXED, AND HALF-FIXED IS WORSE
        # THAN CONSISTENTLY WRONG.
        # The images the engine sees are rectified with `initUndistortRectifyMap
        # (M1, D1, R1, P1, ...)`, so the ONLY camera model that describes them
        # is P1's - focal AND principal point together.  A previous session
        # correctly moved the FOCAL and the BASELINE to the rectified pair
        # ("f*B comes from the RECTIFIED pair, not fx*baseline, 5.1 % high")
        # and left the PRINCIPAL POINT on the raw M1, so the shipped rig paired
        # a rectified focal with a raw centre.
        #
        # MEASURED on EuRoC: raw (367.215, 248.375) against rectified
        # (364.441, 256.952) - dcx -2.774 px, dcy **+8.577 px**, |d| 9.014 px.
        # At the rectified focal of 436.244 that is **1.184 deg** of systematic
        # bearing error on every keypoint, and it unprojects into a shear of
        # the reconstructed scene proportional to depth, because
        # `x = (u - cx) z / f`.
        #
        # Two independent things pointed at the same number before it was
        # found: `tools/build_map.py` has always used P1's principal point, so
        # the package contradicted itself about the same images; and a
        # residual analysis of V1_01 attributed 46.9 % of its per-keyframe
        # rotation error to ONE constant camera-frame rotation of **1.15 deg**.
        #
        # The kit's own reprojection check could not see it: it tests that a
        # landmark lands INSIDE its host image, which a 9 px error still does.
        cx_rect, cy_rect = float(M1[0][2]), float(M1[1][2])
        try:
            import cv2
            R1, R2, P1, P2, _Q, _r1, _r2 = cv2.stereoRectify(
                M1, np.array(d["D1"], dtype=float),
                np.array(d["M2"], dtype=float),
                np.array(d["D2"], dtype=float), (w, h),
                np.array(d["R_stereo"], dtype=float),
                np.array(d["T_stereo"], dtype=float).reshape(3, 1), alpha=0)
            fx_rect = float(P1[0, 0])
            fB = abs(float(P2[0, 3]))
            if not raw_principal_point:
                cx_rect, cy_rect = float(P1[0, 2]), float(P1[1, 2])
        except Exception as e:      # a rig with no distortion model, or no cv2
            print(" ! rectification unavailable (%s: %s) - f*B falls back to "
                  "fx*B, which is 5 %% high on EuRoC, and the principal point "
                  "falls back to the RAW one.  Every derivation below "
                  "inherits both." % (type(e).__name__, e))
        return cls(name=name or d.get("dataset") or "live",
                   fx=float(M1[0][0]), fy=float(M1[1][1]),
                   cx=cx_rect, cy=cy_rect,
                   width=w, height=h, baseline_m=B,
                   fx_rect=fx_rect, fB_m_px=fB,
                   fps=float(d.get("camera_fps") or 0.0),
                   imu_rate_hz=float(d.get("imu_rate_hz") or 0.0),
                   has_imu_noise=bool(d.get("imu_noise")),
                   td_s=float(d.get("td_s") or 0.0))

    def as_dict(self):
        out = asdict(self)
        out.update(fB=self.fB, max_depth_m=self.max_depth_m,
                   near_clip_m=self.near_clip_m,
                   far_bearing_m=self.far_bearing_m,
                   feature_budget=self.feature_budget())
        return out

    def table(self) -> str:
        rows = [
            ("f*B", "B", "-P2[0,3], the RECTIFIED pair (not fx*B)",
             "%.3f m.px" % self.fB),
            ("max_depth_m", "B", "0.2 * f*B  (10 % error at 0.5 px)",
             "%.2f m" % self.max_depth_m),
            ("near_clip_m", "B", "margin * f*B / num_disparities",
             "%.3f m" % self.near_clip_m),
            ("far_bearing_m", "B", "f*B / min_disp_px",
             "%.1f m" % self.far_bearing_m),
            ("feature budget (seed)", "C",
             "max(density*area, min_inl/(match*inlier))",
             "%d kp" % self.feature_budget()),
            ("conditioning floor", "C", "min_inliers / (match_rate*inlier_rate)",
             "%d kp" % self.conditioning_floor()),
            ("sigma_z at max_depth", "B", "z^2 * sigma_d / f*B",
             "%.3f m" % self.depth_sigma_m(self.max_depth_m)),
            ("min_pnp_inliers", "D", "6-DoF conditioning",
             "%d" % MIN_PNP_INLIERS),
            ("huber k", "D", "95 % efficiency at the Gaussian",
             "%.3f" % HUBER_K),
            ("camera-IMU td", "A", "DECLARED input; unmodelled today",
             "%.4f s" % self.td_s),
        ]
        w = "  %-22s %-3s %-42s %s"
        out = ["  rig: %s   %dx%d  fx=%.2f  B=%.4f m  %.0f fps"
               % (self.name, self.width, self.height, self.fx,
                  self.baseline_m, self.fps),
               w % ("quantity", "cls", "derivation", "value"),
               "  " + "-" * 92]
        out += [w % r for r in rows]
        if not self.has_imu_noise:
            out.append("  ! no imu_noise block: the engine does not use the "
                       "IMU on this rig (vision only)")
        return "\n".join(out)


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="print a rig's derivations")
    ap.add_argument("calib", nargs="+", help="calib.json path(s), e.g. C:\\data\\xclean\\MH_01_easy\\calib.json")
    a = ap.parse_args()
    for p in a.calib:
        print(Rig.from_calib(p).table())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

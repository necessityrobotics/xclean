# xclean/vio conventions (the live engine's sliding window). Binding for window.py, marg.py, engine_live.py, front/*,
# imu_factor.py

Written for the live engine, 2026-10-23. imu_factor.py follows the "pose block" and "frame state" rules below.

## Frames
- **w**: the odometry (odom) frame of the current segment. It is gravity-free in VO mode. After VI init it is NOT
  rotated; gravity is a separate window variable g_w.
- **b**: the IMU body frame. On a rig without an admitted IMU, b is identical to c (T_bc = I).
- **c**: the RECTIFIED left camera (cam0 rectified). Every pixel measurement is in the rectified pair.
  - The published pose is T_w_c (`FRAME_CONVENTION = "T_world_cam0_rectified"`, as kit/writer.py and run.py).
- **Extrinsic** T_bc (camera -> body) = calib `T_body_cam0` x [[R1^T, 0], [0, 1]].
  - R1 is the left rectification rotation from `cv2.stereoRectify` with the calib's M1/D1/M2/D2/R_stereo/T_stereo,
    alpha = 0, exactly as `io/dataset.py rectify_maps` and `live/engine.py LiveSource`.
  - So x_rect = R1 x_cam0, and a vector in the rectified frame maps to the raw cam0 frame by R1^T.

## Pose variables in the window
- Every window pose (keyframe or frame) is stored as the BODY pose **T_wb = (R_wb, p_wb)**.
  - The camera pose is derived: T_wc = T_wb T_bc.
  - In VO mode T_bc = I, so T_wb = T_wc.
- **Perturbation (the native form of the IMU factor):**
  - rotation: R_wb <- R_wb Exp(dphi), i.e. RIGHT (body-frame) perturbation;
  - position: p_wb <- p_wb + dp, ADDITIVE in w.
- **Error-state order of a pose block: [dphi (3), dp (3)]**, 6 dims. The window's reduced system is the concatenation of
  pose blocks in slot order: keyframes oldest -> newest, then frames oldest -> newest.
- **VI frame states**: [dphi, dp, dv, dbg, dba], 15 dims.
  - v_w is additive in w; the biases are additive in b.
  - The first 6 dims of a frame-state block are the pose block above.
- **Box-minus** (used by the marginal prior and by FEJ): x [-] x0 = [Log(R0^T R), p - p0] (, v - v0, bg - bg0, ba - ba0).
  - The prior's Jacobian wrt dphi is taken as I (small-difference approximation, Basalt practice). This is stated, not
    hidden.

## Landmarks
- Each landmark is hosted in a keyframe h. Its 3 free parameters are (a, b, rho):
  - m = (a, b, 1) is the normalised RECTIFIED-camera ray of the host observation;
  - rho is the inverse DEPTH along the camera z axis, so X_c,h = m / rho.
- **Observation in camera t** (any window pose, t may equal h):
  - q = R_ct (R_wc,h m + rho (p_wc,h - p_wc,t)), where R_ct = R_wc,t^T;
  - left pixel (u, v) = (fx q0/q2 + cx, fy q1/q2 + cy);
  - right pixel (rectified) u_R = fx (q0 - rho B)/q2 + cx, with B = the rectified baseline, so fx B = f*B of the
    RECTIFIED pair (derive.Rig.fB_m_px).
- **Stereo default:** a host inverse-depth prior r = (rho - rho_meas)/sigma_rho.
  - rho_meas = d / (fx B), with d the 1-D stereo disparity at creation;
  - sigma_rho = kappa_d sigma_LK / (fx B), where sigma_LK is the 1-D LK Gauss-Newton sigma and kappa_d is the
    Class-C running robust scale.
  - The research switch XCLEAN_P1_STEREO=ur instead adds a u_R residual on every frame that has a stereo measurement.

## Residuals and noise
- **Whitened residual** = (prediction - measurement) / sigma. Huber on the whitened norm, k = 1.345 (class D).
- **Normal equations:** minimise 1/2 ||r + J dx||^2, H = J^T J, b = J^T r, H dx = -b.
- **Pixel sigma:** FEATURE_SIGMA_PX at start (class D, carried). Then the Class-C running robust scale of the window's
  whitened reprojection residuals. It is held fixed inside one LM pass.

## Time and IMU (the causal rule)
- No IMU sample newer than the released horizon is ever read. Between the last released sample and the frame stamp, the
  last released rate is held (zero-order).
- Timestamps are the dataset/feeder frame stamps in seconds (float64), the same clock as the IMU stamps.

## Camera-IMU time offset (designed 2026-10-26, then removed the same day)
- **The engine applies NO time offset:** every IMU span of camera interval [t_i, t_j] is [t_i, t_j]. The items below
  record the removed integration and td_est.py's sign, for a possible re-proposal; none is live.
- **Sign (td_est.py):** t_imu = t_cam + td. A physical instant stamped t_c by the camera is
  stamped t_c + td by the IMU. vi_init.offset_align uses the opposite sign (-td).
- **Every IMU span of a camera interval [t_i, t_j] is [t_i + td_apply, t_j + td_apply]** (engine_live `td_apply`, 0
  unless an offset is applied): `_predict` (VO gyro rotation), `_predict_vi` (propagation = the frame's IMU factor),
  `_try_d1` (stationarity sub-windows), `_try_vi_admission` (keyframe stamps passed to vi_init). Slot stamps `s["t"]`,
  `_frame_t`, `_kf_hist` stay camera stamps; the shift is applied at the call.
- **Horizons are release limits and are never shifted.** With td > 0 the newest factor's last td seconds lie past the
  horizon and are held (the causal rule above, `extrapolated`), then re-preintegrated by `_repreintegrate` once released.
- **set_td(x)** re-preintegrates every window IMU factor at [t_i + x, t_j + x] with its own linearisation biases; the
  marginal prior is not re-linearised (a stated approximation).
- **The offset estimator's input attitude is vision-only:** the frame's PnP pose while the IMU is in the solve, the
  window-solved pose of a vision-only window otherwise.

## Pose-free landmarks (2026-10-26)
- A landmark is POSE-FREE in a window solve when its only valid observation is its host keyframe's LEFT observation
  (stereo form "prior"): at t == h, q = m, so its pose Jacobians are zero and it adds nothing to the Schur system.
- engine_live._problem(posefree=True) (the window solve only; never a marginalisation) moves them into
  `Problem.pf` (window.PoseFree: host slot, host left pixel, the common left whitening 1 / sigma_px, the inverse-depth
  prior, the state). window.solve keeps their robust cost in both LM costs and moves their own damped 3x3 step with the
  solve's accept decisions; final_linearisation gives their marginals. The result equals the full solve bit for bit.
- Diagnostics that change meaning: the frame log's `vis_win_n` / `vis_win_chi2_sum` cover the Schur (pose-coupled)
  landmarks only; `window_L` counts the solved landmarks, `window_L_posefree` the excluded ones.

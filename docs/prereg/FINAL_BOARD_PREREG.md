# FINAL BOARD PRE-REGISTRATION (lead; v1 2026-10-27 10:40 IST, v2 2026-10-27 11:40 IST, v3 2026-10-27 13:20 IST, v4 2026-10-27 14:50 IST, v4.1 2026-10-27 ~16:30 IST, v4.2 2026-10-28 13:44 IST, v4.3 2026-10-28 16:34 IST (owner ruling 8.40))

**Declared BEFORE the final xclean system exists and BEFORE any never-run row is run.** v2 corrected v1 after an independent
verification, v3 corrected v2 after a second one, and v4 corrects v3 after a third (section 8 logs every correction with its
date, the defect, its root and the fix: 8.1-8.12 for v2, 8.13-8.22 for v3, 8.23-8.30 for v4, 8.31-8.34 for v4.1). v1 is kept byte for byte in
runs/xclean/board/v1_20261027_1040/ (prereg, block, evaluator, census), v2 in runs/xclean/board/v2_20261027_1140/ (prereg, block,
evaluator, runner, census, dry-test maps) and v3 in runs/xclean/board/v3_20261027_1320/ (prereg sha1 0a0f756e with the owner's
n = 3 ruling line, block, evaluator, runner, census, dry-test maps, dryrun3_logic.py, blocktest/ and the v3 dry-test outputs),
and v4 in runs/xclean/board/v4_20261027_1526/ (prereg sha1 786c5720, block, evaluator, runner, census, dry-test maps,
dryrun3_logic.py, the v4 blocktest/ scripts and outputs and the v4 dry-test outputs),
so every change can be audited. v2, v3 and v4 were written before any never-run row was run, before K14 ran (no k14_ run dir
exists at 14:20) and before the final snapshot exists; v4.1 (the audit's last minor B1 and three K14-interface corrections,
8.31-8.34) too: no k14_ run dir, k14_ledger.tsv or k14_void exists at 15:26, and no never-run row has been run.
- State at writing: P1.3 is in fix round 2 (workflow wf_b7838d76-659, tree D:/xoneslam_runs/xclean/p13f2_tree). K14 has NOT run (its
  block D:/xoneslam_runs/xclean/k14_block.sh waits for the fix-2 snapshot). The final frozen snapshot does not exist: its path and
  `_engine_md5` are PARAMETERS here (`SNAP`, `SNAP_MD5`, `SNAP_LS` of board_block.sh; `--snap-md5`, `--snap-ls-md5` of
  board_eval.py), never constants.
- No rule in this file was derived from any result on a never-run, K14 or sealed row. About those rows the author opened only:
  registry paths, file existence, frame / IMU counts and stamps, durations, register flag CODES, and the existence of baseline files.
  No GT trajectory and no result of xclean on those rows was read. Two inadvertent exposures to OTHER systems' numbers on unseen
  rows happened; both are disclosed here WITHOUT the values, and no rule depends on them (every rule is mechanical and fixed here):
  - **Exposure 1 (v1, ~10:10).** Reading the header of runs/xclean/research/live/table_usab.md (named in the task) showed its first
    10 rows. Among them were the OLD x1 system's usability values (U2) on one never-run row, v2_01_easy. The other rows shown were
    already-run rows or the K14 rows mh_02_easy / v2_02_medium, whose values K14's selection had already used. The file was not
    read further. (v1 quoted the values; v2 removes them, correction 8.5.)
  - **Exposure 2 (v2, ~11:25).** While checking the field names of website/data/board.json, the author printed its first row entry.
    That entry is the SEALED row 4s_20201222_115: it showed the frozen paper-1 v3 board record of that row (the xtenslam v3 and
    ORB-SLAM3 untuned ATE ranges and the board verdict). The board-rule design of C4 (correction 8.4) had been decided and its
    runner identity-tested before this print; nothing after it uses those values. board_eval.py reads only the `row`,
    `board_row` and `set` fields of board.json (as v1 did).
- Owner rulings implemented:
  - 2026-10-26 ~22:00: the rows xclean has never run stay unseen until the final full-board report of the frozen full-SLAM
    system; that report is their clean held-out test.
  - 2026-10-27 ~07:40: 7 rows were spent as K14 held-out and leave the unseen set.
  - Finish line, 2026-10-26 ~16:00: ONE full-board report of the frozen config beside untuned ORB-SLAM3, with every row shown,
    design rows marked and losses shown.
- Binding together with:
  - runs/xclean/livebars/LIVE_BARS.md (v1, validity protocol v2 as amended for the board by 8.36 and 8.40, GTF1, amendments), including B5 NOW "n = 3 for any claim";
  - runs/xclean/skeleton/P1_3_PREREG.md (K13 / K14, rulings R15-R19, Amendments L, E1, S1, F2);
  - the memory rules: board honesty + no harm (headline = the board's own rule, KF level, common window, symmetric lost-draw
    hygiene), staged untuned target, xclean directive ("n = 3 only for claims"; target "under the existing board rule (scorer_v3,
    honest hygiene, KF level) ... report alongside the causal / live column"), eval tiers + GT / no-GT flags, real-time latency,
    time-first evidence, dataset-defect survey.
- Files of this pre-registration (all new; md5 / sha1 in 6.1):
  - runs/xclean/board/census.py, rows.json (FROZEN, sha1 81b41675a688c0d77821e1c370c5a9323870072e, unchanged in v2 and v3);
  - runs/xclean/board/board_eval.py, board_rule_runner.py;
  - runs/xclean/board/dryrun_rows.json, dryrun2_rows.json, dryrun3_logic.py (v3; v4 adds T7-T9), fixtures/dry2/ (v3: the
    dry-test-2 FAIL fixtures, moved out of the session scratchpad, correction 8.22), blocktest/ (v3: the block dry-test
    scripts, the fake harness and their outputs; v4: selftest4.sh, integrity4.sh, the fake snapshot's front/fastpaths.py and
    engine_summary.json, and their outputs; v4.1: selftest41.sh and its output selftest41.out, the v4.1 re-runs
    selftest4_v41.out, selftest2v4_v41.out, integrity4_v41.out, evalcheck4_v41.out);
  - runs/xclean/board/data/census.{json,md}, data/dryrun_board{.json,_tables.md}, data/dryrun2_board{.json,_tables.md},
    data/dryrun3_logic.txt (v3), data/board_rule/ (board-rule outputs and logs; the identity test x1_dev8_n1_identity),
    data/kfboard_dry1_60809bf3/ and data/kfboard_dry2_60809bf3/ (dry-test board-rule exports), data/derived/dry_* (dry-test
    usability files);
  - runs/xclean/board/v1_20261027_1040/ (the v1 files), v2_20261027_1140/ (the v2 files), v3_20261027_1320/ (the v3 files) and
    v4_20261027_1526/ (the v4 files);
  - D:/xoneslam_runs/xclean/board_block.sh.

## 1. Row census (re-derived 2026-10-27 10:14 for v1 and 11:28 for v2 by runs/xclean/board/census.py; data/census.json, data/census.md)

### 1.1 Method
- **Board rows:** the 67 rows of the frozen board website/data/board.json (paper 1 v3: 39 "development" + 28 "heldout" =
  extended). Only names, set and family were read.
- **Registry:** compactslam/bench/rows.py + runs/extra_rows.json (dataset and GT paths).
- **xclean run records:** every top-level entry of D:/xoneslam_runs/xclean was matched by name, and by the harness.json tag
  where one exists:
  - v2 (11:28): 1383 top-level dirs (engines/, logs/ and stale_locks*/ skipped), 1177 of them with a harness.json, 167 named after
    a row without a harness.json, 40 without a row name (v1 printed "1215 top-level dirs", which was harness dirs + unmatched
    dirs: correction 8.10);
  - plus a depth-3 NAME scan (files and dirs) for any artefact naming a row.
- **Status rule** (declared sets; the census then checks them against the run records):
  - design: P1_3_PREREG "Rows" (11) + mh_01_easy and kitti_07, which became design rows after the K13 diagnosis (KILL_LOG
    "P1.3 / K13");
  - K13 held-out clean: kitti_05, kitti_06, mh_03_medium, v1_01_easy (seen; K14 text: "no longer blind");
  - K12-e held-out: rosario_01, rosario_02, rosario_05, rosario_06, v1_03_difficult, mh_05_difficult (run by P1.2 lock-step only,
    never by P1.3);
  - K14 held-out: the 7 rows of P1_3_PREREG "K14";
  - sealed: imu_common.SEALED = diag/usability/draws.SEALED = the runs/xsesto/README.md charter (all three equal; 16 rows);
  - never-run unsealed: every other board row.
- **Integrity result:**
  - 0 conflicts: no unseen, sealed or K14 row has any xclean artefact, at any depth;
  - every seen row has run dirs;
  - the 40 top-level dirs without a row name (builder trees, caches, OAK IMU captures, smoke tests; 38 in v1, plus the fix-2
    trees p13f2_tree and p13f2_ref_tree) are listed in census.json, and none names a board row.

### 1.2 Counts and the reconciliation with "28 unsealed + 16 sealed"

| status | rows |
|---|---:|
| design | 13 |
| K13 held-out (clean, seen) | 4 |
| K12-e held-out (seen by P1.2 lock-step only; on the board) | 5 |
| K14 held-out (spent 2026-10-27, not yet run) | 7 |
| **never-run unsealed** | **22** |
| **sealed (never run)** | **16** |
| total | 67 |

**Reconciliation with the owner's 2026-10-26 figure "~44 = 28 unsealed + 16 sealed":**
- Before K14 the true figure is **29 unsealed + 16 sealed = 45**: 51 unsealed board rows minus the 22 board rows xclean has run.
- The difference of one row is **rosario_05**. K12-e ran it, so "~44" subtracted it, but it is NOT a board row: it was never on the
  board because of its non-monotonic GT (runs/target1/BOARD_DECISION.md).
- After K14 spent 7 rows: **22 unsealed + 16 sealed = 38 unseen rows**. They are the clean held-out test of this report.
- Also note that board.json's "heldout" label (28 rows) means the paper-1 v3 EXTENDED set (16 sealed + 12 unsealed), not an
  xclean held-out set. The equal count with "28 unsealed" is a coincidence.
- Off-board registry rows (not reported):
  - rosario_05: run by K12-e;
  - madmax_C2, malaga_01, malaga_03: bad-GT rule, user "drop" 2026-10-03;
  - 4s_20201008_093: S-GTTAIL, user ruling 3.
  - None of the last four has an xclean run.

### 1.3 Per-row census (readiness = file existence + counts + duration only; flags = register codes only)
Every row is ready: calib.json present, left / right image counts equal to the timestamps.csv frame count, and the GT file present.
- KITTI has no IMU (KITTI odometry).
- Drives: C: = EuRoC (NVMe); D: = KITTI / 4Seasons board / MadMax board / Rosario (NVMe); E: = the 28 extended rows and OAK (USB).
- The P1.3 K13 paced run of madmax_E1 read from E: with a valid feeder, so the E: path is known to work.

| # | row | status | family (board set) | dataset | calib | IMU samples (Hz) | GT file | frames | duration s | period ms | register flags (non-competitor; status if not confirmed) |
|---:|---|---|---|---|---|---|---|---:|---:|---:|---|
| 1 | madmax_E1 | design | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_E1 | yes | 95081 (100) | present | 12803 | 950.7 | 74.3 | M-DROW, M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited) |
| 2 | 4s_20200324_173 | design | 4Seasons board (development) | D:/wm/datasets/4seasons_oak/fourseasons_2020-03-24_17-36-22 | yes | 1002819 (2000) | present | 15177 | 501.4 | 33.0 | S-EXTR (handled), S-FEED, S-GTGAP, S-GTSCALE (handled), S-HOOD, S-SAME, S-VSCALE (candidate) |
| 3 | 4s_20210107_120 | design | 4Seasons board (development) | D:/wm/datasets/4seasons_oak/fourseasons_2021-01-07_12-04-03 | yes | 908267 (2000) | present | 13746 | 454.1 | 33.0 | S-EXTR (handled), S-FEED, S-GTGAP, S-GTSCALE (handled), S-HOOD, S-SAME, S-VSCALE (candidate) |
| 4 | kitti_00 | design | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_00 | yes | none (KITTI odometry) | present | 4541 | 470.6 | 103.7 | K-CAL |
| 5 | kitti_01 | design | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_01 | yes | none (KITTI odometry) | present | 1101 | 114.0 | 103.7 | K-CAL, K-GTS |
| 6 | kitti_07 | design | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_07 | yes | none (KITTI odometry) | present | 1101 | 114.3 | 103.9 | K-CAL |
| 7 | madmax_F3 | design | MadMax board (development) | D:/wm/datasets/madmax_oak/madmax_F3 | yes | 69309 (100) | present | 8190 | 693.0 | 84.6 | M-EXTRTILT, M-FDROP, M-GTSAW, M-GTSD, M-TD, VROW |
| 8 | mh_01_easy | design | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/MH_01_easy | yes | 36820 (200) | present | 3682 | 184.0 | 50.0 | - |
| 9 | mh_04_difficult | design | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/MH_04_difficult | yes | 20320 (200) | present | 2032 | 101.5 | 50.0 | - |
| 10 | rosario_03 | design | Rosario (development) | D:/wm/datasets/rosario_oak/rosario_03 | yes | 28456 (142) | present | 3090 | 200.5 | 64.9 | R-GTORI, R-IMUGAP, R-RS, R-SAT, R-TD |
| 11 | rosario_04 | design | Rosario (development) | D:/wm/datasets/rosario_oak/rosario_04 | yes | 23388 (142) | present | 2540 | 164.8 | 64.9 | R-GTORI, R-RS, R-SAT, R-TD |
| 12 | v1_02_medium | design | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/V1_02_medium | yes | 17100 (200) | present | 1710 | 85.4 | 50.0 | E-V1DOFF (candidate), E-V1ORI |
| 13 | v2_03_difficult | design | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/V2_03_difficult | yes | 23370 (200) | present | 1921 | 116.7 | 60.8 | E-V203DROP, E-V2CAM, E-V2ROW |
| 14 | kitti_05 | K13 held-out (clean) | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_05 | yes | none (KITTI odometry) | present | 2761 | 287.5 | 104.2 | K-CAL, K-DOFF |
| 15 | kitti_06 | K13 held-out (clean) | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_06 | yes | none (KITTI odometry) | present | 1101 | 114.3 | 103.9 | K-CAL, K-DOFF, K-GTS |
| 16 | mh_03_medium | K13 held-out (clean) | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/MH_03_medium | yes | 27008 (200) | present | 2700 | 134.9 | 50.0 | - |
| 17 | v1_01_easy | K13 held-out (clean) | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/V1_01_easy | yes | 29120 (200) | present | 2912 | 145.6 | 50.0 | E-V1DOFF (candidate), E-V1ORI |
| 18 | mh_05_difficult | K12-e (P1.2 only) | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/MH_05_difficult | yes | 22721 (200) | present | 2273 | 113.6 | 50.0 | - |
| 19 | rosario_01 | K12-e (P1.2 only) | Rosario (development) | D:/wm/datasets/rosario_oak/rosario_01 | yes | 63138 (142) | present | 6856 | 444.9 | 64.9 | R-GTORI, R-IMUGAP, R-RS, R-SAT, R-TAIL (handled) |
| 20 | rosario_02 | K12-e (P1.2 only) | Rosario (development) | D:/wm/datasets/rosario_oak/rosario_02 | yes | 38260 (142) | present | 4155 | 269.6 | 64.9 | R-GTORI, R-IMUGAP, R-RS, R-SAT, R-TD |
| 21 | rosario_06 | K12-e (P1.2 only) | Rosario (development) | D:/wm/datasets/rosario_oak/rosario_06 | yes | 63125 (142) | present | 6854 | 444.7 | 64.9 | R-GTORI, R-IMUGAP, R-RS, R-SAT, R-TAIL (handled), R-TD |
| 22 | v1_03_difficult | K12-e (P1.2 only) | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/V1_03_difficult | yes | 21500 (200) | present | 2149 | 107.4 | 50.0 | E-V1DOFF (candidate), E-V1ORI |
| 23 | madmax_E0 | K14 held-out | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_E0 | yes | 89289 (100) | present | 11876 | 892.6 | 75.2 | M-DROW, M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited) |
| 24 | 4s_20200612_101 | K14 held-out | 4Seasons board (development) | D:/wm/datasets/4seasons_oak/fourseasons_2020-06-12_10-10-57 | yes | 990056 (2000) | present | 14983 | 495.0 | 33.0 | S-EXTR (handled), S-FEED, S-GTGAP, S-GTSCALE (handled), S-HOOD, S-RTK, S-SAME, S-VSCALE (candidate) |
| 25 | kitti_02 | K14 held-out | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_02 | yes | none (KITTI odometry) | present | 4661 | 483.3 | 103.7 | K-CAL |
| 26 | kitti_09 | K14 held-out | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_09 | yes | none (KITTI odometry) | present | 1591 | 165.0 | 103.8 | K-CAL, K-DOFF, K-DOFF09 |
| 27 | mh_02_easy | K14 held-out | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/MH_02_easy | yes | 30400 (200) | present | 3040 | 152.0 | 50.0 | - |
| 28 | rosariov2_01 | K14 held-out | Rosario (development) | D:/wm/datasets/rosario_oak/rosariov2_01 | yes | 187362 (199) | present | 13821 | 940.4 | 68.0 | RV2-DROP |
| 29 | v2_02_medium | K14 held-out | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/V2_02_medium | yes | 23490 (200) | present | 2348 | 117.3 | 50.0 | E-V2CAM, E-V2ROW |
| 30 | 4s_20210107_131 | never-run | 4Seasons extended (heldout) | E:/wm_ext/4seasons_oak/fourseasons_2021-01-07_13-12-23 | yes | 794483 (2000) | present | 12023 | 397.2 | 33.0 | S-EXTR (handled), S-FEED, S-FQ (candidate), S-GTSCALE (handled), S-VSCALE (inherited) |
| 31 | 4s_20210225_141 | never-run | 4Seasons extended (heldout) | E:/wm_ext/4seasons_oak/fourseasons_2021-02-25_14-16-43 | yes | 830650 (2000) | present | 12540 | 415.3 | 33.1 | S-EXTR (handled), S-FEED, S-FQ (candidate), S-GTGAP, S-GTSCALE (handled), S-HOOD, S-VSCALE (inherited) |
| 32 | madmax_F5 | never-run | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_F5 | yes | 40510 (100) | present | 4837 | 405.0 | 83.7 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 33 | madmax_F1 | never-run | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_F1 | yes | 45161 (100) | present | 5608 | 451.5 | 80.5 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 34 | madmax_F4 | never-run | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_F4 | yes | 51104 (100) | present | 6158 | 510.9 | 83.0 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 35 | madmax_D2 | never-run | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_D2 | yes | 42593 (100) | present | 5733 | 425.8 | 74.3 | M-DROW, M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited) |
| 36 | madmax_D0 | never-run | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_D0 | yes | 50009 (100) | present | 6601 | 499.6 | 75.7 | M-DROW, M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited) |
| 37 | madmax_D1 | never-run | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_D1 | yes | 51298 (100) | present | 6974 | 512.9 | 73.6 | M-DROW, M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited) |
| 38 | madmax_B4 | never-run | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_B4 | yes | 106939 (100) | present | 13773 | 1069.2 | 77.6 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 39 | madmax_B6 | never-run | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_B6 | yes | 110004 (100) | present | 14788 | 1099.9 | 74.4 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 40 | 4s_20200324_174 | never-run | 4Seasons board (development) | D:/wm/datasets/4seasons_oak/fourseasons_2020-03-24_17-45-31 | yes | 850634 (2000) | present | 12873 | 425.3 | 33.0 | S-EXTR (handled), S-FEED, S-GTGAP, S-GTSCALE (handled), S-HOOD, S-SAME, S-VSCALE (candidate) |
| 41 | 4s_20200407_102 | never-run | 4Seasons board (development) | D:/wm/datasets/4seasons_oak/fourseasons_2020-04-07_10-20-32 | yes | 1031912 (2000) | present | 15524 | 515.9 | 33.2 | S-EXTR (handled), S-FEED, S-GTGAP, S-GTSCALE (handled), S-HOOD, S-SAME, S-VSCALE (candidate) |
| 42 | 4s_20210225_135 | never-run | 4Seasons board (development) | D:/wm/datasets/4seasons_oak/fourseasons_2021-02-25_13-51-57 | yes | 951418 (2000) | present | 14359 | 475.7 | 33.1 | S-EXTR (handled), S-FEED, S-GTGAP, S-GTSCALE (handled), S-HOOD, S-SAME, S-VSCALE (candidate) |
| 43 | kitti_03 | never-run | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_03 | yes | none (KITTI odometry) | present | 801 | 82.7 | 103.4 | K-CAL, K-DOFF, K-GTS, VROW |
| 44 | kitti_04 | never-run | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_04 | yes | none (KITTI odometry) | present | 271 | 28.1 | 104.1 | K-CAL, K-DOFF |
| 45 | kitti_08 | never-run | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_08 | yes | none (KITTI odometry) | present | 4071 | 422.9 | 103.9 | K-CAL, K-GTS |
| 46 | kitti_10 | never-run | KITTI (development) | D:/wm/datasets/kitti_oak/kitti_10 | yes | none (KITTI odometry) | present | 1201 | 124.5 | 103.7 | K-CAL |
| 47 | madmax_A6 | never-run | MadMax board (development) | D:/wm/datasets/madmax_oak/madmax_A6 | yes | 69533 (100) | present | 9357 | 695.2 | 74.3 | M-A6SUN, M-EXTRTILT, M-FDROP, M-GTSAW, M-GTSD, M-TD, VROW |
| 48 | madmax_B0 | never-run | MadMax board (development) | D:/wm/datasets/madmax_oak/madmax_B0 | yes | 314825 (100) | present | 41101 | 3148.1 | 76.6 | M-EXTRTILT, M-FDROP, M-GTSAW, M-GTSD, M-TD, VROW |
| 49 | madmax_G2 | never-run | MadMax board (development) | D:/wm/datasets/madmax_oak/madmax_G2 | yes | 63352 (100) | present | 8218 | 633.4 | 77.1 | M-EXTRTILT, M-FDROP, M-G2ROW, M-GTSAW, M-GTSD, M-TD |
| 50 | madmax_H0 | never-run | MadMax board (development) | D:/wm/datasets/madmax_oak/madmax_H0 | yes | 33147 (100) | present | 4257 | 331.4 | 77.9 | M-EXTRTILT, M-FDROP, M-GTSAW, M-GTSD, M-TD, VROW |
| 51 | v2_01_easy | never-run | EuRoC (development) | C:/Users/USER/Downloads/euroc_oak/V2_01_easy | yes | 22800 (200) | present | 2280 | 114.0 | 50.0 | E-V2CAM, E-V2ROW |
| 52 | 4s_20201222_115 | SEALED | 4Seasons extended (heldout) | E:/wm_ext/4seasons_oak/fourseasons_2020-12-22_11-54-24 | yes | 646125 (2000) | present | 9775 | 323.0 | 33.1 | S-EXTR (handled), S-FEED, S-FQ (candidate), S-GTGAP, S-GTSCALE (handled), S-HOOD, S-VSCALE (inherited) |
| 53 | 4s_20201007_144 | SEALED | 4Seasons extended (heldout) | E:/wm_ext/4seasons_oak/fourseasons_2020-10-07_14-47-51 | yes | 662963 (2000) | present | 10030 | 331.4 | 33.0 | S-EXTR (handled), S-FEED, S-FQ (candidate), S-GTSCALE (handled), S-HOOD, S-VSCALE (inherited) |
| 54 | 4s_20200612_112 | SEALED | 4Seasons extended (heldout) | E:/wm_ext/4seasons_oak/fourseasons_2020-06-12_11-26-43 | yes | 1136265 (2000) | present | 17194 | 568.1 | 33.0 | S-EXTR (handled), S-FEED, S-FQ (candidate), S-GTSCALE (handled), S-HOOD, S-VSCALE (inherited) |
| 55 | madmax_A1 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_A1 | yes | 42599 (100) | present | 5565 | 425.9 | 76.5 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 56 | madmax_A2 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_A2 | yes | 42150 (100) | present | 5550 | 421.4 | 75.9 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 57 | madmax_A0 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_A0 | yes | 45726 (100) | present | 6167 | 457.2 | 74.1 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 58 | madmax_A5 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_A5 | yes | 45530 (100) | present | 5962 | 455.2 | 76.4 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 59 | madmax_G0 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_G0 | yes | 58698 (100) | present | 7742 | 586.7 | 75.8 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-G01CAL, M-GTSAW, M-GTSD, M-TD (inherited) |
| 60 | madmax_G1 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_G1 | yes | 58252 (100) | present | 7543 | 582.3 | 77.2 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-G01CAL, M-GTSAW, M-GTSD, M-TD (inherited) |
| 61 | madmax_B1 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_B1 | yes | 68010 (100) | present | 8914 | 679.5 | 76.2 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 62 | madmax_A3 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_A3 | yes | 59054 (100) | present | 7887 | 590.4 | 74.9 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 63 | madmax_B3 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_B3 | yes | 74207 (100) | present | 9946 | 742.0 | 74.6 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 64 | madmax_A4 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_A4 | yes | 66882 (100) | present | 8758 | 668.5 | 76.3 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 65 | madmax_B5 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_B5 | yes | 95171 (100) | present | 12589 | 951.6 | 75.6 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited), VROW |
| 66 | madmax_C0 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_C0 | yes | 120852 (100) | present | 15790 | 1208.4 | 76.5 | M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited) |
| 67 | madmax_E2 | SEALED | MadMax extended (heldout) | E:/wm_ext/madmax_oak/madmax_E2 | yes | 151820 (100) | present | 20369 | 1518.1 | 74.5 | M-DROW, M-EXTRTILT (inherited), M-FDROP, M-FPS, M-GTSAW, M-GTSD, M-TD (inherited) |

OAK recordings (no GT; FLAG NO-GT):

| tag | dataset | calib | IMU samples (Hz) | frames | duration s | xclean run dirs |
|---|---|---|---|---:|---:|---:|
| ver2_oak_161834 | E:/wm_oak/oak_20260329_161834 | yes | 39506 (198) | 5991 | 199.6 | 5 |
| ver2_oak_161206 | E:/wm_oak/oak_20260329_161206 | yes | 53559 (197) | 8122 | 271.7 | 1 |
| ver2_oak_20260329_160543 | E:/wm_oak/oak_20260329_160543 | yes | 63180 (198) | 9581 | 319.3 | 0 |
| oak_20260327_164424 | E:/wm_oak/oak_20260327_164424 | yes | 26866 (198) | 4068 | 135.8 | 0 |
| oak_20260327_164002 | E:/wm_oak/oak_20260327_164002 | yes | 23099 (198) | 3497 | 116.7 | 0 |

## 2. Measures per row (fixed now; computed by board_eval.py)

### 2.1 Run of record, attempts and validity
- **Draws.** Every row (GT and OAK) is run **n = 3** times, paced (section 4.3). Draw 1 is `board_<row>`; draws 2 and 3 are
  `board2_<row>` and `board3_<row>`. For a seen row (phase C), draw 1 may be a valid paced K14 run `k14_<row>` of the SAME
  `_engine_md5` (section 5).
- **Run of record** of a draw = its FIRST valid attempt. Validity is checked by script BEFORE any accuracy number is read, under
  validity protocol v2: --pin --reference, pre-flight clean, feeder jitter p99 <= 1 ms and slope 1.000 ± 0.001, no
  non-allow-listed process above 5 % of a core, no non-allowed GPU app, engine tree pinned, fence clean.
- A draw's run is **ok** only if, in addition (board_eval.py `run_ok`):
  - its `_engine_md5` is the declared final md5 and its mode is paced;
  - `not_for_bars` is false;
  - the loop worker STARTED and exited cleanly: harness `checks.p13.worker` present with `worker = slam`, exit code 0, not
    terminated, and `checks.p13.inactive_why` None (correction 8.9; `loop_on()` alone is True even when the worker failed to
    import and the run silently ran without it);
  - `XCLEAN_ENGINE = p1`, and its engine env flags (minus the witness-registry path) EQUAL those of the session's reference
    sentinel (one configuration for the whole board). The block also refuses to start when any `XCLEAN_*`, `XTEN_*` or `XONE_*`
    variable is set in the launching shell, so only the snapshot's DEFAULT configuration can run: a configuration of record
    (e.g. P1_3_PREREG Amendment F2's `head = mv`) must therefore be the frozen snapshot's default;
  - (v4, correction 8.23) its FRAME PATH is the registered one: engine_summary.json `p1.lat` EQUALS the snapshot's registered
    set = the `LAT_DEFAULT` line of the snapshot's own xclean/front/fastpaths.py (the default with a visible GPU, the GPU front
    `gf` included; p13b_fix1a: e1, e2, e3, g1, gf, m1, mf), and, when that set holds `gf`, engine_summary `gpu_front` is set (the
    GPU front was initialised). The env flags cannot show this: the snapshot drops `gf` SILENTLY when no GPU is visible
    (CUDA_VISIBLE_DEVICES -1 or empty, or a cupy / driver failure at start). The paced phases therefore also refuse any
    `CUDA_VISIBLE_DEVICES` in the launching shell (5.1);
  - it is not SUSPECT (5.2: a run taken after the last good sentinel and before a sentinel that failed).
- **The block applies the same test** (v3, correction 8.17; v4 adds the frame path, 8.23): board_block.sh state_of() checks the
  oracle flags, XCLEAN_ENGINE, the loop worker's start, the env flags against the session reference and the frame path exactly
  as run_ok does, so the block never keeps a run the evaluator would reject. A K14 run is reused as draw 1 only when it passes
  that test (block and evaluator alike).
- A row is **counted** when all n draws have an ok run of record and the row is not a system failure (2.6, 5.3).
- **Attempts.** Every attempt that is not a run of record is MOVED (never deleted) to `board_void/<name>__<suffix>` (`a<k>` =
  attempt k, `prev` = left by an earlier launch, `incomplete` = no harness.json after a machine event, `suspect` = parked after a
  sentinel failed, 5.2). Every attempt gets a start and an end line in D:/xoneslam_runs/xclean/board_ledger.tsv, and every
  attempt whose harness created its run dir also gets a `board_attempt.json` (class, detail, launch md5, start / end, harness
  exit code, hang limit; v4: written to a temp file and renamed, correction 8.25). An attempt whose harness died before creating
  the dir has no record: it is its two ledger lines only (v3 said "every attempt", correction 8.30). **The ledger's end line is
  the class of record** (v4, 8.25): the block's counts use it, a relaunch never re-classifies an attempt that has one, and the
  evaluator takes an attempt's class from it (the record is only a copy, read when it parses). board_eval.py lists EVERY attempt
  of every draw of a row (kept dir, parked dirs, ledger-only ends, each once), with its class, reasons and own B5 (correction
  8.7); sentinel attempts are listed apart, from their dirs AND the ledger (8.26). None is used for a claim. **Draw-1 runs reused
  from K14** (v4.1, correction 8.32; v4 said here that k14_block.sh deletes its VOID attempts, which k14_block.sh v2 no longer
  does): k14_block.sh MOVES every K14 attempt that is not a run of record to D:/xoneslam_runs/xclean/k14_void/<name>__<suffix>
  (a<k>, prev, incomplete, md5_<x> = another candidate's run) and writes a start and an end line per attempt in
  D:/xoneslam_runs/xclean/k14_ledger.tsv (time, event, name, row, attempt, md5, class, detail). For a row whose draw 1 is the
  reused k14_<row>, board_eval.py lists that run's K14 attempts (every class: VOID, FAIL, MD5, CONFIG, INCOMPLETE, another md5;
  the parked dirs beside them) as the draw-1 hygiene record, and counts its FAILs of the final md5 exactly as the board's own
  (the row's system failure, 2.6 / 5.3, and C3a); board_block.sh's FAIL count of the row includes them too. A FAIL of the
  final md5 in any OTHER K14 candidate arm of the row (k14_<row> when not reused, k14ls_, k14n2_, k14n3_) is listed and fails
  C3a on that row ("(K14 run)"), as the sentinel's and the lock-step run's FAILs do (8.26, 8.20); it is not a board draw, so it
  makes no system failure. A reused draw 1 whose ledger is absent, or has no valid line of that run, is flagged
  (k14_hygiene_unverifiable) and C3a cannot pass on its table: never a silent pass.
- **Classes** (decided by script, FAIL first because a crashed engine never writes engine_summary.json and so has no md5,
  correction 8.1): FAIL = an engine / worker / recorder / feeder exit code, a fence error, a harness exit without harness.json,
  or a HANG (no exit within 1.5 x stream + 600 s; the block then kills the process tree it started); VOID = machine (validity
  protocol v2); MD5 = another engine md5 (paced rounds: aborts the block, exit 4: the snapshot changed; phase E: the attempt is
  parked and is one of the lock-step run's 2 attempts, 2.10, correction 8.30); CONFIG (v3, correction 8.17) = a paced run
  that is not of the board's one configuration: oracle flags, XCLEAN_ENGINE != p1, the loop worker never started (a worker
  that started and then failed is a crash: FAIL), env flags != the session reference, or (v4, 8.23) the frame path != the
  registered one / the GPU front not initialised (aborts the block, exit 4: a configuration fault is the same on every row, so
  going on would only spend unseen rows). One exception (v4, correction 8.29): a CONFIG whose ONLY reason is a loop worker that
  never started (the harness could not import xclean.slam.worker, which may be transient: a locked numba cache, a DLL or memory
  error) gets ONE more attempt on that draw; a second such CONFIG on the draw, or any other CONFIG, aborts. INCOMPLETE = no
  harness.json (or no run dir) found at a relaunch for an attempt WITHOUT an end line (the block itself died: a machine event,
  not an attempt; v4: an attempt whose end line exists keeps that class, so a FAIL is never re-recorded INCOMPLETE, 8.25 /
  8.30); SUSPECT (v3, correction 8.14) = a run of record taken
  after the last good sentinel and before a sentinel that failed (5.2): parked as `board_void/<name>__suspect`, its
  board_attempt.json class SUSPECT (`class_at_end` keeps its class), a `suspect` line in the ledger; never a run of record.

### 2.2 Scorers
- **Live scorer, frozen:** xclean/live/score_live.py md5 **72c9c0936282561b406bdadfd42dc8fe** (GTF1, DT1).
  - It equals p13b_fix1a's copy. board_eval.py REFUSES to run with any other md5.
  - The run's own live_score.json (written by the snapshot's scorer) is only compared with the frozen score; a mismatch is
    reported.
- **Board scorer (the C4 headline, correction 8.4):** the frozen board rule, REPO/scorer_v3 (the versioned honest scorer; the
  md5 of each of its files is recorded in every output), driven through runs/xclean/score/score_arm.py UNCHANGED by
  runs/xclean/board/board_rule_runner.py. The runner changes only three process-level things: the output dir (under
  runs/xclean/board/data/board_rule), the scorer's temp dir, and the worker pool (in process, so the two redirections reach the
  jobs). Identity test (6.2): on x1 dev8 n = 1 it reproduces runs/xclean/score/out/x1_dev8_n1 on all 15 (row, board) entries and
  every tally.

### 2.3 Accuracy (B5 and the board reading)
- **B5 LIVE-MAP:** frame level, SE(3) Umeyama in cam0 over the full GT span, 20 ms association, a zero-order hold over every
  released frame, no hygiene. Also reported: coverage, and ATE over unheld frames only.
- **B5 LIVE-ODOM:** the same run's LIVE-ODOM stream, scored identically (the K13 / K14 evaluator's definition).
- **within-run ratio** = LIVE-MAP / LIVE-ODOM of the same draw (R19).
- A row's B5 and within-run ratio are the MEDIAN over its n draws; the per-draw values and the range are shown.
- **registered ratio** = the row's median LIVE-MAP B5 / the frozen P1.2 paced reference's B5, only where such a reference already
  exists (k14ref_, k13ref_, now5_; valid, paced, md5 1cdc6b2c or d30dc0fe). Reported.
- **Board reading (C4 headline):** each draw's LIVE-MAP pose published at each keyframe stamp the engine passed to the loop
  worker is exported as a board draw (`bkf<d>_ver2_<row>`: trajectory_kf.csv; trajectory_frames.csv = LIVE-MAP at every
  published frame, the board's fallback level; summary.json `coasted_pct` = 100 x the fraction of frames without a tracked
  LIVE-MAP message, raised just above 10 % when 2.6 calls the draw LOST only through an unrecovered loss). scorer_v3 then applies
  the board's own rule: KF level, each (our draw, ORB-SLAM3 draw) pair on its common window, honest lost-draw hygiene (ours:
  coasted > 10 %; ORB-SLAM3: final atlas > 1 map), the range rule and the forfeit policy. Our draws are the LIVE stream (causal,
  never revised); ORB-SLAM3's are its offline final keyframe files: that asymmetry is against us and is printed with every C4
  line (open issue 1).
- A draw of a counted row that has no published LIVE-MAP pose (at the KF stamps) cannot be exported: OUR side gave the board
  rule nothing, so the row is a forfeit "ORB3*" (ours lost) with the reason, never a 'no verdict' row (v3, correction 8.19).
  'no verdict' stays for a row the scorer could not score (a data error, symmetric).

### 2.4 Latency and drops (B1, B2; paced runs only)
- **Latency:** p50 / p95 / p99 / max in ms and in camera periods, over ALL frames, with a dropped frame counted as +inf.
- **Bars, NOW:**
  - rows with period >= 50 ms: p95 <= 1 period, p99 <= 2, max <= 4;
  - rows with period < 40 ms (4Seasons, OAK; "30 Hz rows"): p95 <= 1.5 periods; p99 <= 2 and max <= 4 are read as for the other
    rows (reading R-B1hz);
  - frame-loop gap <= 3 periods.
- **B2:** dropped <= 1 % (<= 3 % on 30 Hz rows), <= 2 consecutive, no 1 s window with > 10 % dropped, latency drift (p95 of the
  last 10 % / first 10 %) <= 1.2.
- **NEXT B1:** p95 <= 1, p99 <= 2, gap <= 2 periods. Reported, not claimed.
- **Reading R-B1max (literal):** the max clause uses +inf for a dropped frame, so any engine drop fails it. The max over
  PUBLISHED frames is reported beside it as a secondary line, never as the claim.
- **Reading R-B1g, declared AFTER the dry test on design rows** (section 6.3), and applied identically to every row:
  - the frame-loop gap of a published frame b = (receipt b − receipt of the previous published frame) − (release b − release of
    the frame released just before b) + 1 period;
  - a frame the engine dropped still counts against it; a stamp gap IN THE DATASET does not;
  - on a uniform stream with no drop it equals the literal receipt gap.
- **Why R-B1g:** madmax_F3 reads 10.0 periods literally, which is exactly its dataset stamp gap (register M-FDROP), against an
  engine excess of 0.33. The literal value is reported beside it.

### 2.5 Robustness and continuity (B4)
- **Reported:**
  - unsupported %, no-message frames, state-not-ok frames, rel2s failures;
  - longest unsupported run, first-pose frame;
  - LIVE-ODOM jumps, LIVE-MAP jumps, unflagged jumps;
  - **live_map_jumps_unflagged_not_odom** (R17), and LIVE-MAP corrected frames.
- **Bar reading:**
  - unsupported <= 2 % per non-flagged row, <= 10 % on v2_03;
  - rows whose GT the frozen scorer flags (GT_POSITION_ONLY = Rosario v1; GT_FAULTS = MadMax) are owner decisions (LIVE_BARS
    amendment 2026-10-23 ~18:10). Both readings are reported there (full; GT-free = no message + state not ok), and NEITHER
    counts as a pass. They are "pending", outside the bar;
  - no-GT rows (OAK) use the state-only reading;
  - registered GT codes are shown and do not change the reading.

### 2.6 Lost tracking
- A draw is **LOST** if more than 10 % of its frames have no LIVE-MAP message, or a state other than tracking / relocalized / init,
  or are degraded, or if the loss is unrecovered. This is the board-honesty hygiene, applied in live form.
- Also reported:
  - loss events;
  - the longest recovery in seconds (L5: relocalization <= 1 s after the images are trackable is reported per event; it is not
    a bar, because "trackable" is not measurable here);
  - an unrecovered loss at the end.
- A SYSTEM FAILURE row (2 FAIL attempts of the final md5, 5.3) counts as LOST in every claim (4.4).

### 2.7 Usability
- **Tier 1 (eval-tiers rule, on every row and EVERY draw):** diag/usability tool 1.2, unchanged, on each draw's LIVE-MAP pose
  published at each keyframe stamp the engine passed to the loop worker (U1 frame states from the LIVE-MAP messages).
  - U1: lost, tracked-frame gap, span.
  - U2: KF-matched revisit gap beyond GT, h and v, worst cluster p90, own vertical. It is `-g_w` of P1 where the IMU was
    admitted (EuRoC); the pitch-free causal camera -y on ground rigs.
  - U3: KF outliers at the adopted k = 2.0, m = 4.5 (runs/xsesto/usability/bars.json).
  - U4: climbs vs pitch on ground rigs.
  - U5: full-run SE(3) and 4-DoF ATE.
  - The bars are report.py's adopted constants: U1 gap < 2 s, span >= 0.95; U2 v <= 0.2 m, h <= max(1 m, kitti_00 1.1 m); U3 0;
    U4 0.
  - U2b uses ORB-SLAM3's per-cluster median over its non-lost draws, computed the same way on its draws.
  - These are DISCLOSURE bars (adopted 2026-10-09: "failures disclosed"). They are counted, never a pass / fail claim.
- **Tier 2:** runs/xclean/oak/analyse_run.py (no GT; depth-at-poses check at returns to the start) on every board run, after
  the battery, on the CPU. Reported only.
- **Tier 3:** builder maps only for rows shown on the website (maps track, not this report).

### 2.8 Loop-worker facts (from p13_worker.json and p13_worker_jobs.jsonl)
- **Counts:** jobs, published events, loops carried, provisional events, errors, shed, dropped, attempts and the stage funnel
  (floor / cold / pcm / admitted / informative), episodes, merges, anchor fallbacks, edge mismatches.
- **Latency (paced only, worker clock "wall"):**
  - **target latency** = event publish − closing-KF stamp (R15; L3 bar p95 <= 2.0 s);
  - decision age = finish − KF stamp; queue wait.
- **Backlog:** max, plus first-10 % vs last-10 % mean.
  - Reading of L3: bounded = max <= 2 KFs (the NEXT B1b number, reported);
  - no upward drift = last-10 % mean <= first-10 % mean + 1 KF;
  - no silent drop = every dropped job is logged.
- **Slew-done latency** (R15, reported): from the first frame of a map epoch to the first frame whose pending correction is
  <= 0.1 mm / 0.001 deg.
- **K13-f:**
  - segments created / merged;
  - an unmerged segment that revisits an older one is a reported defect. A revisit means: a later-segment frame within the U2
    radius R of an older-segment frame that is >= 5 s earlier and >= 10 R of GT path back.

### 2.9 Resources (B6 NOW, L4)
- **Engine tree:** host <= 4 GB, VRAM delta <= 2 GB, growth <= 100 KB per KF, CPU <= 2 cores (10 s max).
- **Whole tree** (engine + worker + recorder): peak memory, growth and CPU, reported as the headroom left for the robot.

### 2.10 Frame path untouched (K-c on the board)
- Lock-step runs of the final snapshot (`boardls_`, of the declared lock-step md5) and of frozen P1.2 (`boardlsref_`,
  engines/p12b_int2, md5 1cdc6b2c); both md5s are checked (correction 8.11).
- Every lock-step attempt is accounted like a paced one (v3, correction 8.20): ledger start / end lines, board_attempt.json,
  parks (`a<k>`, `prev`, `incomplete`), a hang watchdog (3 x stream + 1200 s; K13 measured 0.6-1.7 x stream with 12 concurrent
  runs) that kills the process tree the block started and classes the attempt FAIL, and at most 2 attempts per lock-step run
  (a FAIL, VOID or MD5 attempt is one of them; an MD5 here does not abort the parallel phase, 8.30).
  A FAIL of the final lock-step snapshot is a crash of the system (C3a, listed as "<row> (lock-step)"); a row whose lock-step
  run has no valid attempt fails C10.
- LIVE-ODOM must be bit-identical on every frame, except the stream's last frame, where |dp| <= 1e-6 m (R16).
- The same lock-step runs give the deterministic second reading for R19-B (section 4.3; reported).

### 2.11 GT faults and no-GT rows
- **GT gaps:** the scorer's 20 ms association excludes frames without GT identically for every system. The usability tool
  never interpolates across a GT gap, and excludes plausibility faults (spikes / bumps / vertical steps).
- **Position-only GT (Rosario v1):** the rotation clause is not applicable, and translation is taken at its lower bound
  (flagged). B4 is pending, as in 2.5.
- **MadMax GT_FAULTS** (M-GTSAW sawtooth, attitude vs gyro): B4 pending, as in 2.5. B5 and U2 are reported with the flag.
- **Registered GT codes** are shown beside the row. They never re-score, drop or excuse a row: S-GTGAP, S-RTK on 4s_101, K-GTS,
  E-V2CAM, E-V1ORI, RV2-DROP, M-GTSD (LIMITATION, kept), and the K13-diagnosis note "EuRoC MH_01 GT attitude fault"
  (unregistered; shown on mh_01_easy).
- **kitti_00:** the U2 h bar has the adopted GT-noise allowance of 1.1 m (BARS.md). No other row has one.
- **No-GT rows (OAK):**
  - no accuracy number of any kind is computed or shown;
  - reported: B1 / B2, B4 (state only), B6, worker facts, U1 / U3 (step rule) / U4, and tier 2 (analyse_run.py);
  - "never present OAK numbers as accuracy".

## 3. Baselines (AVAILABILITY only; the two exposures disclosed in the header are the only baseline numbers of unseen rows seen)

### 3.1 Systems and fairness labels

| system | files used | per row | FAIRNESS label | role |
|---|---|---|---|---|
| **xclean final (ours)** | board_<row>, board2_<row>, board3_<row> recorders | n = 3 paced draws (4.3) | **LIVE, causal, PACED (real time), validity v2** | the system |
| ORB-SLAM3 untuned (local bake-off) | the board's own draws: keyframe files traj_k.csv.kf.csv (C4 headline, board rule) and per-frame shutdown files traj_k.csv (C4f, secondary); roots = the frozen board's (scorer_v3 local_fair: draws, draws_untuned_extfix for the 6 4Seasons rows, draws_untuned_b0n5 for B0; extended rows draws_untuned_ext) | 5 draws, every board row | **OFFLINE final trajectory** (written at shutdown after global BA and loops; unpaced; 37-68 ms / frame, not real time on 3 of 4 MadMax rows). NOT ORB-SLAM3's live stream: LIVE_BARS TARGET's live ORB3 driver does not exist | the owner's board comparison (headline) |
| ORB-SLAM3 tuned reference | the conservative board of scorer_v3 (worst verdict over local_fair / official / reference); frame level: draws_official (KITTI, official yaml), draws_reference (Rosario CIFASIS, 4Seasons DM-VIO crop, MadMax authors' config) | 5 draws where present | **OFFLINE final trajectory, TUNED per dataset** | secondary board (stated beside, never the headline) |
| OKVIS2-X (our reproduction, CPU) | okvi1c (causal), okvi1n (non-causal), okvi1 (final BA) trajectory_frames.csv | n = 1, all 67 rows | okvi1c = **CAUSAL per-frame stream, UNPACED, CPU under K = 4 load at below-normal priority** (accuracy only, no latency claim); okvi1 / okvi1n = offline | LIVE_BARS NEXT head-to-head (C6) |
| x1 = xsesto frozen | kit/trajectory_frames_causal.csv (x1, x12, x13, x14, x15) | 0-5 draws | **CAUSAL per-frame stream, lock-step UNPACED** (x1 is not real time: RTF ~2.6) | LIVE_BARS B5 NOW form (C5) |
| P1.2 frozen (xclean VIO core) | now5_ / k13ref_ / k14ref_ (paced), p12_k12_ / k13lsref_ / boardlsref_ (lock-step) | seen rows only (paced); every GT row (lock-step, phase E) | paced = **LIVE causal PACED, same harness** (the only like-for-like); lock-step = causal unpaced | registered ratio (seen rows), K-c identity (C10) |
| xtenslam v3 (paper 1) | gpe..gpe5 trajectory_frames.csv | 2-5 draws | **OFFLINE end-of-run** | context only, never a win / loss |
| DPVO (local) | bakeoff/dpvo traj.csv | 1 | OFFLINE, monocular (scale-aligned) | context only; not computed by board_eval |
| MAC-VO | mac1 | 10 rows | OFFLINE, over the real-time budget (1.75 fps) | ceiling reference only (realtime rule); not computed |

Scoring of every baseline:
- **C4 headline (ORB-SLAM3 untuned):** the board's own rule, scorer_v3 (2.2, 2.3 "board reading"), on both sides' keyframe
  files; the conservative (tuned) board beside it (C4t). This is the board-honesty rule's headline (correction 8.4).
- **Frame-level readings (C4f secondary; C5; C6):** each system's own per-frame file, zero-order held on the dataset's frame
  stamps, by the SAME frozen `score_frames` our live stream is scored by (a frame without its own output holds the last pose;
  frames before its first output are unscored); "own frames only" (has = ones; the baseline_freeze.py precedent) beside it.
  C5 and C6 are defined by LIVE_BARS at this level (B5 NOW: "x1 per-frame causal re-scored with the SAME live scorer"; NEXT:
  head-to-head vs OKVIS2-X causal). The frame-level ORB-SLAM3 reading is never used to claim a row (board-honesty rule).
- **Each system's own loss criterion** (board-honesty hygiene, symmetric):
  - ORB-SLAM3: final atlas > 1 map;
  - x1: coasted > 10 %;
  - OKVIS2-X: has no loss criterion in our tools, so it is counted lost when its outputs cover < 90 % of the frames;
  - ours: 2.6 (frame-level readings); coasted > 10 % plus the unrecovered-loss clause (board reading, 2.3).

### 3.2 Availability per row (file existence only)

| row | status | ORB3 untuned draws (root) | ORB3 tuned-ref draws | DPVO | OKVIS2-X vi: finalBA / non-causal / causal | MAC-VO | x1 causal draws | xtenslam v3 draws | P1.2 frozen runs (xclean) | P1.3 K13 run |
|---|---|---|---|---|---|---|---|---|---|---|
| madmax_E1 | design | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | now5(paced,VOID), k13ref(paced), p12_k12(lockstep) | k13(paced) |
| 4s_20200324_173 | design | 5 (draws_untuned_extfix) | reference 5 | y | y / y / y | - | 2 | 5 | k13ref(paced), p12_k12(lockstep) | k13(paced) |
| 4s_20210107_120 | design | 5 (draws_untuned_extfix) | reference 5 | y | y / y / y | - | 2 | 5 | k13ref(paced), p12_k12(lockstep) | k13(paced) |
| kitti_00 | design | 5 (draws) | official 5 | y | y / y / y | - | 3 | 5 | now5(paced), p12_k12(lockstep) | k13(paced) |
| kitti_01 | design | 5 (draws) | official 5 | y | y / y / y | y | 5 | 2 | now5(paced), p12_k12(lockstep) | k13(paced) |
| kitti_07 | design | 5 (draws) | official 5 | y | y / y / y | - | 3 | 5 | k13ref(paced), k13lsref(lockstep) | k13(paced), k13ls(lockstep) |
| madmax_F3 | design | 5 (draws) | reference 5 | y | y / y / y | - | 3 | 5 | now5(paced), p12_k12(lockstep) | k13(paced) |
| mh_01_easy | design | 5 (draws) | - | y | y / y / y | - | 3 | 5 | k13ref(paced), k13lsref(lockstep) | k13(paced), k13ls(lockstep) |
| mh_04_difficult | design | 5 (draws) | - | y | y / y / y | - | 3 | 5 | now5(paced), p12_k12(lockstep) | k13(paced) |
| rosario_03 | design | 5 (draws) | reference 5 | y | y / y / y | y | 3 | 5 | now5(paced), p12_k12(lockstep) | k13(paced) |
| rosario_04 | design | 5 (draws) | reference 5 | y | y / y / y | y | 3 | 5 | now5(paced), p12_k12(lockstep) | k13(paced) |
| v1_02_medium | design | 5 (draws) | - | y | y / y / y | y | 3 | 5 | now5(paced), p12_k12(lockstep) | k13(paced) |
| v2_03_difficult | design | 5 (draws) | - | y | y / y / y | y | 3 | 5 | now5(paced), p12_k12(lockstep) | k13(paced) |
| kitti_05 | K13 held-out (clean) | 5 (draws) | official 5 | y | y / y / y | - | 3 | 5 | k13ref(paced), k13lsref(lockstep) | k13(paced), k13ls(lockstep) |
| kitti_06 | K13 held-out (clean) | 5 (draws) | official 5 | y | y / y / y | - | 3 | 5 | k13ref(paced), k13lsref(lockstep) | k13(paced), k13ls(lockstep) |
| mh_03_medium | K13 held-out (clean) | 5 (draws) | - | y | y / y / y | - | 3 | 5 | k13ref(paced), k13lsref(lockstep) | k13(paced), k13ls(lockstep) |
| v1_01_easy | K13 held-out (clean) | 5 (draws) | - | y | y / y / y | - | 3 | 5 | k13ref(paced), k13lsref(lockstep) | k13(paced), k13ls(lockstep) |
| mh_05_difficult | K12-e (P1.2 only) | 5 (draws) | - | y | y / y / y | - | 3 | 5 | p12_k12(lockstep) | - |
| rosario_01 | K12-e (P1.2 only) | 5 (draws) | reference 5 | y | y / y / y | - | 3 | 5 | p12_k12(lockstep) | - |
| rosario_02 | K12-e (P1.2 only) | 5 (draws) | reference 5 | y | y / y / y | - | 3 | 5 | p12_k12(lockstep) | - |
| rosario_06 | K12-e (P1.2 only) | 5 (draws) | reference 5 | y | y / y / y | - | 3 | 5 | p12_k12(lockstep) | - |
| v1_03_difficult | K12-e (P1.2 only) | 5 (draws) | - | y | y / y / y | - | 3 | 5 | p12_k12(lockstep) | - |
| madmax_E0 | K14 held-out | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| 4s_20200612_101 | K14 held-out | 5 (draws_untuned_extfix) | reference 5 | y | y / y / y | - | 2 | 5 | - | - |
| kitti_02 | K14 held-out | 5 (draws) | official 5 | - | y / y / y | - | 3 | 5 | - | - |
| kitti_09 | K14 held-out | 5 (draws) | official 5 | y | y / y / y | - | 3 | 5 | - | - |
| mh_02_easy | K14 held-out | 5 (draws) | - | y | y / y / y | - | 3 | 5 | - | - |
| rosariov2_01 | K14 held-out | 5 (draws) | reference 5 | y | y / y / y | - | 1 | 5 | - | - |
| v2_02_medium | K14 held-out | 5 (draws) | - | y | y / y / y | - | 3 | 5 | - | - |
| 4s_20210107_131 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | y | 1 | 5 | - | - |
| 4s_20210225_141 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | - | 1 | 5 | - | - |
| madmax_F5 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | y | 3 | 5 | - | - |
| madmax_F1 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_F4 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_D2 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_D0 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_D1 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_B4 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_B6 | never-run | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| 4s_20200324_174 | never-run | 5 (draws_untuned_extfix) | reference 5 | y | y / y / y | - | 2 | 5 | - | - |
| 4s_20200407_102 | never-run | 5 (draws_untuned_extfix) | reference 5 | y | y / y / y | - | 1 | 5 | - | - |
| 4s_20210225_135 | never-run | 5 (draws_untuned_extfix) | reference 5 | y | y / y / y | - | 2 | 5 | - | - |
| kitti_03 | never-run | 5 (draws) | official 5 | y | y / y / y | - | 5 | 2 | - | - |
| kitti_04 | never-run | 5 (draws) | official 5 | y | y / y / y | y | 3 | 2 | - | - |
| kitti_08 | never-run | 5 (draws) | official 5 | y | y / y / y | - | 3 | 2 | - | - |
| kitti_10 | never-run | 5 (draws) | official 5 | y | y / y / y | - | 3 | 2 | - | - |
| madmax_A6 | never-run | 5 (draws) | reference 5 | y | y / y / y | - | 2 | 5 | - | - |
| madmax_B0 | never-run | 5 (draws_untuned_b0n5) | - | - | y / y / y | - | 0 | 5 | - | - |
| madmax_G2 | never-run | 5 (draws) | reference 5 | y | y / y / y | y | 2 | 5 | - | - |
| madmax_H0 | never-run | 5 (draws) | reference 5 | y | y / y / y | y | 3 | 5 | - | - |
| v2_01_easy | never-run | 5 (draws) | - | y | y / y / y | - | 3 | 5 | - | - |
| 4s_20201222_115 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 2 | 5 | - | - |
| 4s_20201007_144 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| 4s_20200612_112 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 1 | 5 | - | - |
| madmax_A1 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_A2 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_A0 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_A5 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_G0 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_G1 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_B1 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_A3 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_B3 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_A4 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_B5 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 3 | 5 | - | - |
| madmax_C0 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 2 | 5 | - | - |
| madmax_E2 | SEALED | 5 (draws_untuned_ext) | - | - | y / y / y | - | 0 | 5 | - | - |

**P1.2 references:** a valid paced P1.2 reference exists for all 13 design rows and all 4 K13-clean rows. now5_madmax_E1 is VOID,
but k13ref_madmax_E1 is valid. The 7 K14 rows get theirs from the K14 battery (k14ref_).
- No P1.2 paced reference exists, or will be made, for the 5 K12-e rows and the 38 unseen rows (section 5.4).

## 4. Claims the paper may make, with pass / fail rules declared now

### 4.1 Tables (never pooled for a claim)
- **U = the clean held-out test:** U-unsealed (22) and U-sealed (16), each separate, plus U-all (38).
  - **The headline claims are U-all's.**
- **K14 held-out (7):** separate. Each row's label is decided mechanically at evaluation (correction 8.12):
  - K14 ran it -- (v4.1, correction 8.33) ANY K14 trace of the row by any md5: a run dir of any K14 arm (k14_, k14ref_, k14ls_,
    k14lsref_, k14n2_, k14n3_, k14refn2_, k14refn3_), an attempt parked in k14_void/ (k14<arm>_<row>__<suffix>), or a line of
    the row in k14_ledger.tsv (v4 looked at four run-dir names only, so a K14 run parked into k14_void read as "never run") --
    and a final `_engine_md5` is among the md5s of its candidate arms: "K14 held-out for this config";
  - K14 ran it on another md5: "seen by an earlier config (K14 md5 ...)", the candidate md5(s) listed, or, when only the P1.2
    reference arm ever ran on it, "P1.2 reference arm only" with that md5 (the row was still run and scored by K14); a
    candidate-arm trace whose md5 nothing records (no harness.json, no ledger line) is "spent, md5 unknown": never shown as
    held-out for this config, never as never-run;
  - K14 never ran it (K14 stopped at stage 1 under P1_3_PREREG S1 and the system was frozen without another round): "never run
    before the board; held-out for this config". The block then runs such a row in phase A2, right after the 38 unseen rows and
    before any seen row (5.2). It stays in the K14 table (the owner's 07:40 ruling took the 7 rows out of the unseen set).
- **S = already-seen rows:** S-design (13, marked "design"), S-K13-clean (4), S-K12-e (5).
- **OAK (5, no GT):** separate; no accuracy claims.
- An "ALL GT rows" line is printed for completeness, design rows marked. It is never a claim.

### 4.2 Claims
Each claim is evaluated per table. **PASS needs every row of the table to be counted** (all n draws ok, no system failure,
2.1). A missing, VOID-only, suspect-only, incomplete, failed or not-evaluated row FAILS every claim of its table, and in every
head-to-head (C4, C4t, C4f, C6) it is ALSO a FORFEIT to the other system (ours lost), never dropped from the tally (corrections
8.2, 8.13). Both sanctions apply: the forfeit is in the tally, and the claim fails regardless of the tally. This holds for EVERY
claim with a PASS rule, the head-to-heads C4 (stage 1 and stage 2) and C6 and the identity claim C10 included (v2 applied it
to the others only; correction 8.13). **Each claim has ONE PASS field** in board.json (`pass`; None only for C7b "not
applicable"); **C4's PASS is its stage 1** (`pass` = `pass_stage1`); stage 2 is reported beside it as `pass_stage2`.
board_tables.md prints the PASS / FAIL of every claim per table. Bars use the n-draw reading:
- **quantity bars** (C1, C3b, C7c target latency and drift, C8): the median of the measured quantity over the n draws against
  the bar, i.e. the clause holds on a majority of the draws (4.3); "every draw passes" is reported beside it;
- **event claims** (C2, C3a, C7c silent drop): must hold on EVERY draw.

| id | claim (paper wording if PASS) | PASS rule (per table; every row counted) |
|---|---|---|
| **C1** | "runs in real time on every row" | every row passes ALL NOW B1 + B2 clauses of 2.4 (R-B1max literal, R-B1g, R-B1hz), each clause by the median reading. Reported beside it: rows passing on every draw, NEXT B1 count, the secondary "max over published frames" count |
| **C2** | "every map correction is flagged; the odometry stream is never corrected" | live_map_jumps_unflagged_not_odom = 0 on every draw of every row (R17). LIVE-ODOM jump count reported; "no odometry jumps" is claimed only for rows with 0 on every draw |
| **C3a** | "tracks every row to the end" | no LOST draw (2.6), no unrecovered loss on any draw, and NO FAIL attempt (crash / hang) of the final md5 on any row of the table, sentinel runs included for the sentinel's own row (mh_04_difficult; v1_02_medium if the fallback sentinel ran, 5.2) and the final lock-step snapshot's phase-E runs included (2.10) |
| **C3b** | B4 NOW unsupported bar | <= 2 % (v2_03 <= 10 %) on every NON-flagged row by the median reading; flagged rows listed as "pending owner" with both readings |
| **C4** | vs untuned ORB-SLAM3 (owner's board) — **HEADLINE** | **the board's own rule** (scorer_v3: KF level, common window, honest lost-draw hygiene, range rule, forfeits; 2.3) on our n = 3 paced draws vs ORB-SLAM3's 5 untuned draws; a row without a counted run = "ORB3*" (forfeit). Tally = the board's own: ours / tie / ORB3 over all rows, and over the both-tracked subset (rows whose ORB-SLAM3 status is not lost / diverged; our forfeits stay in it as losses). **Stage 1** (untuned target) = **C4's PASS** (`pass`, = `pass_stage1`): every row of the table counted AND ours > ORB3 over all rows AND ours > ORB3 on the both-tracked subset. **Stage 2** (`pass_stage2`): every row counted AND no row lost = zero ORB3 rows (forfeits included) and no 'no verdict' row. Reported: the per-row medians and ranges, the conservative (tuned) board C4t (dev rows; secondary), and the frame-level live reading C4f (secondary, never a claim). Fairness label 3.1 printed with it |
| **C5** | "more accurate live than our previous system" | per row: our median LIVE-MAP B5 <= max(x1 worst kept draw, 1.05 x x1 kept median) (LIVE_BARS B5 NOW), AND the median over rows of ours / x1 kept median <= 1.0. Rows without a kept x1 draw: "no x1" (listed) |
| **C6** | vs OKVIS2-X causal | per row: median ratio < 0.95 win, > 1.05 loss, else tie (the 5 % band is the K13 / K14 materiality); forfeits as C4. **PASS = every row of the table counted AND wins >= losses over every row of the table** (LIVE_BARS NEXT wording; computed, corrections 8.2, 8.13) |
| **C7a** | "loop closure does no harm" | median within-run ratio <= 1.05 on every GT row (a row without a within-run ratio fails). On seen rows the registered ratio is reported |
| **C7b** | "loop closure pays where the route revisits" | over the rows with E1 oracle_r <= 0.95 (computed mechanically from GT at evaluation on draw 1: the first U2 revisit instant t*; the draw's own LIVE-ODOM replaced by GT from t* in its own alignment; the same scorer): median of their median within-run ratios <= 0.90; every GT row counted with its oracle computed. No eligible row in a table: "not applicable". Reported beside it: how many rows meet their E1 bar 1 − 0.5 (1 − oracle_r) |
| **C7c** | "the loop worker keeps up live" | on every row: target-latency p95 <= 2.0 s (L3 / R15) and no upward backlog drift (median reading), no silent drop on any draw. Reported: backlog max vs 2 KFs, slew-done latency |
| **C8** | "fits the NOW resource budget" | B6 NOW on every row (engine tree), each sub-clause by the median reading; whole-tree figures reported (L4) |
| **C9** | usability (tier 1) | DISCLOSURE only: rows where EVERY draw passes all absolute U1-U4 bars, ours vs ORB-SLAM3 (rows where every one of its draws passes). Tier 2 reported |
| **C10** | "the loop worker never touches the frame path" | every row of the table counted (correction 8.13) AND lock-step LIVE-ODOM identity (R16) with frozen P1.2 on every GT row, the lock-step runs being of the declared lock-step md5 (`boardls_`) and of P1.2 md5 1cdc6b2c (`boardlsref_`); lock-step FAIL attempts are listed (2.10) |
| **C11** | relocalization / segments | reported only: segments, merges, unmerged-with-revisit defects, recovery times |

### 4.3 n rule: n = 3 paced draws per row for every claim (correction 8.3)
- **Registered: n = 3** on every row, GT and OAK. Sources, all binding: LIVE_BARS B5 NOW ("n = 3 for any claim"; C5 IS that bar);
  the xclean directive ("n = 1 on the dev set while developing; n = 3 only for claims"); the board-honesty rule's sufficient
  evidence ("n >= 3 both arms"). v1 registered n = 1 with targeted top-ups and acknowledged none of these rules; that was a defect.
- **Round order:** round d runs draw d of every row in the frozen order (A, A2, B, C, D), so draw 1 of the whole board (the clean
  held-out first) exists before any draw 2. An interrupted block resumes where it stopped.
- **Readings at n = 3:** medians for B5 and the ratios, the median reading for quantity bars, every draw for event claims (4.2),
  the range over the 3 draws for C4 / C4t / C4f, the median ratio for C6, majority-lost for forfeits (the board's rule).
- **Operating point of C4:** with iid draws the range rule gives a chance "win" (and, symmetrically, a chance "loss") with
  probability 1 / C(n + 5, n): 1/6 at n = 1 vs 5 (v1), **1/56 at n = 3 vs 5 (registered)**, 1/252 at 5 vs 5.
- **The untuned target's n = 5.** The staged untuned target (2026-09-30) agreed "FINAL at n = 5 on our side" for the xtenslam
  FINAL bank. For xclean the directive (2026-10-23) sets "n = 3 only for claims", and the goal lock of 2026-10-23 put ORB-SLAM3
  full-board parity out of scope. This prereg therefore registers n = 3 for C4 too and listed n = 5 as an OWNER DECISION (open
  issue 1): had the owner ruled it before the block starts, the block would run with NDRAW=5 (rounds 4 and 5 on every row, +
  about 20.2 h) and the evaluator with --n-claim 5. **CLOSED** by the owner ruling line at the end of this section (n = 3; v4
  states it here too, correction 8.30).
- **The cheaper v1 plan** (n = 1, plus 2 more draws only where R19 or R19-B flags a row as noise-limited; about 8-9 h instead of
  about 31.5-36 h) was an OWNER option too (open issue 2). **CLOSED** by the same ruling: not taken.
- **How an owner ruling on n is recorded:** a line appended to this section, starting at column 0 with the words OWNER RULING,
  then `n = <n>`, then the date in parentheses (`(YYYY-MM-DD HH:MM, ...`) and the owner's words. board_block.sh (NDRAW) and
  board_eval.py (--n-claim) refuse any n other than 3 unless such a line exists for that n; a mention of the marker inside a
  sentence, like this one, does not count.
- **R19 and R19-B** (registered vs within-run ratio across 1.05; paced vs deterministic lock-step LIVE-MAP verdicts) are REPORTED
  per row. They would trigger extra draws only under the n = 1 option, which the ruling below closed. The lock-step reading is
  never a verdict itself.

OWNER RULING n = 3 (2026-10-27 13:00 IST, recorded by the lead from the owner's chat message): "Keep n = 3 for every claim on the final board, as registered." This closes open issues 1 and 2 of section 7: neither n = 5 nor the n = 1 plan is taken.

### 4.4 What is reported when a claim fails (nothing is dropped)
- Every claim appears in the paper's claim table with PASS / FAIL, its counts and the failing rows.
- A failed claim is stated as failed ("what the system does not do yet"). It is never re-tested with a changed bar, reading or
  row set, and never restricted to a subset after the fact.
- **Mechanically (v3, correction 8.18):** claims are always taken over ALL rows of each table of the frozen rows.json. An
  evaluation with `--only` (a subset, for diagnosis) treats every row it left out as "not evaluated" (uncounted: every claim of
  that table fails, every head-to-head forfeits it), records `only` / `subset` in its output, writes data/board_subset.json by
  default and refuses to write the board of record data/board.json; phase F refuses a subset board.json.
- No row is removed:
  - VOID-only rows are shown as "no valid run", with every attempt's B5 (timing void); they fail every claim and forfeit every
    head-to-head;
  - **SYSTEM FAILURE rows** (2 FAIL attempts of the final md5 on the row, over all its draws and every launch of the block,
    5.3) count as LOST: they fail every claim, forfeit every head-to-head, and are never retried;
  - a row with exactly one FAIL attempt and n ok draws is counted; the crash is listed on the row and fails C3a.
- GT-fault and dataset-defect flags are shown beside the numbers. They excuse nothing, except the two readings declared in 2.4 /
  2.5, which apply to every row.
- Design rows are always marked.
- The unseen table is never merged with seen rows to rescue a claim.

## 5. Run plan

### 5.1 Protocol
- Paced, validity protocol v2, --pin --reference, under PACED_ACTIVE, one engine at a time, in an exclusive machine window. The
  monitor's LOAD_HOLD is honoured.
- The protocol-v3 isolation test FAILED (2026-10-26 21:34), so no background work runs during the paced rounds. The lead closes
  nothing of the owner's.
- Default config (loop ON); the block refuses to start with any engine flag in the launching shell (2.1). GPU front as frozen:
  (v4, correction 8.23) the paced phases refuse any CUDA_VISIBLE_DEVICES in the launching shell, and every paced run's frame
  path (engine_summary p1.lat, gpu_front) is checked against the snapshot's registered set by the block and the evaluator (2.1).
- CPU-only work (evaluation, lock-step) waits for the window to end.
- **Integrity checks before anything starts** (any failure = exit 2, nothing launched; `PHASES=X` runs them alone): rows.json
  sha1; `_engine_md5` of SNAP recomputed and equal to SNAP_MD5; SNAP_LS's md5 recorded and, when SNAP_LS differs from SNAP, the
  twin check (every .py identical except live/harness.py, whose only change may be added ALLOW-list `.exe` lines);
  engines/p12b_int2 = 1cdc6b2c; NDRAW = 3 or an owner-ruling line in this file (4.3); no engine flag in the environment; (v4)
  no CUDA_VISIBLE_DEVICES when a paced phase or X is selected (8.23); the snapshot's front/fastpaths.py has exactly one
  LAT_DEFAULT line (the registered frame path, logged on the start line, 8.23); the registered block parameters MAXTRY = 6,
  RETRY = 3, NPAR = 6, HANG_ADD_S = 600, HANG_LS_ADD_S = 1200, PAUSE_S = 60 and the run dir / ledger / board_void paths are not
  changed by the environment (8.27; they are logged on the start line). Only the block's own dry test (BOARD_SELFTEST) may change
  them, and it is refused on the run dir and with any snapshot that is not the FAKE harness.

### 5.2 Order: draw 1 of every row first; in each round never-run rows FIRST (A), then unspent K14 rows (A2), K12-e (B), seen rows (C), OAK (D)
Why unseen rows go first rather than interleaved:
- **(i) Interruption risk.** Two power outages hit in 24 h (2026-10-26 16:50, 2026-10-27 06:00). The clean held-out test is the
  report's one irreplaceable measurement, so its first draw is taken first, and its draws 2 and 3 open rounds 2 and 3.
- **(ii) Seen rows add little.** Their behaviour under the final config is largely known from K14. When the md5 matches, their
  K14 runs are reused as draw 1 at no cost.
- **(iii) Drift control is cheaper without interleaving.** Interleaving seen rows would buy only that, and here it is bought
  directly: a **session sentinel** (mh_04_difficult, 102 s) runs at the start of every launch, after every 10 row-draws and at
  the end.
  - The FIRST valid sentinel of the final md5 is the witness reference (`--reference`); its dir is recorded in
    `board_sentinel_ref_<md5>.txt`, and every later sentinel is judged against THAT dir (`--witness-ref`, ±20 %, protocol v2 b).
    v1 judged later sentinels against the registry entry, which a `board_mh_04_difficult --reference` run overwrites
    (correction 8.8). Its env flags are the session's one configuration (2.1). (v4.1, correction 8.31) A sentinel instance
    whose attempt ended valid but whose `sentinel` ledger line (and, for the `--reference` instance, the reference file) was
    never written -- the block died after the run, or the run was found valid at the relaunch -- gets them at the relaunch,
    before the opening sentinel (resolve_sentinels, ledger order): so the FIRST valid instance stays the witness reference,
    and board_eval.py lists every such instance in sentinel_instances (rebuilt from its valid end line when the line is
    missing).
  - **The window** (v3, corrections 8.14 / 8.16): "the row-draws since the last good sentinel" is read from the LEDGER (every
    paced row-draw end line of the final md5 after the last valid sentinel end line), so it survives a relaunch. **Cadence:** a
    sentinel runs once 10 row-draws in the window got an attempt; it is checked after every row-draw in the rounds, the retry
    pass and phase F alike; a draw skipped as already valid gets no attempt and does not count. The closing sentinel runs only
    when a row-draw ran since the last good sentinel. (v4, correction 8.24) An attempt an earlier launch left WITHOUT its end
    line (the power failed during the run or its scoring), whose park was interrupted, or whose record is unreadable is
    resolved at the relaunch BEFORE the opening sentinel, so a run of record found then is inside that sentinel's window.
  - **A sentinel instance** (its own `bsent<time>_<row>` name) gets its own budget: 3 VOID attempts and 2 FAILs. If it has no
    valid run within that budget (ledger `sentinel` line STOP_VOID / STOP_FAIL), every run of record in the window becomes
    SUSPECT (2.1): a `suspect` ledger line, board_attempt.json class SUSPECT, the dir parked as `board_void/<name>__suspect`,
    a line in board_suspect.txt; then the block stops (exit 3).
  - **Re-run:** the lead re-launches the block (ONLY is not needed) once the machine is clean; the relaunch's opening sentinel
    is that check. The suspect draws have no run dir any more, so the rounds re-run them. (v4, correction 8.24) The suspects
    are RE-DERIVED from the whole ledger at every launch and after every sentinel failure: a failed instance is its STOP line,
    or its 3rd VOID / 2nd FAIL / 3rd INCOMPLETE end line when the block died before that line (the missing STOP line is then
    written), and its window is every valid row-draw end line between the last good sentinel before it and it. A power cut
    in the middle of the marking therefore loses no suspect; every suspect still in place is parked (an unreadable record is
    first rebuilt from the ledger, 8.25). The first runs stay in board_void/ and are
    reported as suspect (board_eval.py lists them per row and never uses one; a kept run the ledger marks suspect is not ok).
  - **No lock-out** (v3, correction 8.15): sentinel FAILs are counted per instance, never over launches, so every relaunch gets
    a fresh sentinel. Every sentinel FAIL of the final md5 still fails C3a on the sentinel's row.
  - **Fallback sentinel** (correction 8.15): a sentinel row whose LAST TWO instances each stopped with FAIL x2 (an engine crash
    on that row, not a machine fault) is replaced by the registered fallback sentinel row **v1_02_medium** (design row, EuRoC,
    C: NVMe, 85 s), with its own witness reference (`board_sentinel_ref_<md5>_v1_02_medium.txt`, first valid instance). If the
    fallback's last two instances also stopped with FAIL x2, the block exits 6 and 5.3 decides (a crash fix restarts the board
    on the fixed md5, or the board is reported as it stands: every row without a run is "missing" and fails every claim). No
    ledger edit or dir move by hand is ever part of the procedure.
- **Within phase A:** a family round-robin (EuRoC, KITTI, Rosario, 4Seasons, MadMax; registry order inside a family), so an
  interrupted block still covers every family. Sealed rows are interleaved by the same rule. The order is frozen in rows.json.
- **Phase A2** (correction 8.12): a K14 row with no K14 artefact at block start is run right after phase A in every round
  (whenever phase C is selected), so it is first run by the board before any seen row. (v4.1, correction 8.33) "No K14
  artefact" = no K14 trace at all, as in 4.1: no run dir of any K14 arm, nothing of the row in k14_void/, no line of it in
  k14_ledger.tsv; a row with any trace is a seen row (phase C).

### 5.3 Retries and failures (board_block.sh v2; the ledger makes every count survive a relaunch)
- **VOID** (machine): up to 6 VOID attempts per draw in the rounds, each after the pre-flight (up to 90 min) and the LOAD_HOLD
  wait; then one retry pass raises the budget to 9 for every draw still without a valid run. The first valid attempt is the run
  of record (attempts stop there, so there is no choice among valid runs).
- **FAIL** (correction 8.1): engine / worker / recorder / feeder exit code, fence error, harness exit without harness.json, or a
  HANG. The FAIL test comes FIRST (a crashed engine has engine_md5 None). One FAIL gets another attempt (it may be a machine
  fault). **The second FAIL of the final md5 on a row, counted over all its draws and every launch, makes the row a reported
  SYSTEM FAILURE: no further attempt on any of its draws, in any pass**; it counts as lost (4.4). (v4.1, correction 8.32) When
  draw 1 is the reused K14 run k14_<row>, the FAIL attempts k14_block.sh recorded for that run in k14_ledger.tsv (its first end
  line per attempt; a leftover found at a K14 launch as prev_FAIL too) are FAILs of that draw: the block and the evaluator both
  count them.
- **HANG watchdog** (correction 8.11): an attempt with no exit within 1.5 x stream + 600 s (madmax_B0: 89 min against a normal
  53 min) is a FAIL; the block kills the process tree IT started (taskkill /T on its own subshell) and goes on.
- **INCOMPLETE** (no harness.json, or no run dir, found at a relaunch for an attempt without an end line: the block itself
  died): parked, not an attempt; after 3 on the same draw the draw is left without a run (missing). An attempt whose end line
  exists keeps its class (v4, 8.25).
- **MD5** (a run launched from SNAP reports another md5): the snapshot changed under the block; the paced rounds abort (exit 4).
  In phase E it is parked and counts as one of the lock-step run's 2 attempts (2.10).
- **CONFIG** (v3, correction 8.17; 2.1; v4 adds the frame path, 8.23): the run is not of the board's one configuration; the
  block aborts (exit 4). The opening sentinel (a design row) is the first paced run of every launch, so a configuration fault
  stops the block before any unseen row runs. Exception (v4, 8.29): a CONFIG whose only reason is a loop worker that never
  started gets one more attempt on that draw (a transient import failure would otherwise stop the unattended board); the
  second on the draw aborts. The CONFIG attempt is parked `a<k>` and listed, never a run of record.
- **Exit codes of the block:** 0 done; 2 refused at integrity (nothing launched); 3 a sentinel failed (5.2); 4 MD5 / CONFIG
  abort; 5 phase F without the full board.json of SNAP_MD5; 6 both sentinel rows failed (5.2).
- If a row is VOID in every attempt because its E: (USB) dataset feeds with jitter, it may be copied byte-identically (md5
  checked) to D: and retried. That is the only data handling allowed.
- A PACED_ACTIVE written by an earlier launch of this block that is STALE is removed by the relaunch; any other PACED_ACTIVE /
  TIMING_ACTIVE is waited for, with a log line every 10 min. (v4, correction 8.28) The lock records the block's winpid AND
  that process's start time; it is stale only when no harness of this board (a python whose command line runs
  xclean.live.harness with `--out <run dir>/b...`) is still running AND (the lock was written before the last boot, or the
  recorded process is gone, or its PID now belongs to another process: another start time). A lock without a usable PID and
  no reboot since is waited for (logged); v3 judged by the PID alone, so a PID reused after a reboot or an empty PID waited
  forever without a log line, and a block whose bash died while its harness still ran was judged stale.
- **Registered parameters** (v4, correction 8.27): MAXTRY = 6, RETRY = 3, NPAR = 6, HANG_ADD_S = 600, HANG_LS_ADD_S = 1200,
  PAUSE_S = 60 and the run dir / ledger / board_void paths; another value inherited from the launching shell is refused at
  integrity (exit 2), so a dry-test value left in a shell (e.g. HANG_ADD_S = 8: a 50-s hang limit on kitti_04's 28-s stream)
  cannot reach the board.
- **Post-freeze fixes** (the finish line allows crash / hang / real-time fixes only):
  - the board of record is ONE `_engine_md5`;
  - if a fix is made, the board restarts on the fixed md5 for ALL rows and draws. The pre-fix runs are reported in an appendix
    ("pre-fix md5") and never mixed into a claim.
  - The alternative is to keep the frozen md5 and report the failing row as a failure. That is the lead's call, recorded with
    its reason.

### 5.4 P1.2 references: NOT needed, and none are run
The loop effect is judged by the within-run ratio, for three reasons:
- its paced noise cancels (R19);
- K13 showed it is the conservative reading where the two ratios disagreed: v2_03 0.607 registered vs 0.859 within, 4s_173 0.829
  vs 0.923;
- that LIVE-ODOM IS the P1.2 frame path is checked on every row by the phase-E lock-step identity (C10).

Cost and coverage:
- A P1.2 paced reference per unseen and K12-e row would add ~7.3 h of exclusive time and no decision (time-first rule).
- The registered ratio is still reported where a reference exists (24 seen rows).

### 5.5 OAK rows (phase D, no GT)
- **Recordings:** ver2_oak_161834 (199.6 s), ver2_oak_161206 (271.7 s), ver2_oak_20260329_160543 (319.3 s), oak_20260327_164424
  (135.8 s), oak_20260327_164002 (116.7 s). n = 3 draws each, like every row.
- **Measured:** B1 / B2 / B4-state / B6 / worker / U1 / U3 / U4, plus tier 2 (analyse_run.py).
- **Calibration:** the measured imu_noise (CHANGES OAK-IMU-NOISE) is already in the E:/wm_oak calibs.
- **The owner's live walk** (the on-device demo; its static on-device smoke is
  D:/xoneslam_runs/xclean/oak_live_static_p13_20261027) is reported separately as the live demo, run on the same final
  snapshot. Never as accuracy.

### 5.6 Phases after the exclusive window
- **Phase E (CPU, parallel, no PACED_ACTIVE):**
  - lock-step `boardls_<row>` (SNAP_LS, judged against SNAP_LS's md5) and `boardlsref_<row>` (engines/p12b_int2, judged against
    1cdc6b2c), every GT row, longest first (correction 8.6: v1 judged both against SNAP's md5, so every P1.2 run looked stale and
    was re-run on each relaunch); each run with the hang watchdog, the ledger and at most 2 attempts (2.10, correction 8.20);
  - up to NPAR = 6 rows (12 processes) at once, CUDA_VISIBLE_DEVICES=-1, NUMBA_NUM_THREADS=1, under the monitor's safety limits.
- **Evaluation:** board_eval.py on the CPU writes data/board.json and data/board_tables.md, including the board-rule run
  (data/board_rule/board_<md5>_n3/). Then analyse_run.py (tier 2) runs on every board run.
- **Phase F** exists only under the owner's n = 1 option (4.3): the R19-B top-ups read from board.json; a missing board.json or
  one of another md5 ABORTS the phase (exit 5) instead of silently running no top-up (correction 8.11).

### 5.7 Wall-clock estimate (paced overhead measured on the 35 K13 / now5 paced runs: median 7 s, max 19 s per run; with the
pre-flight, ~35 s per row)
- **One round** (every row once): A 38 rows 6.84 h, B 5 rows 0.43 h, C 24 rows 2.48 h, D 5 rows 0.34 h = **10.09 h**.
- **Round 1:** 7.61 h if K14 ran the final md5 (C's draw 1 reused), else 10.09 h. **Rounds 2 and 3:** 10.09 h each.
- **Sentinels:** ~23 x 2.3 min, about 0.9 h (216 row-draws / 10, plus the opening and closing ones; v3 adds one opening
  sentinel per relaunch and none for draws skipped as already valid).
- **VOID retries:** expect about 10-15 % of runs, mostly on 30 Hz 4Seasons rows with Windows services (K13: 4s_120 VOID x4); about
  +2.8-4.5 h.
- **=> Exclusive window at n = 3: about 31.5-33 h** if C's draw 1 is reused, **about 34-36 h** if not. It may be split over
  several sittings: the block resumes, and the sentinel controls drift across them.
  - The longest single run is madmax_B0 (3148 s, 52 min).
- **Owner options (4.3), CLOSED by the 13:00 ruling (n = 3), kept for the record:** n = 1 (v1 plan) about 8-9 h plus phase F
  (2-3.5 h); n = 5: + about 20.2 h (rounds 4-5 on every row).
- **Phase E:** about 2-2.5 h CPU. The critical path is B0 lock-step at ~1.7 x stream; 12 concurrent lock-step runs took 0.6-1.7
  x stream in K13.
- **Evaluation:** the dry tests took 50 s (6 rows, n = 1) and 61 s (n = 3 logic test), the board rule included; the full board
  at n = 3 is expected at 10-30 min.
- **Per-row order and estimates (one round; rounds 2 and 3 repeat it; A2 = the K14 rows of phase C that K14 never ran):**

| order | phase | row | tag | status | drive | stream s | est. wall min (stream + 35 s) |
|---:|---|---|---|---|---|---:|---:|
| 1 | A_unseen | v2_01_easy | ver2_v2_01_easy | never-run unsealed | C: | 114 | 2.5 |
| 2 | A_unseen | kitti_03 | ver2_kitti_03 | never-run unsealed | D: | 83 | 2.0 |
| 3 | A_unseen | 4s_20200324_174 | ver2_4s_20200324_174 | never-run unsealed | D: | 425 | 7.7 |
| 4 | A_unseen | madmax_A6 | ver2_madmax_A6 | never-run unsealed | D: | 695 | 12.2 |
| 5 | A_unseen | kitti_04 | ver2_kitti_04 | never-run unsealed | D: | 28 | 1.1 |
| 6 | A_unseen | 4s_20200407_102 | ver2_4s_20200407_102 | never-run unsealed | D: | 516 | 9.2 |
| 7 | A_unseen | madmax_B0 | ver2_madmax_B0 | never-run unsealed | D: | 3148 | 53.1 |
| 8 | A_unseen | kitti_08 | ver2_kitti_08 | never-run unsealed | D: | 423 | 7.6 |
| 9 | A_unseen | 4s_20210225_135 | ver2_4s_20210225_135 | never-run unsealed | D: | 476 | 8.5 |
| 10 | A_unseen | madmax_G2 | ver2_madmax_G2 | never-run unsealed | D: | 633 | 11.1 |
| 11 | A_unseen | kitti_10 | ver2_kitti_10 | never-run unsealed | D: | 124 | 2.7 |
| 12 | A_unseen | 4s_20200612_112 | ver2_4s_20200612_112 | sealed | E: | 568 | 10.1 |
| 13 | A_unseen | madmax_H0 | ver2_madmax_H0 | never-run unsealed | D: | 331 | 6.1 |
| 14 | A_unseen | 4s_20201007_144 | ver2_4s_20201007_144 | sealed | E: | 331 | 6.1 |
| 15 | A_unseen | madmax_A0 | ver2_madmax_A0 | sealed | E: | 457 | 8.2 |
| 16 | A_unseen | 4s_20201222_115 | ver2_4s_20201222_115 | sealed | E: | 323 | 6.0 |
| 17 | A_unseen | madmax_A1 | ver2_madmax_A1 | sealed | E: | 426 | 7.7 |
| 18 | A_unseen | 4s_20210107_131 | ver2_4s_20210107_131 | never-run unsealed | E: | 397 | 7.2 |
| 19 | A_unseen | madmax_A2 | ver2_madmax_A2 | sealed | E: | 421 | 7.6 |
| 20 | A_unseen | 4s_20210225_141 | ver2_4s_20210225_141 | never-run unsealed | E: | 415 | 7.5 |
| 21 | A_unseen | madmax_A3 | ver2_madmax_A3 | sealed | E: | 590 | 10.4 |
| 22 | A_unseen | madmax_A4 | ver2_madmax_A4 | sealed | E: | 668 | 11.7 |
| 23 | A_unseen | madmax_A5 | ver2_madmax_A5 | sealed | E: | 455 | 8.2 |
| 24 | A_unseen | madmax_B1 | ver2_madmax_B1 | sealed | E: | 680 | 11.9 |
| 25 | A_unseen | madmax_B3 | ver2_madmax_B3 | sealed | E: | 742 | 12.9 |
| 26 | A_unseen | madmax_B4 | ver2_madmax_B4 | never-run unsealed | E: | 1069 | 18.4 |
| 27 | A_unseen | madmax_B5 | ver2_madmax_B5 | sealed | E: | 952 | 16.4 |
| 28 | A_unseen | madmax_B6 | ver2_madmax_B6 | never-run unsealed | E: | 1100 | 18.9 |
| 29 | A_unseen | madmax_C0 | ver2_madmax_C0 | sealed | E: | 1208 | 20.7 |
| 30 | A_unseen | madmax_D0 | ver2_madmax_D0 | never-run unsealed | E: | 500 | 8.9 |
| 31 | A_unseen | madmax_D1 | ver2_madmax_D1 | never-run unsealed | E: | 513 | 9.1 |
| 32 | A_unseen | madmax_D2 | ver2_madmax_D2 | never-run unsealed | E: | 426 | 7.7 |
| 33 | A_unseen | madmax_E2 | ver2_madmax_E2 | sealed | E: | 1518 | 25.9 |
| 34 | A_unseen | madmax_F1 | ver2_madmax_F1 | never-run unsealed | E: | 451 | 8.1 |
| 35 | A_unseen | madmax_F4 | ver2_madmax_F4 | never-run unsealed | E: | 511 | 9.1 |
| 36 | A_unseen | madmax_F5 | ver2_madmax_F5 | never-run unsealed | E: | 405 | 7.3 |
| 37 | A_unseen | madmax_G0 | ver2_madmax_G0 | sealed | E: | 587 | 10.4 |
| 38 | A_unseen | madmax_G1 | ver2_madmax_G1 | sealed | E: | 582 | 10.3 |
| 39 | B_k12e | rosario_01 | ver2_rosario_01 | K12-e held-out | D: | 445 | 8.0 |
| 40 | B_k12e | rosario_02 | ver2_rosario_02 | K12-e held-out | D: | 270 | 5.1 |
| 41 | B_k12e | rosario_06 | ver2_rosario_06 | K12-e held-out | D: | 445 | 8.0 |
| 42 | B_k12e | v1_03_difficult | ver2_v1_03_difficult | K12-e held-out | C: | 107 | 2.4 |
| 43 | B_k12e | mh_05_difficult | ver2_mh_05_difficult | K12-e held-out | C: | 114 | 2.5 |
| 44 | C_seen | mh_04_difficult | ver2_mh_04_difficult | design | C: | 102 | 2.3 |
| 45 | C_seen | v2_03_difficult | ver2_v2_03_difficult | design | C: | 117 | 2.5 |
| 46 | C_seen | kitti_00 | ver2_kitti_00 | design | D: | 471 | 8.4 |
| 47 | C_seen | kitti_01 | ver2_kitti_01 | design | D: | 114 | 2.5 |
| 48 | C_seen | madmax_F3 | ver2_madmax_F3 | design | D: | 693 | 12.1 |
| 49 | C_seen | madmax_E1 | ver2_madmax_E1 | design | E: | 951 | 16.4 |
| 50 | C_seen | 4s_20210107_120 | ver2_4s_20210107_120 | design | D: | 454 | 8.2 |
| 51 | C_seen | rosario_04 | ver2_rosario_04 | design | D: | 165 | 3.3 |
| 52 | C_seen | rosario_03 | ver2_rosario_03 | design | D: | 200 | 3.9 |
| 53 | C_seen | 4s_20200324_173 | ver2_4s_20200324_173 | design | D: | 501 | 8.9 |
| 54 | C_seen | v1_02_medium | ver2_v1_02_medium | design | C: | 85 | 2.0 |
| 55 | C_seen | mh_01_easy | ver2_mh_01_easy | design | C: | 184 | 3.7 |
| 56 | C_seen | kitti_07 | ver2_kitti_07 | design | D: | 114 | 2.5 |
| 57 | C_seen | kitti_05 | ver2_kitti_05 | K13 held-out | D: | 288 | 5.4 |
| 58 | C_seen | kitti_06 | ver2_kitti_06 | K13 held-out | D: | 114 | 2.5 |
| 59 | C_seen | mh_03_medium | ver2_mh_03_medium | K13 held-out | C: | 135 | 2.8 |
| 60 | C_seen | v1_01_easy | ver2_v1_01_easy | K13 held-out | C: | 146 | 3.0 |
| 61 | C_seen | v2_02_medium | ver2_v2_02_medium | K14 held-out | C: | 117 | 2.5 |
| 62 | C_seen | mh_02_easy | ver2_mh_02_easy | K14 held-out | C: | 152 | 3.1 |
| 63 | C_seen | kitti_09 | ver2_kitti_09 | K14 held-out | D: | 165 | 3.3 |
| 64 | C_seen | kitti_02 | ver2_kitti_02 | K14 held-out | D: | 483 | 8.6 |
| 65 | C_seen | 4s_20200612_101 | ver2_4s_20200612_101 | K14 held-out | D: | 495 | 8.8 |
| 66 | C_seen | madmax_E0 | ver2_madmax_E0 | K14 held-out | E: | 893 | 15.5 |
| 67 | C_seen | rosariov2_01 | ver2_rosariov2_01 | K14 held-out | D: | 940 | 16.3 |
| 68 | D_oak_nogt | oak_161834 | ver2_oak_161834 | OAK own recording | E: | 200 | 3.9 |
| 69 | D_oak_nogt | oak_161206 | ver2_oak_161206 | OAK own recording | E: | 272 | 5.1 |
| 70 | D_oak_nogt | oak_20260329_160543 | ver2_oak_20260329_160543 | OAK own recording | E: | 319 | 5.9 |
| 71 | D_oak_nogt | oak_20260327_164424 | oak_20260327_164424 | OAK own recording | E: | 136 | 2.8 |
| 72 | D_oak_nogt | oak_20260327_164002 | oak_20260327_164002 | OAK own recording | E: | 117 | 2.5 |

- One round: A_unseen 38 rows 6.84 h, B_k12e 5 rows 0.43 h, C_seen 24 rows 2.48 h (of which the 7 K14 rows 0.97 h, run as A2
  if K14 never ran them), D_oak_nogt 5 rows 0.34 h (first attempts valid).

### 5.8 Command lines (to be filled with the frozen snapshot at freeze time; nothing else changes)
```
SNAP=D:/xoneslam_runs/xclean/engines/<final paced ALLOW twin> SNAP_MD5=<its _engine_md5> \
  SNAP_LS=D:/xoneslam_runs/xclean/engines/<final non-twin> bash D:/xoneslam_runs/xclean/board_block.sh       # rounds 1-3, A A2 B C D
SNAP=... SNAP_MD5=... SNAP_LS=... PHASES=E bash D:/xoneslam_runs/xclean/board_block.sh                      # lock-step
CUDA_VISIBLE_DEVICES=-1 C:/Users/USER/anaconda3/envs/cvcuda133/python.exe -W ignore runs/xclean/board/board_eval.py \
  --snap-md5 <SNAP_MD5> --snap-ls-md5 <SNAP_LS md5> --snap-dir <SNAP>                                    # evaluation
```
(Integrity only, nothing launched: `PHASES=X`.) The paced lines are launched from a shell WITHOUT CUDA_VISIBLE_DEVICES and
without any MAXTRY / RETRY / NPAR / HANG_* / PAUSE_S / BOARD_B / LEDGER / VOIDD / BOARD_SELFTEST (v4: the block refuses them,
8.23 / 8.27). The evaluator sets CUDA_VISIBLE_DEVICES=-1 for ITSELF only (CPU); `--snap-dir` must have SNAP_MD5's
`_engine_md5` (checked), and its front/fastpaths.py gives the registered frame path (v4, 8.23).

## 6. Scripts and the dry tests

### 6.1 Files

| file | md5 / sha1 at registration (v4.1) | role |
|---|---|---|
| D:/xoneslam_runs/xclean/board_block.sh | md5 b6d7da1e06e4001c7125ceb57e2eae8d | the block v4.1 (5.1-5.3, 5.6; v4.1: resolve_sentinels 8.31, the reused draw 1's K14 FAILs in fails_of 8.32, k14_spent from every K14 trace 8.33). `bash -n` OK |
| runs/xclean/board/board_eval.py | md5 d5887521b65db9a4084fba765d72c68a | the evaluator v4.1 (sections 2-4; v4.1: sentinel_instances 8.31, the K14 attempts of a reused draw 1 and the other K14 FAILs 8.32 (option --k14-ledger), k14_label from every K14 trace 8.33); refuses a live-scorer md5 other than the frozen one, an n other than 3 without the owner-ruling line (outside --dry, which runs already-run rows only and refuses never-run, K14 and sealed rows; correction 8.30), a --snap-dir whose `_engine_md5` is not --snap-md5 or that has no LAT_DEFAULT line (8.23), and --only onto data/board.json |
| runs/xclean/board/board_rule_runner.py | md5 c7569a94f2df19a9584bdc556003d3bb | the board-rule runner (2.2; unchanged in v3 and v4) |
| runs/xclean/board/census.py | md5 a8a9d75b58836621a3cfeaf609ea26ba | section 1 (unchanged in v3 and v4) |
| runs/xclean/board/rows.json | sha1 81b41675a688c0d77821e1c370c5a9323870072e | FROZEN row list, order, tags, tables, sentinel (unchanged since v1) |
| runs/xclean/board/dryrun_rows.json | sha1 c57a29bdbd173685cdc92ee066043ae96f05c1bc | dry test 1 map (design rows -> K13 dirs, n = 1; unchanged; v4 runs it with --snap-dir engines/p13b_fix1a) |
| runs/xclean/board/dryrun2_rows.json | sha1 2198b118b11c5a23a4aa8826c45d6b07861a9ac8 | dry test 2 map (n = 3 logic test, synthetic; v3: the FAIL fixtures repointed to fixtures/dry2/, correction 8.22; unchanged in v4) |
| runs/xclean/board/fixtures/dry2/ | each harness.json = D:/.../p12d_w3_e1crash_madmax_E1/harness.json byte for byte (md5 b9fcd116) | the dry-test-2 FAIL fixtures (3 dirs) |
| runs/xclean/board/dryrun3_logic.py | md5 c4453c1e86613c0b3abda99f84fe5468 | dry test 3 (v3; v4 adds T7-T9; v4.1 adds T10-T12): the claim / suspect / CONFIG / frame-path / record / export / sentinel-instance / K14-trace / K14-attempt logic on synthetic rows |
| runs/xclean/board/blocktest/ | selftest4.sh md5 0b5d6f46, selftest2v4.sh md5 025b8334, integrity4.sh md5 a7d51218, evalcheck4.py md5 faf27bfd, fake harness md5 f9159e15, fake front/fastpaths.py md5 0d5d01a4 (fake snapshot `_engine_md5` f7da674a); selftest3.sh, selftest2v3.sh and their v3 outputs kept; v4.1: selftest41.sh md5 3a44ae93c7c4686fbc8d330708becd41, integrity41.sh md5 35a41b58bd942b156c1da0f82ec94a94 (integrity4.sh with the start-line grep accepting v4.1), evalcheck41.py md5 bcb577e68d1086bcf4e8d1861e4226b9 | the block dry tests 1 and 2, the integrity checks and the evaluator's view of the power-cut dirs (v4), with their outputs (selftest4.out, selftest2v4.out, integrity4.out, evalcheck4.out: the v4 runs); v4.1: the B1 / B2 / B3 scenarios W1-W5 (selftest41.out on v4.1, selftest41_v4.out the same script on v4) and the v4.1 re-runs selftest4_v41.out, selftest2v4_v41.out, integrity4_v41.out, evalcheck4_v41.out, evalcheck41.out (evalcheck41_v4dirs.out: the v4.1 evaluator on the v4 block's W dirs); run from a copy (they create B_* dirs under SDIR) |
| runs/xclean/board/v4_20261027_1526/ | v4 prereg sha1 786c5720, block md5 3b1b619e, evaluator md5 c6b2c091, dryrun3_logic md5 5703c136, runner md5 c7569a94, census md5 a8a9d75b | the registered v4 files (audit), with the v4 blocktest/ scripts and outputs and the v4 dry-test outputs (data/) |
| runs/xclean/board/v3_20261027_1320/ | v3 prereg sha1 0a0f756e, block md5 580c0bbb, evaluator md5 657dc852, dryrun3_logic md5 98a6e650, runner md5 c7569a94, census md5 a8a9d75b | the registered v3 files (audit), with blocktest/ and the v3 dry-test outputs (data/) |
| runs/xclean/board/v1_20261027_1040/ | v1 prereg sha1 9c975565, block md5 7e9655ad, evaluator md5 234dde6a, census md5 fa55081c | the registered v1 files (audit) |
| runs/xclean/board/v2_20261027_1140/ | v2 prereg sha1 8a8dedae, block md5 d558245f, evaluator md5 d946c9ac, runner md5 c7569a94, census md5 a8a9d75b, dryrun2_rows sha1 e37e960a | the registered v2 files (audit) |

### 6.2 Checks run (CPU only; no engine, no harness, no never-run row)
- **v4.1 re-run of every check (2026-10-27 15:40-16:10 IST, CPU only, from copies in the session scratchpad, the v4 block /
  evaluator run beside the v4.1 ones on the same inputs; outputs in runs/xclean/board/blocktest/*_v41.out and selftest41.out,
  data/dryrun3_logic.txt; the v4 outputs stay in v4_20261027_1526/).**
  - **Integrity (`PHASES=X`, integrity41.sh = integrity4.sh with the start-line grep accepting "v4.1"):** every verdict and
    exit code identical to v4 (and the v4 block re-run on the same inputs reproduces the registered integrity4.out).
  - **Block dry test 1 (selftest4.sh, unchanged):** S0-S10 identical to the v4 block's re-run line for line after normalising
    paths, times, instance names and PIDs; the only differences are the order of phase E's parallel lock-step jobs (S8) and a
    PID number (S10). The v4 re-run reproduces the registered selftest4.out the same way.
  - **Block dry test 2 (selftest2v4.sh, unchanged):** the same harness-call sequence as v4, line for line.
  - **v4.1 additions (selftest41.sh, run on the v4 block AND the v4.1 block; output selftest41.out = the v4.1 run, the v4 run
    beside it in selftest41_v4.out):** W1-W3b (8.31): a sentinel instance found valid at the relaunch, or left without its
    sentinel line / reference file, gets them (v4: neither; in W2 / W3 / W3b v4 made the relaunch's NEW instance the witness
    reference); W4 (8.33): v4 put all 7 K14 rows in phase A2 although 4 had K14 traces (k14_void, ledger, a k14n2_ dir); v4.1 puts
    those 4 in C, the 3 untouched rows in A2, and k14_madmax_E00__a1 / k14_pin.json / k14_stage1_*.json / stale_locks spend
    nothing; W5 (8.32): fails_of counts the reused draw 1's K14 FAIL (v4 0 -> v4.1 1), a board FAIL on draw 2 then makes the row a
    SYSTEM FAILURE and the next draw is not attempted (v4 attempted it); a K14 run that is not reused counts nothing (both).
  - **The evaluator on the block's dirs (evalcheck4.py unchanged; evalcheck41.py new):** evalcheck4 identical to v4 (and the v4
    re-run = the registered evalcheck4.out). evalcheck41: on the v4.1 block's W dirs every valid sentinel instance is listed and
    the reference file names the first valid instance; on the v4 block's W dirs the v4.1 evaluator still lists the instances that
    lack their sentinel line (rebuilt from the end line) while the v4 evaluator lost them; k14_label agrees with the block's
    k14_spent on W4 (candidate traces without a recorded md5 are "spent, md5 unknown"); W5 counts the one FAIL of the final md5.
  - No fake-harness process was left running; every process the tests started is recorded in the scratchpad pid file.
- **v4 re-run of every check (2026-10-27 14:10-14:50 IST, CPU only, from copies in the session scratchpad; the v3 results
  below are kept as they were).** Outputs in runs/xclean/board/blocktest/ (integrity4.out, selftest4.out, selftest2v4.out,
  evalcheck4.out) and data/ (dryrun_board.json, dryrun2_board.json, dryrun3_logic.txt); the v3 outputs are in
  v3_20261027_1320/.
  - **Integrity (`PHASES=X`, integrity4.sh on a copy of the v4 block; nothing launched, no lock / ledger / board_void on D:):**
    the stand-in p13b_fix1a (60809bf3) with SNAP_LS = p13b_fix1 ACCEPTED, its start
    line logging MAXTRY 6, RETRY 3, NPAR 6, the hang limits, PAUSE_S 60 and the frame path e1, e2, e3, g1, gf, m1, mf; the v3
    refusals unchanged (SNAP_LS = p12b_int2 not a twin, a wrong SNAP_MD5, NDRAW = 1 and NDRAW = 5 without an owner-ruling line,
    an inherited XTEN_FOO=1: exit 2); REFUSED (exit 2) in v4: CUDA_VISIBLE_DEVICES = -1, empty and 0 (8.23); HANG_ADD_S = 8,
    HANG_LS_ADD_S = 8, MAXTRY = 2 / RETRY = 1 / PAUSE_S = 0, NPAR = 12, another LEDGER, VOIDD or BOARD_B (8.27); BOARD_SELFTEST
    with the real snapshot, with the fake snapshot on the run dir, and with a ledger on the run dir in msys path form;
    ACCEPTED: the registered values given explicitly, and BOARD_SELFTEST with the fake snapshot and B in the scratchpad (PHASES=X
    exits before sourcing it). The premise of 8.23, on a scratch copy of the stand-in's front/fastpaths.py: under
    CUDA_VISIBLE_DEVICES -1 and empty, gpu_visible() is False and the default set is e1, e2, e3, g1, m1, mf (gf dropped
    silently, no error).
  - **Block dry test 1 v4 (selftest4.sh, fake harness writing harness.json AND engine_summary.json, B / ledger / board_void in
    the scratchpad; output selftest4.out):** S0-S8 = the v3 scenarios with
    the SAME summary lines as v3 (compared after normalising paths, times and instance names), the v4 additions apart: S1 a COPY
    of k13_mh_04_difficult's harness.json and engine_summary.json with gf dropped and gpu_front None -> CONFIG (frame path + GPU
    front) while the original stays valid; now5_kitti_00 -> CONFIG marked [worker-only]; S2 rowK (frame path without gf) ->
    CONFIG, exit 4 (8.23), rowW (worker never started once, then valid) -> RC 0 with the CONFIG attempt parked a1, rowI (twice)
    -> exit 4 (8.29). S9 (the verifier's power-cut cases): N1 the run whose end line was lost is resolved before the
    relaunch's opening sentinel, so when that sentinel fails BOTH f1 and f2 are SUSPECT and both are re-run (attempt 2); N2
    after a cut inside the marking the relaunch writes f2's and f3's suspect lines, parks them and re-runs f1-f3; N2b a sentinel
    instance with 3 VOID end lines and no STOP line gets its STOP_VOID line at reconcile and its window is SUSPECT; N3 a 0-byte
    record of a suspect run is rebuilt from the ledger, the run parked SUSPECT and re-run; N3b a zero-filled record of a kept
    run is rewritten (attempt 1, valid) and the run kept, not re-run; N4 a FAIL without harness.json whose park was interrupted
    stays FAIL (record and ledger), is parked a1 and counted once by the block; N4b a HANG FAIL whose dir holds a valid
    harness.json is parked a1, never kept; N5 two dir-less sentinel FAILs are in the ledger; N6 an attempt with no dir and no
    end line gets an INCOMPLETE end line and the draw is run. S10 (8.28): dead winpid -> stale; a live PID with another start
    time -> stale; the same PID with its start time -> live; a v3 lock (no start time) of a live PID -> live; empty winpid ->
    live, and stale once the lock predates the last boot; a dead winpid while a (fake) harness of the board runs -> live, stale
    once it ended; take_lock removes a stale lock and writes winpid + start time; a foreign lock is waited for with a log line.
    Exit 0.
  - **Block dry test 2 v4 (selftest2v4.sh = selftest2v3.sh + the engine_summary.json the frame-path test reads):**
    the same harness-call sequence as v3, line for line (kitti_09 as A2, the two K14
    reuses, kitti_04 skipped as a SYSTEM FAILURE in every round, the cadence sentinel after the 10th attempted row-draw, the
    retry-pass call with budget 9 and the closing sentinel).
  - **The evaluator on the block's power-cut dirs (evalcheck4.py, output evalcheck4.out):** every N case is listed as the block left it, each attempt once: N1 / N2 / N3
    parked attempt 1 SUSPECT beside the kept valid attempt 2; N4 / N4b FAIL attempt 1 (evaluator FAIL count 1 = the block's) and
    the kept attempt 2; N5 two ledger-only sentinel FAILs of the final md5 (C3a input); N6 the INCOMPLETE attempt 1 (ledger only)
    and the kept attempt 2.
  - No fake-harness process was left running (checked by command line); every process the tests started is recorded in the
    scratchpad pid file.
- **Block integrity (`PHASES=X`, nothing launched):** the final-snapshot stand-in p13b_fix1a (60809bf3) with SNAP_LS = p13b_fix1
  (d4382668) was ACCEPTED (twin check ok); SNAP_LS = p12b_int2 was REFUSED (not a twin: 7 differing files); a wrong SNAP_MD5,
  NDRAW = 1 without the owner-ruling line and an inherited `XTEN_FOO=1` were each REFUSED (exit 2). No PACED_ACTIVE, ledger or
  board_void was created on D:. **Re-run on the v3 block (2026-10-27 12:58): identical verdicts**; the start line now also
  reads the sentinel rows from rows.json (mh_04_difficult 101.55 s, fallback v1_02_medium 85.45 s, every 10).
- **Block dry test 1 v3 (`BOARD_SELFTEST=blocktest/selftest3.sh`, 2026-10-27 12:51-12:58; output blocktest/selftest3.out):**
  B, the ledger and board_void pointed into the session scratchpad; the "snapshot" was a FAKE harness (blocktest/fakesnap: no
  engine, no data) that writes a scripted harness.json with the worker and env fields; the pre-flight stubbed; the P1.2 arm of
  phase E pointed at the fake snapshot, so no real harness could start. Results, all as expected:
  - S0: SDUR / the fallback's duration / EVERY read from the real rows.json; `num` refuses "" and "--witness-ref x" (8.21);
  - S1 state_of on REAL already-run dirs (read only): k13_mh_04 paced 60809bf3 valid (also with k13_kitti_00 as the env
    reference: equal flags); md5 1cdc6b2c -> MD5; lock-step mode -> VOID; p12d_w3_e1crash_madmax_E1 -> FAIL; p12_k12_mh_04
    lock-step -> valid; **now5_kitti_00 (P1.2 paced, no loop worker) -> CONFIG** (v2 called it valid);
  - S2 (the v2 cases, unchanged results): valid kept; VOID, VOID -> no run at budget 2, valid in the retry pass; crash, VOID,
    crash -> SYSTEM FAILURE, then no attempt on draw 2; HANG -> killed, FAIL, then valid; no harness.json -> FAIL, then valid;
    leftover without harness.json -> INCOMPLETE parked; a crashed leftover + 1 new crash -> SYSTEM FAILURE; relaunch -> skip;
    another md5 -> exit 4; **a run whose loop worker never started -> CONFIG, exit 4**;
  - S3: first sentinel `--reference` and recorded, second `--witness-ref <that dir>`; **a run whose env flags differ from that
    sentinel's -> CONFIG, exit 4**;
  - S4 (5 fake rows, EVERY 3, NDRAW 2): sentinels after the 3rd, 6th and 9th attempted row-draws and a closing one; **a
    relaunch over the complete board made ONE harness call (the opening sentinel)**, no closing sentinel (v2: 6 sentinel runs);
  - S5: the cadence sentinel after f1-f3 VOID x3 -> exit 3; f1-f3 got `suspect` ledger lines, class SUSPECT, parked
    `__suspect`, listed in board_suspect.txt; **the relaunch re-ran exactly f1-f3** (attempt 2), took a good cadence sentinel,
    ran f4, f5; its closing sentinel crashed twice -> exit 3 with **only f4, f5 suspect** (f1-f3 preceded a good sentinel);
    a third launch got a FRESH sentinel instance (no lock-out), re-ran f4, f5 and closed valid; reconcile parked a suspect run
    put back in place and left the re-run (attempt 2) untouched;
  - S6: sentinel crash x2 -> exit 3; the relaunch made 2 NEW attempts (v2: 0) -> exit 3; the third instance used the fallback
    v1_02_medium (`--reference`, own reference file); two fallback FAIL-stops later -> exit 6; mh_04_difficult's board-row FAIL
    count stayed 0 (sentinel FAILs are their own row in the ledger, C3a lists them);
  - S7 (EVERY 2, MAXTRY 1, 4 rows VOID once): sentinels inside the retry pass after every 2 attempted draws (v2: none);
  - S8 phase E: lock-step HANG -> killed (3 x stream + HANG_LS_ADD_S), FAIL, then valid; crash, crash -> no 3rd attempt, and
    none at a relaunch; an INCOMPLETE leftover parked, then valid; phase_e over 5 fake rows ran both arms as background jobs,
    every attempt in the ledger with its arm (ls / lsref). No fake-harness process was left running (checked).
- **Block dry test 2 v3 (round logic on the REAL rows.json, ONLY = 7 rows; paced and the sentinel stubbed, the stubs writing
  the ledger lines the real functions write; blocktest/selftest2v3.sh, output selftest2v3.out):** kitti_09 (no K14 artefact)
  ran as A2; draw 1 of mh_04 and mh_02_easy was reused from (fake) k14 runs that pass the full state_of test; kitti_04 (2 ledger
  FAILs) was skipped in all 3 rounds and the retry pass; draw 1 of every row before any draw 2; the cadence sentinel followed
  the 10th attempted row-draw; the one draw left without a run got one retry-pass call with budget 9; the closing sentinel
  followed it.
- **Board-rule runner identity test:** x1 dev8 (8 design rows, n = 1) through board_rule_runner.py reproduces
  runs/xclean/score/out/x1_dev8_n1/SCORE_subset.json exactly: 15 of 15 (row, board) entries identical, all tallies identical, the
  scorer_v3 file md5s identical; runs/xclean/score/out was not touched.
- **--dry refusal:** naming v2_01_easy / kitti_09 is refused (unchanged).
- **--only refusal (v3):** `--only mh_04_difficult --out data/board.json` is refused before any evaluation; a dry subset run
  (2 of the 7 dry-test-1 rows, written to the scratchpad) recorded `only` / `subset`, listed the 4 other design rows as not
  evaluated, failed every claim of S_design and printed the SUBSET mark on its claim table.

### 6.3 Dry tests of the evaluator (already-run rows only; v3 re-run 2026-10-27 13:05-13:10 on evaluator md5 657dc852, the
board-rule temp dir in the session scratchpad)
- **v4.1 re-run (2026-10-27 16:00-16:06 IST, evaluator md5 d5887521b65db9a4084fba765d72c68a, run from a byte-identical copy in the session
  scratchpad, the same arguments as v4, the v4 evaluator re-run beside it on the same inputs):** dry tests 1 and 2 give results
  IDENTICAL to v4 on every claim PASS field, tally, counted flag, system-failure flag, B5 median, within ratio, registered ratio,
  run_ok verdict and attempt class (0 differences, compared field by field, v4.1 vs v4 re-run and v4 re-run vs the registered
  data/dryrun_board.json / dryrun2_board.json, which are therefore kept as they are): the dry rows map their draws explicitly (no
  K14 reuse), no k14_ledger.tsv exists, and the new C3a field k14_hygiene_unverifiable_rows is empty. Dry test 3 now has 58 checks
  (T10 sentinel_instances, T11 the K14 label from every trace, T12 a reused draw 1's K14 attempts through evaluate_row), all
  PASS; T0-T9 print the same 42 lines as v4 (data/dryrun3_logic.txt).
- **v4 re-run (2026-10-27 14:15-14:20 IST, evaluator md5 c6b2c091d15afd700701b320c0bbb19a, run from a byte-identical copy in the session
  scratchpad, `--snap-dir D:/xoneslam_runs/xclean/engines/p13b_fix1a` (its `_engine_md5` = 60809bf3 = --snap-md5, checked;
  registered frame path e1, e2, e3, g1, gf, m1, mf), the board-rule temp dir in the scratchpad):** dry tests 1 and 2 give
  results IDENTICAL to v3 on every claim PASS field, tally, counted flag, system-failure flag, B5 median, within ratio,
  registered ratio, run_ok verdict and attempt class (0 differences, compared field by field with
  v3_20261027_1320/data/): every K13 run of record carries the registered frame path with the GPU front initialised, so 8.23
  removes none of them. Dry test 3 now has 42 checks (T7 frame path, T8 unreadable records, T9 dir-less sentinel
  instances added), all PASS. The v3 results below therefore stand.
- **Dry test 1 (`--n-claim 1`, data/dryrun_board.json):** the 6 design rows mapped to their K13 dirs (P1.3 p13b_fix1a paced;
  lock-step final = p13_fx1real_ p13b_fix1; lock-step P1.2 = p12_k12_), plus one already-run OAK P1.2 paced run for the no-GT
  path. CPU 59 s, board rule included (21 s; exports in data/kfboard_dry1_60809bf3, dry test 2's in kfboard_dry2_60809bf3,
  both rebuilt by v3). v3 results equal v2's on every number below; the claim table now prints ONE
  PASS / FAIL per claim: S_design C1 FAIL, C2 PASS, C3a PASS, C3b PASS, C4 FAIL (stage 1 and 2), C5 PASS, C6 PASS (3 / 0 / 3),
  C7a PASS, C7b PASS, C7c PASS, C8 FAIL, C10 PASS (dry rows, K13 config: not a claim).
  - **B5:** LIVE-MAP, LIVE-ODOM and the registered ratio reproduce runs/xclean/skeleton/allboxes/data/p13_k13.json EXACTLY on all
    6 rows (mh_04 0.0453 / 0.0685; v2_03 0.0873 / 0.1016; kitti_00 1.7925 / 3.0372; F3 0.9829 / 0.9829; rosario_04 1.4086 / 1.4086;
    4s_120 15.5319 / 18.0184), as in v1.
  - **Lock-step identity:** p13_fx1real vs p12_k12 PASS on 6 of 6, with both md5 checks.
  - **Run of record:** all 6 K13 runs are ok (loop worker started, env flags equal); the OAK run is not (VOID, P1.2 md5, no
    worker).
  - **Board reading (new C4 headline) on the K13 config:** ours / tie / ORB3 = 1 / 1 / 4 (both-tracked 0 / 1 / 4): rosario_04
    ours (ORB-SLAM3 diverged on 5 of 5 draws), v2_03 tie, mh_04, kitti_00, madmax_F3 and 4s_120 ORB3. **The frame-level reading
    (now C4f) says 2 / 1 / 3: it calls mh_04 "ours", the board's own rule calls it "ORB3".** This is the bias the board-honesty
    rule forbids in a headline, measured.
  - The other facts the final board will measure again (unchanged from v1): C1 fails kitti_00 (latency drift 1.22 > 1.2) and
    4s_120 (one dropped frame, so the literal max is +inf); B6 growth exceeds 100 KB per KF on 4 of 6 rows; madmax_F3 has a
    K13-f defect (segment 2 revisits segment 1, never merged); R19-B flags 4s_120 (OKVIS2-X ratio across the 1.05 band).
- **Dry test 2 (`--n-claim 3`, data/dryrun2_board.json): a LOGIC test, not a measurement.** Synthetic draw maps on the same
  design rows plus the FAIL fixtures of fixtures/dry2/ (v3; v2 read them from the scratchpad) carrying the crashed run's
  harness.json. Results, all as registered:
  - 3 ok draws -> counted, medians and C4 ranges over 3 draws (mh_04, 4s_120);
  - a missing draw 3 (v2_03) and a draw of another md5 (madmax_F3; also "loop worker not started") -> not counted, every claim
    of the table fails, and C4 lists them as ORB3* forfeits, C6 / C4f as theirs*;
  - 2 FAIL fixtures of the final md5 (kitti_00) -> SYSTEM FAILURE: not counted although 3 valid draws exist, a C4 forfeit;
  - 1 FAIL fixture then 3 ok draws (rosario_04) -> counted, and C3a fails on the listed crash;
  - VOID runs (OAK) -> not counted;
  - C4 tally 1 / 0 / 5 with the 3 forfeits: `pass` (stage 1) and `pass_stage2` False; C6 2 wins / 4 losses: False;
  - **C10 FAIL although the lock-step identity holds on 6 of 6 rows** (3 rows uncounted; v2 printed C10 PASS here: the
    verifier's finding, correction 8.13). Every claim of S_design is FAIL.
- **Dry test 3 (v3, dryrun3_logic.py, data/dryrun3_logic.txt): a LOGIC test on SYNTHETIC rows** (no run dir, no data; the
  attempt dirs and ledger of T4 in a fresh temp dir). 24 of 24 checks PASS:
  - T0 control: 6 counted winning rows -> every claim PASS (C7b not applicable), so the test is not trivially failing;
  - T1 the verifier's case (5 counted rows + 1 SYSTEM FAILURE row) -> every claim FAILS: C1, C2, C3a, C3b, C4 `pass` /
    `pass_stage1` / `pass_stage2`, C5, C6, C7a, C7b, C7c, C8, C10; the row is ALSO an ORB3* / theirs* forfeit in the tallies;
  - T2 `--only` subset (5 of 7 rows) -> 2 rows "not evaluated", every claim FAILS, both forfeited in C4 and C6;
  - T3 a counted row whose export failed on our side -> ORB3* (not 'no verdict'); export_board_draw raises on a draw without a
    published LIVE-MAP pose;
  - T4 a parked suspect run is listed SUSPECT beside the kept re-run; a kept run the ledger marks suspect is listed SUSPECT and
    refused by run_ok;
  - T5 classify_harness = state_of: valid; worker never started -> CONFIG; env mismatch -> CONFIG; a crash -> FAIL first;
    lock-step mode skips the paced configuration test;
  - T6 a fallback-sentinel FAIL fails C3a on its own row (v1_02_medium).
- **Readings declared after the v1 dry test:** R-B1g (2.4), as in v1. No rule was set or changed from a dry-test number in v2
  or v3: the corrections answer the verifiers' defects, and the dry tests were run after the rules were written.

## 7. Open issues (stated, not resolved here)
Items 1 and 2 are CLOSED by the owner ruling of 2026-10-27 13:00 IST (the line at the end of 4.3: n = 3 for every claim; neither
n = 5 nor the n = 1 plan is taken). Their text is kept for the record (v4 marks them, correction 8.30).
1. **CLOSED (owner ruling n = 3). Was: OWNER DECISION, before the block starts: n = 5 for C4.** The staged untuned target agreed "FINAL at n = 5 on our side"
   (xtenslam FINAL bank, 2026-09-30); the xclean directive sets "n = 3 only for claims". Registered: n = 3 for every claim, C4
   included (chance verdict 1/56 each way at 3 vs 5, against 1/252 at 5 vs 5). If the owner wants n = 5: rounds 4-5 on every row,
   about +20.2 h, every claim at n = 5; the ruling line is appended to 4.3 before the block starts.
2. **CLOSED (owner ruling n = 3). Was: OWNER DECISION, before the block starts: the cheaper v1 plan** (n = 1, R19 / R19-B top-ups; about 8-9 h + 2-3.5 h instead of
   about 31.5-36 h). It contradicts LIVE_BARS B5 NOW ("n = 3 for any claim") and the directive, so it runs only under an owner
   ruling line in 4.3; the block and the evaluator refuse it otherwise. Under it, C4 would run at the 1-vs-5 operating point
   (chance verdict 1/6 each way) and that would be printed with every C4 line.
3. **LIVE_BARS TARGET wants ORB-SLAM3's LIVE stream** (our own patched driver, same FEEDER, n = 5). It does not exist, so C4
   compares our live stream with ORB-SLAM3's offline final keyframes, a comparison biased against us. The board rule (KF level,
   common window) is applied as the board defines it; the bias is disclosed, not corrected. Building that driver is an owner
   decision. It needs GPU / CPU time and an ORB-SLAM3 code patch (GPLv3: it stays a separate local tool).
4. **B4 on Rosario v1 and MadMax** stays "pending" until the owner rules on the reading. C3b therefore covers only the other rows.
5. **The B1 max clause read literally** fails any row with a single engine drop, while B2 allows up to 1 % drops. The literal
   reading is the claim; an owner ruling could change it only BEFORE the board runs.
6. **Tier 2 for GT rows:** analyse_run.py checks only returns to the start. No general depth-at-poses revisit tool exists for
   xclean live runs.
7. **K14 config vs final config:** the reuse of K14 runs (draw 1 of phase C) and the "held-out for this config" label of the K14
   rows both depend on the md5 match. If a crash / hang fix follows K14, phase C runs in full in round 1 (+2.5 h). If K14 stops at
   stage 1 and the system is frozen anyway, the 7 rows run as A2 (4.1, 5.2).
8. **The dry tests show the K13 config failing C1 (2 rows), B6 growth (4 rows) and the board-rule C4 (1 / 1 / 4 on design rows).**
   If the final config keeps these, the board will report C1 / C8 as FAIL and C4 as lost. The fix round, not this
   pre-registration, decides whether they change.
9. **madmax_B0 (52 min) and the 28 E: rows** stress the USB feed. The fallback (5.3) is a byte-identical copy to D: (D: free
   92 GB at the census).
10. **P1_3_PREREG.md gained Amendment F2 while v1 was written.** The prereg requires the F2 configuration of record to be the
    frozen snapshot's default, because board_block.sh passes only XCLEAN_ENGINE=p1 and refuses any inherited engine flag.
11. **Two disclosed exposures** to other systems' numbers on unseen rows (header of this file): x1's U2 on v2_01_easy, and the
    frozen paper-1 v3 board record of the sealed row 4s_20201222_115. No rule depends on them; the values are quoted nowhere.
12. **The P1_3_PREREG pointer line** (appended once, 10:45) names this file; v2 and v3 keep the same path, so it still points
    here. The task allows one pointer line, so v2 and v3 add none.

## 8. Corrections log (v2, 2026-10-27 11:00-11:40 IST; before any never-run row is run, before K14, before the final snapshot)
Each entry: the verifier's finding, its root, the fix, where it is tested.
- **8.1 (major) FAIL handling did not implement 5.3 / 4.4.** Root: v1 state_of tested the md5 BEFORE the crash, and a crashed
  engine never writes engine_summary.json (engine_md5 None), so a crash was classed "MD5 None" and retried as a VOID (6 + 3
  attempts); the FAIL counter lived in one call of paced(), so the retry pass and every relaunch reset it. Fix: the crash test
  comes first (block and evaluator, same rule); FAILs are counted per ROW over all draws and every launch from the ledger; the
  second FAIL makes the row a SYSTEM FAILURE with no further attempt anywhere; the evaluator marks it lost (not counted, forfeit
  everywhere). Tested: block dry test 1 (p12d_w3_e1crash_madmax_E1 = FAIL; crash-VOID-crash = system failure; no retry; FAILs
  survive a relaunch), evaluator dry test 2 (kitti_00).
- **8.2 (major) claim PASS rules let missing / void / crashed rows vanish from C4, C5, C6, C7b.** Root: those tallies were built
  from counted rows only, and C6's registered rule (wins >= losses) was never computed. Fix: every claim requires every row of the
  table counted; in every head-to-head an uncounted row is a forfeit to the other system (C4 ORB3*, C6 / C4f theirs*); C4 stage 2
  needs zero ORB3 rows including forfeits; C6 PASS = wins >= losses computed; C7a needs a within-run ratio on every GT row; C7b
  needs every GT row counted with its oracle (else not applicable only when no row is eligible). Tested: evaluator dry test 2.
- **8.3 (major) the claim n contradicted three binding rules unacknowledged.** Root: v1 set n = 1 with R19 / R19-B top-ups,
  against LIVE_BARS B5 NOW ("n = 3 for any claim"), the directive ("n = 3 only for claims") and the untuned target ("FINAL at n = 5
  on our side"), and ran C4's range rule at 1 vs 5. Fix: n = 3 registered for every claim, in rounds (draw 1 of the whole board
  first); the C4 operating point stated; n = 5 for C4 and the v1 n = 1 plan listed as OWNER decisions (open issues 1-2), each
  enforced by an owner-ruling line that the block and the evaluator check. Tested: NDRAW = 1 refused; dry test 2 at n = 3.
- **8.4 (major) the C4 headline was not the board's own rule.** Root: v1 scored both sides with the frame-level live scorer and
  ORB-SLAM3's per-frame shutdown file, which the board-honesty rule allows only as a secondary column; the board reading was not
  computed. Fix: C4 = scorer_v3's rule (KF level, common window, honest hygiene, range rule, forfeits) on our LIVE-MAP keyframe
  poses vs ORB-SLAM3's untuned keyframe files, through score_arm.py unchanged (board_rule_runner.py); the frame-level reading is
  C4f, secondary. Tested: the runner reproduces score_arm's own x1 dev8 output exactly; dry test 1 shows the frame level calling
  mh_04 "ours" where the board rule calls it "ORB3".
- **8.5 (major) the prereg quoted a never-run row's baseline numbers.** Root: the exposure disclosure carried the values. Fix: the
  values are removed; the disclosure stays without them (header; the second exposure is disclosed the same way).
- **8.6 (minor) P1.2 lock-step runs were never "valid" for the block.** Root: one md5 test (SNAP / SNAP_LS) for both lock-step
  arms. Fix: state_of takes the expected md5 per arm (boardls_ = SNAP_LS's, boardlsref_ = 1cdc6b2c). Tested: p12_k12_mh_04 valid.
- **8.7 (minor) not every attempt was listed per row.** Root: v1 parked top-ups, `_prev`, `_incomplete` and sentinel attempts under
  names the evaluator's `<row>_a*` glob missed. Fix: every park is `board_void/<run name>__<suffix>`; every attempt has a
  board_attempt.json and two ledger lines; the evaluator lists the kept dir, every parked dir and every ledger-only end per draw
  name, and the sentinels apart. Tested: block dry test 1 (11 parked dirs, all with records), evaluator dry test 2.
- **8.8 (minor) a board_mh_04_difficult run could overwrite the sentinel's witness reference.** Root: later sentinels used the
  registry entry tag|md5|mask, which any `--reference` run of that tag rewrites. Fix: the first valid sentinel's dir is recorded
  in board_sentinel_ref_<md5>.txt and every later sentinel passes `--witness-ref <that dir>`. Tested: block dry test 1.
- **8.9 (minor) the run of record did not check that the loop worker started.** Root: `p13_loop_on` comes from loop_on(), True
  even when the worker never started. Fix: run_ok requires checks.p13.worker present (slam, exit 0, not terminated) and
  inactive_why None; also XCLEAN_ENGINE = p1, env flags equal to the reference sentinel's, and the block refuses inherited engine
  flags. Tested: dry test 1 (6 K13 runs ok; the OAK P1.2 run not).
- **8.10 (minor) census count label.** Root: "top-level dirs" was len(harness dirs) + len(unmatched). Fix: census.py counts every
  non-skipped top-level dir; re-run 11:28: 1383 top-level dirs, 1177 with a harness.json, 167 row-named without one, 40 without a
  row name; 0 conflicts; rows.json unchanged (sha1 81b41675).
- **8.11 (minor) hang, phase F, C10 md5, n = 3 bars.** Roots and fixes: no hang watchdog -> 1.5 x stream + 600 s, the block kills
  its own tree, FAIL (tested); phase F ran no top-up silently on a missing / foreign board.json -> it aborts (exit 5); C10 did not
  check the lock-step md5s -> it requires boardls_ = the lock-step md5 and boardlsref_ = 1cdc6b2c; at n = 3 the bars used run 0
  -> the median reading per clause (every draw for event claims), 4.2.
- **8.12 (minor) K14 S1 contingency.** Root: if K14 stopped at stage 1, the 7 rows would be first run in phase C after 24 seen rows
  and mislabelled. Fix: a K14 row without any K14 artefact runs as phase A2 right after A in every round, and its label is decided
  mechanically (4.1). Tested: block dry test 2 (kitti_09 as A2).

### 8 (v3). Corrections of the second verification (2026-10-27 12:30-13:20 IST; before any never-run row is run, before K14, before the final snapshot; v2 kept in v2_20261027_1140/)
Each entry: the verifier's finding, its root, the fix, where it is tested. No number of any never-run, K14 or sealed row was read
for any of them; the dry tests ran on already-run dirs and synthetic inputs only, after the rules were written.
- **8.13 (major) claim PASS rules of C4 (stage 1 / stage 2), C6 and C10 lacked the every-row-counted test, and C4 had no single
  PASS field.** Root: v2 made an uncounted row a forfeit in the head-to-head tallies and treated the forfeit as the whole
  sanction there (one loss among n rows), and C10 iterated over the lock-step identities only; 4.2 / 4.4 / 8.2 say an uncounted
  row FAILS every claim of its table AND is a forfeit. The registered text is kept (the alternative, amending 4.2 to forfeits
  only, was rejected: it would let a crash or a missing unseen row cost a single tally loss). Fix: every-row-counted is ANDed
  into C4 (`pass_stage1`, `pass_stage2`), C6 and C10; every claim has ONE `pass` field; **C4's PASS = `pass` = `pass_stage1`**,
  stage 2 = `pass_stage2`; board_tables.md prints a PASS / FAIL table per claim and table. Tested: dry test 3 T0 / T1 (the
  verifier's synthetic case now fails C4, C6 and C10), dry test 2 (C10 FAIL with the identity holding on 6 of 6 rows).
- **8.14 (major) the sentinel-failure path of 5.2 was not implemented.** Root: the "since the last good sentinel" list lived in
  one launch's memory and was only written as text; nothing parked or re-ran the suspect draws (a relaunch found them valid and
  skipped them), and board_eval.py never read the suspect list, so suspect draws became runs of record. Fix: the window is read
  from the LEDGER (every paced row-draw end line of the final md5 after the last valid sentinel end line); on a sentinel failure
  every run of record in it gets a `suspect` ledger line, board_attempt.json class SUSPECT (`class_at_end` kept) and is parked
  as `board_void/<name>__suspect` (board_suspect.txt is the human list); the relaunch re-runs those draws because they have no
  run dir (ONLY not needed); a suspect left in place by a block that died before its park is parked at the next launch; the
  evaluator lists SUSPECT attempts per row, refuses a kept run the ledger marks suspect, and writes the suspect list and the
  sentinel instances into board.json. Tested: block dry test 1 S5, dry test 3 T4.
- **8.15 (major) a second sentinel FAIL locked the board out on that md5.** Root: sentinel FAILs were counted over all launches
  per md5, so after the 2nd every relaunch exited 3 with no attempt, and only an unregistered ledger edit could continue; 5.3's
  "keep the frozen md5 and report" was then impossible. Fix: each sentinel instance has its own budget (3 VOID, 2 FAIL); every
  relaunch gets a fresh instance; a sentinel row whose last two instances each stopped with FAIL x2 is replaced by the
  registered fallback sentinel row v1_02_medium (design, EuRoC, C:, 85 s; own witness reference); both failed -> exit 6 and 5.3
  decides, with no manual step in the procedure. Every sentinel FAIL still fails C3a for its own row. Tested: block dry test 1
  S6, S5 (a fresh instance after a STOP_FAIL).
- **8.16 (minor) the sentinel cadence counted draws skipped as already valid, and the retry pass had no cadence.** Root: the
  counter was incremented for every row-draw walked and never in the retry pass. Fix: the cadence counts the row-draws that got
  an attempt since the last good sentinel (ledger), checked after every row-draw in the rounds, the retry pass and phase F; the
  closing sentinel runs only when a row-draw ran since the last good one. Tested: block dry test 1 S4 (a relaunch over a
  complete board made 1 harness call, v2 made 6), S7 (sentinels inside the retry pass), block dry test 2.
- **8.17 (minor) the block's "valid" was weaker than the evaluator's run_ok.** Root: state_of checked the harness validity, the
  md5 and the mode only, while the harness keeps processes_ok True when the loop worker never started; such a run was the
  block's run of record and was never retried, then failed run_ok, so the row was uncounted without the reason. Fix: state_of
  applies run_ok's configuration test (oracle flags, XCLEAN_ENGINE = p1, the worker started, env flags = the session
  reference): class CONFIG aborts the block (exit 4), since a configuration fault is the same on every row and the opening
  sentinel (a design row) meets it first; K14 reuse uses the same test in the block and the evaluator; the evaluator's
  classify_harness mirrors it. Tested: block dry test 1 S1 (now5_kitti_00 -> CONFIG), S2 / S3 (worker not started, env
  mismatch -> exit 4), dry test 3 T5.
- **8.18 (minor) `--only` produced subset claims with no marker.** Root: claims() kept only the evaluated rows of each table and
  the subset was not recorded; the default output overwrote data/board.json. Fix: claims are always taken over ALL rows of each
  table (rows left out are "not evaluated" placeholders: uncounted, so every claim fails and the head-to-heads forfeit them);
  `only` / `subset` are recorded; `--only` writes data/board_subset.json by default and refuses data/board.json; phase F refuses
  a subset board.json; dry tables come from the rows file, not from the evaluated rows. Tested: dry test 3 T2, the refusal and a
  dry subset run (6.2).
- **8.19 (minor) a counted row whose board-rule export failed became 'no verdict', not a loss.** Root: run_board_rule scores
  only rows exported without error and board_tally treated an unscored counted row as 'no verdict'. Fix: export_board_draw
  raises a named error when a draw has no published LIVE-MAP pose (at the KF stamps); board_tally makes a counted row with an
  export error an ORB3* forfeit (ours lost) with the reason. Tested: dry test 3 T3.
- **8.20 (minor) phase E had no hang watchdog and no FAIL accounting.** Root: run_ls ran the harness in the foreground with no
  limit and re-ran anything not valid at every relaunch, unrecorded. Fix: ls_attempt: watchdog 3 x stream + 1200 s that kills
  the block's own process tree (FAIL), ledger start / end lines with the arm (ls / lsref), board_attempt.json, parks, at most 2
  attempts per lock-step run; the evaluator lists every lock-step attempt; a FAIL of the final lock-step snapshot fails C3a
  ("<row> (lock-step)") and C10 lists the row. Tested: block dry test 1 S8.
- **8.21 (minor) an empty $SDUR made the sentinel's VOID budget non-numeric.** Root: SDUR was read without a check and passed
  unquoted. Fix: SDUR, the fallback's duration and EVERY are checked at integrity (exit 2 if not numeric) and every duration is
  passed quoted. Tested: block dry test 1 S0.
- **8.22 (minor) dry test 2's fixtures lived in the session scratchpad.** Root: dryrun2_rows.json named scratchpad dirs. Fix:
  the 3 fixture dirs are copied byte for byte to runs/xclean/board/fixtures/dry2/ and dryrun2_rows.json points there (sha1
  e37e960a -> 2198b118); the block dry-test scripts, the fake harness and their outputs are kept in runs/xclean/board/blocktest/
  for the same reason. Tested: dry test 2 re-run from the new paths (kitti_00 SYSTEM FAILURE, rosario_04 crash listed, as in v2).

### 8 (v4). Corrections of the third verification (2026-10-27 13:50-14:50 IST; before any never-run row is run, before K14, before the final snapshot; v3 kept in v3_20261027_1320/)
Each entry: the verifier's finding, its root, the fix, where it is tested. No number of any never-run, K14 or sealed row was read
for any of them; the dry tests ran on already-run dirs, synthetic inputs and a fake harness only, on the CPU, from copies in the
session scratchpad, after the rules were written. The owner's ruling line in 4.3 is unchanged.
- **8.23 (major) a GPU front dropped for want of a visible GPU passed the CONFIG test of the block and of the evaluator.** Root:
  the registered configuration includes the GPU front (fast path `gf` of front/fastpaths.py), and the snapshot drops it
  SILENTLY whenever no GPU is visible (CUDA_VISIBLE_DEVICES -1 or empty in the launching shell, or a cupy / driver failure at
  start); state_of / config_why / run_ok compared only env flags (XCLEAN_ / XTEN_ / XONE_ keys), the worker and the oracle
  flags, none of which changes; and when the condition is present at the first launch the reference sentinel has it too, so
  the +-20 % witness test cannot see it either. Every paced run, the clean first draw of each unseen row included, would have
  run an unregistered frame path and been kept by both scripts. Fix: (a) the paced phases (and X) refuse any
  CUDA_VISIBLE_DEVICES in the launching shell (exit 2); (b) the run-of-record configuration test (block state_of, evaluator
  config_why / classify_harness / run_ok, the K14 reuse in both) now also requires engine_summary.json `p1.lat` = the
  snapshot's registered set, which is the `LAT_DEFAULT` line of the snapshot's OWN front/fastpaths.py (the default with a
  visible GPU; read statically, exactly one such line or the block / evaluator refuses), and `gpu_front` set when the set holds
  `gf`. This is an absolute test, not a comparison with the reference sentinel, so it holds at the first launch too. The block
  logs the set on its start line; the evaluator takes `--snap-dir <SNAP>` (its `_engine_md5` must equal the first
  `--snap-md5`) and records `snap_dir` and `lat_registered` in board.json. A paced run without engine_summary.json (frame path
  unknown) is never valid. Tested: integrity (CUDA_VISIBLE_DEVICES -1 / empty / 0 refused; p13b_fix1a accepted with its set
  e1, e2, e3, g1, gf, m1, mf); a scratch copy of p13b_fix1a's fastpaths.py drops gf under -1 and empty (the premise); block dry
  test 1 S1 (a copy of k13_mh_04_difficult's harness.json + its engine_summary.json without gf -> CONFIG, the original ->
  valid) and S2 rowK (exit 4); dry test 3 T5 / T7; dry tests 1 and 2 re-run with --snap-dir: every K13 run of record stays ok
  (their p1.lat is the registered set, gpu_front set), all numbers and claims identical to v3.
- **8.24 (minor) power-cut gaps in the suspect window (the verifier's N1, N2).** Roots: N1 -- a run whose harness.json was
  written but whose ledger end line was not got its end line only when paced() reached it, AFTER the relaunch's opening
  sentinel, so it was outside that sentinel's window and was kept although taken before a failed sentinel; N2 -- a power cut
  inside mark_suspects left the rest of the window as runs of record, because reconcile_suspects re-parked only runs that
  already had a suspect line. Fix: at every launch, BEFORE the opening sentinel, resolve_open resolves every paced attempt of
  SNAP_MD5 left without an end line (or with an interrupted park, or an unreadable record), so a run found then is inside the
  opening sentinel's window; reconcile_suspects RE-DERIVES the suspects from the whole ledger at every launch and after every
  sentinel failure: a failed instance = its STOP line, or its 3rd VOID / 2nd FAIL / 3rd INCOMPLETE end line when the block
  died before that line (the missing STOP line is written), and its window = every valid row-draw end line between the last good
  sentinel before it and it; missing suspect lines are written and every suspect still in place is parked (mark_suspects is now
  this one code path). An attempt whose harness never created its dir and whose end line was lost gets an INCOMPLETE end line
  at the relaunch. Tested: block dry test 1 S9 N1 (f1 AND f2 suspect, both re-run), N2 (f2, f3 get their suspect lines and are
  parked, f1-f3 re-run), N2b (STOP line written at reconcile, the window suspect), N6; S4 / S5 unchanged (a relaunch over a
  complete board still makes ONE harness call).
- **8.25 (minor) board_attempt.json was written in place and trusted when read.** Roots: (a) park_suspect rewrote it in place,
  and a 0-byte / zero-filled file after a power cut (NTFS does not journal file data) made its json.load raise, hidden by
  `|| return 0`, so the suspect was never parked and the evaluator, finding attempt None, did not see it as suspect; (b)
  resolve_leftover's NOHARNESS branch rewrote a FAIL whose park was interrupted as INCOMPLETE while the ledger kept FAIL (block
  1 FAIL, evaluator 0, C3a missed the crash); (c) a parked FAIL dir with an unreadable record was listed twice by attempts_of
  (the dir with attempt None and its ledger end line), so one FAIL counted 2 and made a false SYSTEM FAILURE. Fix: every record
  is written to a temp file and renamed (attempt_json, park_suspect); `[ -f ]` became rec_ok (non-empty, parses, of that
  attempt); **the ledger's end line is the class of record**: a relaunch never re-classifies an attempt that has one (valid ->
  kept, else parked `a<k>`), and a lost / unreadable record is rewritten from it; park_suspect identifies a dir with an
  unreadable record by the ledger (the dir in place is the name's latest attempt) and rebuilds it; park lines carry the attempt
  number. The evaluator reads a record only when it parses, takes an attempt's class from its ledger end line (else the record,
  else the harness), takes the attempt number from the record, else the ledger (kept dir = latest start; parked `a<k>`; the park
  line), else the one unclaimed end line of that name, and never lists a dir and its end line twice; evaluate_row's suspect
  lookup uses the same attempt number. Tested: block dry test 1 S9 N3 (0-byte record + suspect line -> rebuilt, parked SUSPECT,
  re-run), N3b (zero-filled record of a kept run -> rewritten, kept), N4 (FAIL record kept, parked a1, block 1 FAIL), N4b (a
  HANG FAIL whose dir holds a valid harness.json and whose park was interrupted is parked, never kept); dry test 3 T8 (N3: listed
  once, SUSPECT; N4 as v3 left it: the ledger's FAIL counts, 1 = the block's 1; the verifier's double count: 1 FAIL, not 2).
- **8.26 (minor) C3a missed sentinel FAILs that left no run dir.** Root: board_eval main() built the sentinel list from bsent*
  dirs only, and an instance whose attempts died before the harness created the dir exists only in the ledger. Fix:
  sentinel_names() takes the dirs (kept, parked) AND every ledger name with row `sentinel`; attempts_of lists their ledger-only
  end lines. Tested: dry test 3 T9 (2 ledger-only FAILs listed, C3a fails on mh_04_difficult (sentinel)).
- **8.27 (minor) registered block parameters could be overridden from the environment.** Root: HANG_ADD_S, HANG_LS_ADD_S,
  MAXTRY, RETRY, PAUSE_S, NPAR, BOARD_B, LEDGER and VOIDD were read as `${VAR:-default}` and never checked; a dry-test value
  left in a shell (HANG_ADD_S = 8, MAXTRY = 2, RETRY = 1, PAUSE_S = 0) would give kitti_04 (28-s stream) a 50-s hang limit
  (1.5 x 28 + 8) against a harness start-up and join overhead the K13 runs show (wall_s 111.7 s on mh_04's ~102-s stream, 120.7
  s on kitti_01's ~114-s stream; the margin is thin under load), i.e. a risk of HANG FAIL x2 and a SYSTEM FAILURE of an unseen
  row, and would cut every VOID budget; an inherited LEDGER / VOIDD would hide the attempts from the evaluator. Fix: integrity refuses any non-registered value (exit 2), PHASES=X included,
  unless BOARD_SELFTEST is set; BOARD_SELFTEST itself is refused on the run dir and with any snapshot that is not the FAKE
  harness, so a stray one cannot run an engine; the start line logs every registered parameter and the frame-path set. Tested:
  integrity (each of HANG_ADD_S = 8, HANG_LS_ADD_S = 8, MAXTRY / RETRY / PAUSE_S, NPAR = 12, LEDGER, VOIDD, BOARD_B refused; the
  registered values given explicitly accepted; BOARD_SELFTEST with the real snapshot, on the run dir, or with a ledger on the
  run dir refused).
- **8.28 (minor) stale PACED_ACTIVE.** Root: staleness was judged by the recorded winpid alone (`tasklist PID eq N`, any image),
  so a PID reused after a reboot or an empty winpid made the relaunch wait forever without a log line, and a block whose bash
  died while its harness still ran was judged stale (the relaunch would have run beside it). Fix: the lock records the winpid
  AND its process start time; lock_state calls it stale only when no harness of this board is running (a python whose command
  line runs xclean.live.harness with `--out <run dir>/b...`) AND (the lock predates the last boot, or the process is gone, or the
  PID belongs to a process with another start time); a lock with no usable PID and no reboot since is waited for; take_lock logs
  the lock and the verdict every 10 min while waiting. Tested: block dry test 1 S10 (dead PID -> stale; a live PID with another
  start time -> stale; the same PID with its start time -> live; empty PID -> live, and stale once the lock predates the boot; a
  dead PID while a (fake) harness of the board runs -> live, then stale once it ended; take_lock removes a stale lock and writes
  winpid + start; a foreign lock is waited for with a log line).
- **8.29 (minor) a transient import failure of the loop worker stopped the unattended board.** Root: "loop worker never
  started" is also what a transient ImportError of xclean.slam.worker in the harness process gives (a locked numba cache, a DLL
  or memory error); it was CONFIG, exit 4, although "a configuration fault is the same on every row" does not hold for it. Fix:
  a CONFIG whose ONLY reason is the loop worker (state_of / classify_harness mark it `[worker-only]`) gets ONE more attempt on
  that draw; a second on the draw, or any other CONFIG, aborts (exit 4). The attempt is parked `a<k>`, listed, never a run of
  record; no validity rule changes. Tested: block dry test 1 S2 rowW (once, then valid: RC 0) and rowI (twice: exit 4); dry test
  3 T7 (the mark only when the worker is the sole reason).
- **8.30 (minor) text vs code.** (a) 2.1 "every attempt also gets a board_attempt.json": an attempt whose harness never created
  its dir has none -> 2.1 now says so. (b) 6.1 "refuses an n other than 3 without the owner-ruling line": --dry bypasses it
  (dry test 1 runs at n = 1; --dry refuses unseen, K14 and sealed rows) -> 6.1 qualifies it. (c) 2.1 / 5.3 "MD5 aborts the
  block": in phase E an MD5 attempt is parked and counts as one of the lock-step run's 2 attempts -> 2.1, 2.10 and 5.3 say so
  (the parallel phase is not aborted; the row's C10 then fails, against us). (d) 4.3's body, 5.7 and section 7 items 1-2 still
  presented n = 5 and n = 1 as open owner options -> marked CLOSED by the 13:00 ruling (the ruling line itself is unchanged). (e)
  2.1 "INCOMPLETE is not an attempt" while N4 rewrote a FAIL record as INCOMPLETE -> fixed in the code (8.25).

### 8 (v4.1). Corrections of the fourth audit (2026-10-27 15:26-16:30 IST; before any never-run row is run, before K14, before the final snapshot; v4 kept in v4_20261027_1526/)
The v4 audit (verify, fix, confirm) passed v4 with ONE minor left (B1 below); the lead added three K14-interface items after
k14_block.sh v2 changed how K14 keeps its attempts (B2, B3, and the glob check of 8.34). Each entry: the finding, its root, the
fix, where it is tested. No number of any never-run, K14 or sealed row was read; the tests ran on the CPU, from copies in the
session scratchpad, on a fake harness, synthetic inputs and already-run design-row dirs (read only), after the rules were
written. The owner's ruling line in 4.3 is unchanged.
- **8.31 (minor, B1) a sentinel instance found VALID at a relaunch got only its end line.** Root: resolve_open resolves an
  attempt left without its end line through resolve_leftover, which writes "end ... valid found at relaunch" and keeps the
  dir, but the two things check_sentinel writes after a valid instance -- the witness reference file (for the `--reference`
  instance) and the `sentinel ... valid` ledger line -- were written nowhere; the same gap when the block died between paced()'s
  valid end line and check_sentinel's lines (resolve_open does not even select that instance: its end line is valid and its
  record readable). So a first launch cut during its opening (`--reference`) sentinel left no reference file, the relaunch ran a
  second `--reference` instance, and THAT became the witness reference, against 5.2 "the FIRST valid instance is the witness
  reference"; and board.json's sentinel_instances (the ledger's sentinel lines) missed the instance. Fix: board_block.sh
  resolve_sentinels, run at every launch right after resolve_open and before reconcile_suspects and the opening sentinel, walks
  the ledger in order and, for every sentinel instance of SNAP_MD5 whose attempt ended valid, writes the reference file when
  that instance was the `--reference` one (its start line's witness args) and its row has no valid reference file yet, then
  its `sentinel ... valid` line when missing (detail: the start line's witness args, marked "written at relaunch").
  board_eval.py sentinel_instances() lists the ledger's sentinel lines plus, for an instance with a valid end line of the final
  md5 and no sentinel line (a block that died and was never relaunched), a rebuilt valid entry marked "end line". Tested:
  blocktest/selftest41.sh W1 (the confirmer's V2 with the 3rd attempt VALID: v4 no sentinel line, v4.1 the line, no suspect
  either way), W2 (opening `--reference` instance cut before its end line: v4 a second `--reference` instance became the
  reference, v4.1 the cut instance is the reference and the relaunch's opening sentinel runs `--witness-ref` it), W3 / W3b (end
  line written, reference file and sentinel line not; primary and fallback row); dry test 3 T10.
- **8.32 (minor, B2) the K14 attempts of a reused draw 1 were invisible to the board.** Root: 2.1 and board_eval.py stated
  that k14_block.sh deletes its VOID attempts, so none were listed; k14_block.sh v2 (fixer 2026-10-27 ~14:00) instead MOVES
  every attempt that is not a run of record to k14_void/<name>__<suffix> and records every attempt in k14_ledger.tsv, with its
  FAILs (crash, hang). The board listed none of them and counted none of the FAILs, so a final-md5 crash or hang on a seen row's
  draw 1 would never reach the row's system-failure count or C3a. Fix: board_eval.py read_k14_ledger (the K14 format: time,
  event, name, row, attempt, md5, class, detail; absent = None) and k14_attempts (one record per attempt = its first end line,
  a `found` line or an `end ... pre prev_<class>` leftover; a start without an end listed `open`; the k14_void dirs beside);
  for a row whose draw 1 is the reused k14_<row>: draw1_k14_attempts (the draw's hygiene record) and its FAILs of the final md5
  added to fail_attempts, so they count exactly as board FAIL attempts (system failure at 2 over the row's draws, C3a); every
  other K14 candidate arm of the row (k14_ when not reused, k14ls_, k14n2_, k14n3_): k14_other_attempts, its final-md5 FAILs
  fail C3a as "<row> (K14 run)" (the sentinel / lock-step precedent, 8.26 / 8.20), no system failure (not a board draw); a reused
  draw 1 with the ledger absent or without a valid line of that run: k14_hygiene_unverifiable, listed under C3a and C3a cannot
  pass. board_block.sh fails_of adds the same draw-1 FAILs (k14_draw1_fails: counted only when k14_<row> passes state_of,
  i.e. is the reused draw 1), so the block stops attempting a row the evaluator calls a system failure. New option
  `--k14-ledger` (default <B>/k14_ledger.tsv); board.json records the ledger used. The 2.1 / 5.3 text says so. Tested: dry test
  3 T12 (evaluate_row with measure / bars_run stubbed: listed FAIL / valid / other-md5 FAIL / parked dir; one final-md5 FAIL
  counted once, another md5's not; + one board FAIL = system failure; C3a fails; ledger absent and ledger without the run's line
  flagged, C3a fails; a k14ls_ FAIL -> "(K14 run)"; a K14 run that fails the run-of-record test is not reused and its FAIL is an
  "other" K14 FAIL); blocktest/selftest41.sh W5 (fails_of 0 -> 1 with the K14 FAIL, 2 with a board FAIL: the next draw is not
  attempted; a duplicate end line and another md5's FAIL not counted; a K14 run that is not reused not counted).
- **8.33 (minor, B3) "spent" was judged from four run-dir names.** Root: board_eval k14_label and board_block k14_spent looked
  for k14_ / k14ref_ / k14ls_ / k14lsref_ dirs in place only; a K14 run that k14_block.sh parked into k14_void (every VOID, FAIL,
  MD5 or CONFIG attempt; an earlier candidate's run), an attempt whose harness never created its dir (ledger only) or a top-up
  arm (k14n2_ / k14n3_ / k14refn2_ / k14refn3_) made the row read "unspent / never run before the board", and the block would
  run it as phase A2 (first-run, held-out). Fix: both scripts call a row spent on ANY K14 trace by any md5 -- a run dir of any of
  the 8 K14 arms, a k14_void entry `k14<arm>_<row>__...` (anchored: k14_madmax_E00__a1 does not spend madmax_E0), a k14_ledger.tsv
  line whose row is the row or whose name is `k14<arm>_<row>`; board_eval reports the candidate-arm and the P1.2-reference-arm
  md5s apart (k14_trace) and labels "same md5" when a final md5 is among the candidate md5s, else "other md5" with the md5s
  (or "P1.2 reference arm only"), and "spent, md5 unknown" for a candidate-arm trace whose md5 nothing records (never shown as
  held-out for this config, never as never-run). The phase-C reuse itself needed no change: it reuses only k14_<row> IN PLACE that passes
  state_of (a parked run is never a run of record), and its K14 attempts are now counted (8.32). Tested: blocktest/selftest41.sh
  W4 (real rows.json, the 7 K14 rows; k14_void-only, ledger-only, k14n2_-only traces: v4 all 7 in A2, v4.1 those 4 in C and the 3
  untouched rows in A2; k14_pin.json / k14_stage1_*.json / stale_locks / k14_madmax_E00__a1 spend nothing); dry test 3 T11.
- **8.34 (minor) text and the glob check.** (a) 2.1's "k14_block.sh deletes its own VOID attempts" and board_eval.py's note
  "k14_block.sh deletes its VOID attempts, so none are listed" were stale (8.32) -> replaced. (b) Checked, no change needed:
  neither script globs `k14_*` in D:/xoneslam_runs/xclean. Every K14 path is built from a row name (`$B/<arm><row>`,
  `B + "/" + p + row`, `"%s/%s_%s" % (B, p, row)`) or read through an anchored match (k14_void entries `^k14[a-z0-9]*_<row>__`,
  ledger names `^k14[a-z0-9]*_<row>$`), and no row is named void, ledger.tsv, pin.json or stage1_<md5>.json, so k14_void,
  k14_ledger.tsv, k14_pin.json, k14_stage1_*.json and stale_locks are never taken for run dirs (dry test 3 T11, selftest41 W4).
  The only directory listings are `bsent*` (sentinel_names), board_void/<name>__ (attempts_of) and k14_void/ (read only).

## 9. Freeze (section 5.8 filled; lead, 2026-10-28 05:20:49 IST by file time of FREEZE_RECORD.md; prereg sha1 before this section 0661c0e0)
- Owner instruction (chat, 2026-10-28 ~05:10 IST), verbatim: "Freeze p13f4a as the system with K14 recorded as FAILED on d and e
  (disclosed on the board and in the paper, with the audit's causes), then run the final board at n=3 as registered."
- The binding freeze record and the disclosure obligations: runs/xclean/board/FREEZE_RECORD.md (sha1 bfe99714). The frozen system
  did NOT pass K14 (d FAIL madmax_E0, e FAIL kitti_02); it is frozen by the owner's decision with that disclosed. No rule, row, bar,
  reading, n or script of this file changes.
- SNAP = engines/p13f4a (6bb6e65ae901c55288c78cc3cceda22f), SNAP_LS = engines/p13f4 (601a765b4de7bd117f3042ad8def3f29).
  `PHASES=X` on this pair: all passed, 2026-10-28 05:19:31. Draw 1 of the 24 seen rows is the k14_<row> run of this md5 where it
  passes state_of (5.2).
- Command lines (5.8):
```
SNAP=D:/xoneslam_runs/xclean/engines/p13f4a SNAP_MD5=6bb6e65ae901c55288c78cc3cceda22f \
  SNAP_LS=D:/xoneslam_runs/xclean/engines/p13f4 bash D:/xoneslam_runs/xclean/board_block.sh                  # rounds 1-3, A A2 B C D
SNAP=... SNAP_MD5=... SNAP_LS=... PHASES=E bash D:/xoneslam_runs/xclean/board_block.sh                      # lock-step
CUDA_VISIBLE_DEVICES=-1 C:/Users/USER/anaconda3/envs/cvcuda133/python.exe -W ignore runs/xclean/board/board_eval.py \
  --snap-md5 6bb6e65ae901c55288c78cc3cceda22f --snap-ls-md5 601a765b4de7bd117f3042ad8def3f29 \
  --snap-dir D:/xoneslam_runs/xclean/engines/p13f4a --k14-ledger D:/xoneslam_runs/xclean/k14_ledger.tsv     # evaluation
```

### 8 (v4.2). Corrections DURING the board (lead, 2026-10-28 12:24-13:44 IST; after launch 1 stopped, BEFORE any run of the new md5; no accuracy number of any board run read; v4.1 files kept in v41_20261028_1230/)
- **Launch 1** (SNAP p13f4a 6bb6e65a, 05:21:33-12:24:00): the opening sentinel valid (`--reference`); round 1 phase A ran 9 row-draws
  valid and madmax_B0 VOID x6 (NO VALID RUN: Windows services svchost Dnscache / InventorySvc / StateRepository, MemCompression,
  MpSigStub, NVDisplay.Container, and steam.exe); the 12:14 sentinel bsent1028121450 VOID x3 -> SENTINEL FAILED (STOP_VOID) -> the 9
  runs of record parked SUSPECT -> exit 3. The block, the ledger and the parked dirs worked as registered.
- **Owner instruction (chat, 2026-10-28 ~12:30 IST), verbatim:** "fix and check everything and re start board run , make sure no
  system process kill \ void runs".
- **8.35 (defect, block; the cause of the stop).** v4.1 judged every LATER sentinel instance with the harness's `--witness-ref`, whose
  witness_v2 needs the stages rectify, extract AND track; the P1 engine's latency.json has no `track` stage (it records extract, kf,
  marg, match, p13_compose, p13_emit, rectify, stereo, window, window_final), so every later instance was VOID "missing witness
  stages" whatever the machine did. Root: the v4 block dry tests used the FAKE harness, which wrote a `track` stage. Fix (block only):
  every instance runs with `--reference` (the harness applies the machine checks: load, feeder, pinning, fence); a later instance's
  witness test is applied by the block's state_of() for dirs bsent*_<row>: rectify and extract, p50 AND p95, within +-20 % of the
  row's FIRST valid instance (sref_of), the same stages and band the protocol v2 (b) names minus the stage the engine never records;
  missing stages or an unreadable latency.json = VOID. The ledger labels a later instance `witness: --witness-block <dir>`. Dry test
  blocktest/drytest835.sh (13/13 PASS on the REAL 05:21 and 12:14 sentinel dirs + synthetic drift): the 12:14 attempt 1 (whose only
  reason was the missing stage) is valid at ratios 0.98-1.02; attempt 3 is VOID (steam.exe; extract p95 1.226).
- **8.36 (validity, owner instruction): Windows OS processes are exempt from the per-process 5 % bar.** New paced twin
  engines/p13f4b = p13f4a + added ALLOW-list lines in xclean/live/harness.py (the registered twin mechanism, 5.1: twin_check vs
  SNAP_LS p13f4 = ok): svchost.exe, memcompression, registry, smss, csrss, wininit, services, lsass, lsaiso, winlogon, fontdrvhost,
  dllhost, sihost, taskhostw, runtimebroker, ctfmon, audiodg, wmiprvse, searchindexer, searchprotocolhost, searchfilterhost,
  aggregatorhost, compattelrunner, dashost, tiworker, trustedinstaller, mousocoreworker, usocoreworker, backgroundtaskhost, sppsvc,
  sgrmbroker, spoolsv, wudfhost, mpsigstub, mpcmdrun, nissrv, mpdefendercoreservice, securityhealthservice, securityhealthsystray,
  smartscreen, widgets, widgetservice, textinputhost (.exe) and the display-driver service nvdisplay.container.exe. Engine code is
  byte-identical (only live/harness.py differs; audit_v42/AUDIT_TWIN.md). Basis: the harness's own 2026-10-23 amendment exempts "named
  desktop/OS processes"; the list lacked the OS service hosts. User applications are NOT exempt (steam.exe, OneDrive.exe stay void
  causes; the owner closes them for the window); neither are our own tools (python, bash, sleep, grep, conhost).
  - Disclosed effect: exempt OS load is still recorded per run (checks.allow_list_cpu_pct_max / allow_list_total_cpu_pct_max) and is
    REPORTED beside B1 (8.38 c). It can only slow our pinned engine / worker / feeder (counted against us in B1 / B2), with one
    exception the audit named: a delayed loop worker applies fewer corrections, which moves the within-run ratio toward 1.0, so C7a
    is read together with C7c and the per-row worker latency / backlog / allow-list total (8.38 e).
  - **Consequence (5.3): the board of record is `_engine_md5` 1e49d634f155882f9177d3cd2bd36dfc (SNAP p13f4b; SNAP_LS p13f4 601a765b).**
    The board restarts for ALL rows and draws. Every trace of 6bb6e65a (the opening sentinel, the 9 SUSPECT draws, madmax_B0's 6
    VOIDs, the 3 sentinel VOIDs, the 4 other VOIDs) is a pre-fix run: never used for a claim, listed in the "pre-fix md5" appendix
    (8.38 a). No K14 run is a draw any more (prereg 2.1 "the SAME _engine_md5"; 7 item 7 "phase C runs in full"): the 24 seen rows
    get three fresh paced draws.
  - **K14 rows' label (4.1, mechanical):** "seen by an earlier config (K14 md5 6bb6e65a)". 6bb6e65a is the ALLOW-list twin of
    1e49d634 with identical engine code, so these 7 rows are held-out for the board's ENGINE CODE; the board report prints this note
    beside the K14 table (text ruling; no evaluator option).
  - **C2 / C7b on the K14 table** are computed on the three fresh draws as registered. Whatever their verdict, the board report,
    website and paper print beside them the K14 facts of the identical engine code (FREEZE_RECORD 3-5): k14_madmax_E0 1 unflagged
    LIVE-MAP jump not in LIVE-ODOM (frame 7268), kitti_02 0.9878 vs its E1 bar 0.9694, kitti_09 no loop fused. K14-d / K14-e stay
    FAILED. (K14 recorded no FAIL attempt, so C3a is unaffected.)
- **8.37 (the pre-launch audit's block items; audit_v42/AUDIT_BLOCK.md, 3 auditors + 14 adversarial verifiers, workflow wf_0395ea7f-277):**
  (a) a private witness registry per attempt: XCLEAN_WITNESS_REF_V2 = logs/witness_reg/<name>.a<k>.json (paced and lock-step); the
  harness loads the shared REPO file unguarded and rewrote it non-atomically at every --reference run, so a power cut there could have
  failed every later run (the key is excluded from every env comparison already); (b) a block stopped while its paced harness
  still runs KEEPS PACED_ACTIVE (a relaunch then waits instead of running beside it), with a TERM / INT / HUP trap so the EXIT trap
  runs on a stop signal; (c) ref_parse_check at every launch, before any classification: an unreadable witness reference stops the
  launch (exit 7, no ledger line); (d) an attempt of ANOTHER md5 left in place is parked `prev_md5_<8>` with its own md5, never kept
  as this md5's run (the verifier's tested form); (e) runs/xclean/livebars/preflight.py imports the LAUNCHING snapshot's ALLOW list
  (sys.path.append instead of insert), so the pre-flight does not wait 90 min on now-exempt OS load.
- **8.38 (evaluator; specified NOW, implemented and dry-tested on design rows after the paced rounds and BEFORE any board result is
  read; then re-audited; the claim rules do not change):** (a) `--prefix-md5 6bb6e65a`: every trace of that md5 (attempts, sentinels,
  suspect lines, the VOID column) goes to a "pre-fix md5" appendix, out of the board of record's attempts / suspect_rows / sentinels;
  a trace of neither md5 stays listed and flagged; (b) the K14 note and facts of 8.36 printed beside the K14 table and its C2 / C7b;
  (c) per run of record: allow_list_cpu_pct_max / total, and a count of runs where a newly exempt process exceeded 5 % of a core;
  (d) the 8.35 witness ratios of every sentinel instance recomputed and printed, and the same test in the evaluator's fallback
  classification of bsent dirs; (e) C7a printed with C7c and the per-row worker target latency / backlog / allow-list total.
- **8.39 (operations).** The lead's log watcher uses bash builtins only (no process per check); no python, monitor or sampler runs
  during the paced rounds; 41 stale tail / grep log monitors of the lead's 10-27 sessions were stopped by exact PID (12:5x); the owner
  closes steam.exe and OneDrive.exe for the window (user applications; their startup entries are the owner's settings).
- Files: board_block.sh v4.2 md5 87817c9c; preflight.py md5 39622a38; engines/p13f4b `_engine_md5` 1e49d634; board_eval.py unchanged
  (d5887521) until 8.38. Block dry tests on v4.2 (fake harness, scratch B): selftest4 / selftest41 / selftest2v4 exit as on v4.1 (0 / 0 / 0) after the fake harness was made to write a latency.json with the REAL engine's stage keys (blocktest/fakesnap; without it every later fake sentinel was, correctly, VOID 'latency.json unreadable'); the only differences are the intended sentinel label (--witness-block), PIDs, times and the order of parallel lock-step runs (blocktest/*_v42.out); drytest835 13/13. Smoke run of p13f4b (paced, mh_04_difficult design row, outside the board, D:/xoneslam_runs/xclean/smoke_p13f4b_20261028, 13:43): VALID, no non-allow-listed load, loop worker clean, frame path e1..mf with the GPU front, witness stages 0.97-1.02 x the 05:21 reference; it also compiled p13f4b's numba cache (__pycache__, outside the md5).

## 9 (v4.2). Relaunch on the twin (supersedes section 9's SNAP, md5 and draw-1 reuse; lead, 2026-10-28; FREEZE_RECORD section 6)
- Board of record: SNAP = engines/p13f4b, `_engine_md5` **1e49d634f155882f9177d3cd2bd36dfc**; SNAP_LS = engines/p13f4 (601a765b); P1.2
  engines/p12b_int2 (1cdc6b2c). No K14 run is a draw. The 6bb6e65a launch is pre-fix (8.36, 8.38 a).
- Command lines (D:/xoneslam_runs/xclean/board_relaunch.sh runs the first two; it is also the relaunch after an outage):
```
SNAP=D:/xoneslam_runs/xclean/engines/p13f4b SNAP_MD5=1e49d634f155882f9177d3cd2bd36dfc \
  SNAP_LS=D:/xoneslam_runs/xclean/engines/p13f4 bash D:/xoneslam_runs/xclean/board_block.sh                  # rounds 1-3, A A2 B C D
SNAP=... SNAP_MD5=... SNAP_LS=... PHASES=E bash D:/xoneslam_runs/xclean/board_block.sh                      # lock-step
CUDA_VISIBLE_DEVICES=-1 C:/Users/USER/anaconda3/envs/cvcuda133/python.exe -W ignore runs/xclean/board/board_eval.py \
  --snap-md5 1e49d634f155882f9177d3cd2bd36dfc --snap-ls-md5 601a765b4de7bd117f3042ad8def3f29 \
  --snap-dir D:/xoneslam_runs/xclean/engines/p13f4b --k14-ledger D:/xoneslam_runs/xclean/k14_ledger.tsv \
  --prefix-md5 6bb6e65ae901c55288c78cc3cceda22f                                                              # evaluation (8.38)
```

### 8 (v4.3). OWNER RULING on outside load (8.40; lead, 2026-10-28 16:34 IST; the block was stopped by TERM at 16:07:42, between
runs, after madmax_B0 attempt 8 ended; no accuracy number of any board run read; v4.2 block kept in v42_20261028_1610/)
- **Owner, verbatim (chat, 2026-10-28 ~15:55 and ~16:00 IST):** "Switch to the rule where outside load is reported but never voids
  a run , we are only running the n - 3 now and no gpu work , the system will only encounter casual load till board runs , if the
  run performance is effected we immodestly measure it and re run it" and, answering the lead's proposed clause, "no do not add the
  clause , i understand the intend is to present numbers that came from a idle machine , but casual load will not count and will
  be considered as idle machine and not void run until board completes because of system load".
- **The clause the owner declined (the lead's proposal, verbatim from the chat):** "I'll also add your "measure immediately and re-run" clause. If a run had heavy outside load (any outside process at a full core or more), the block will run the drift check straight after it, instead of waiting for 10 rows. If that check shows the machine slowed down, every run since the last good check is set aside and re-run by the block's existing rules.". Without it, no run is re-measured,
  re-run or marked because of outside load: that includes the "measure it and re run it" of the owner's first message.
- **Why (the lead's report to the owner, corrected):**
  - Protocol v2 (d) (LIVE_BARS section 0, amendment 2026-10-23 ~14:00) voided a paced run when any process outside the
    allow-list exceeded 5 % of one core in any 5-s monitor window (10 s in the pre-flight).
  - Load-VOIDs: K14 10 of 55 attempts (4 of them the lead's own python.exe 19600); board launch 1 (6bb6e65a) 12 of 23
    attempts (its 13th VOID, the 12:14 sentinel a1, was the 8.35 defect), madmax_B0 (52.5-min stream) in all 6; launch 2
    (1e49d634, OS processes exempt, 8.36) 1 of 9: madmax_B0 a7, nvcontainer.exe 37.8 % (the NVIDIA App's user container, a
    vendor service not on the 8.36 list); a8 was valid.
  - The witness stages of the design row mh_04_difficult, as ratios to the 05:21 instance (rectify p50 / p95, extract p50 / p95):
    - 12:14 a1, no outside load: 0.98 / 0.99, 1.02 / 1.02;
    - 12:14 a2 (svchost Dnscache 19.6 %, StateRepository 9.0 %, steam.exe 6.1 %): 1.01 / 1.06, 1.01 / 1.01;
    - 12:14 a3 (steam.exe 5.6 %): 1.02 / 1.05, 1.20 / 1.23 (outside +-20 %; VOID by the 8.35 test);
    - the 13:43 smoke run of p13f4b, no outside load: 1.00 / 1.02, 0.97 / 1.00;
    - the 13:45 opening sentinel (the board's witness reference), no outside load: 1.00 / 1.04, 1.09 / 1.12.
  - The recorded outside-load peaks do not predict the witness in either direction, and the witness varies by up to 23 % with
    no recorded cause. The loop worker's target-latency p95 was 0.115-0.120 s in all six instances, loaded or not. Casual load
    has no effect these six instances can detect; that is not a measurement of "no effect".
  - Heavy load does matter. In the 10-26 protocol-v3 test (6 parallel lock-step engine runs confined to CCD0, python.exe up to
    ~106 % of a core each, no GPU work), the witness stages were 1.19-1.52 x and B1 p50 / p95 / p99 1.09-1.18 x.
  - The owner states that the board window has only casual load and no GPU work.
- **Rule (8.40), applied alike by board_block.sh v4.3 state_of() and board_eval.py harness_valid():**
  - The harness reason "non-allow-listed load {...}" (protocol v2 d, CPU) never voids a board run. This holds whatever the
    process (OS, vendor or user application, or our own tooling) and whatever its size: there is no magnitude bound.
  - A run whose only reasons are load is valid. Any other reason keeps it VOID: feeder jitter / slope, engine tree not pinned,
    fence errors, a non-allowed GPU compute app, mode. Process exit codes are a FAIL.
  - The sentinel's witness test (8.35: rectify and extract, p50 and p95, within +-20 % of bsent1028134541) stays. It is the only
    machine-state test on the board, run at every launch, after every 10 row-draws and at the end; the relaunch's opening
    sentinel is the first.
  - Outside load is REPORTED: harness.json checks.others_over_5pct_core (every non-allow-listed process above 5 % of a core,
    with its peak; load under 5 % per process is not recorded, as before); board_eval `outside_load_pct_max` (that dict) and
    `harness_valid_raw` (the harness's protocol-v2 verdict).
  - The GPU clause is kept as the lead's reading of the ruling, which concerns casual CPU load and states that no GPU work runs.
    The GPU check is taken at the start of each run only, so "no GPU work" during a run is an operating commitment, not a check.
- **Supersedes:**
  - 2.1's validity clauses "pre-flight clean" and "no non-allow-listed process above 5 % of a core". The pre-flight was always a
    wait: neither the block nor the evaluator reads its json.
  - 5.3's pre-flight wait on CPU load. The pre-flight still waits (at most 90 min) while a non-allowed GPU compute app runs, and
    LOAD_HOLD is still honoured (5.1).
  - 8.36's "User applications are NOT exempt (steam.exe, OneDrive.exe stay void causes ...); neither are our own tools (python,
    bash, sleep, grep, conhost)": these are now reported, not void causes.
  - 3.1's label of our arm, which now reads "LIVE, causal, PACED (real time), validity v2 with outside CPU load reported, not
    voiding (8.40)".
  - 2.3's "valid" for the P1.2 references now reads "the harness's protocol-v2 verdict (unchanged by 8.40)".
- **Operating commitments stay (5.1, 8.39):** the lead runs no python, monitor, sampler, build, lock-step or GPU job during the
  paced rounds. Nothing voids such a job any more, so one started by mistake is disclosed with the runs it overlapped.
- **No discretionary re-run.** No run of record is re-run, replaced or dropped because of its load or its numbers. The only
  re-runs are the registered ones: 5.2 suspects after a failed sentinel, and the 5.3 classes.
- **Scope.** The rule applies from the v4.3 relaunch until the board completes: rounds 1-3, the retry pass, phase E (lock-step,
  which has no load test) and phase F. Every other paced run stays under protocol v2 (d).
  - The ledger's end line is the class of record (8.25): no attempt is re-classified.
  - On 1e49d634, the 13:45 opening sentinel and 7 row-draws (13:47-16:07) ended valid under v4.2, so none had an outside process
    above 5 %. They are valid under 8.40 too.
  - madmax_B0 attempt 7 (nvcontainer.exe 37.8 %), which 8.40 would accept, stays a parked VOID; attempt 8 stays that draw's run
    of record.
  - The 6bb6e65a attempts keep their classes (pre-fix appendix, 8.38 a).
  - The P1.2 paced references keep their protocol-v2 verdicts (board_eval pick_ref unchanged). On the 24 rows with a reference
    dir the selection is identical under either rule: the only load-only-VOID reference dir, now5_madmax_E1, sits behind a valid
    k13ref_madmax_E1. No reference is re-selected after its results were seen.
- **What 8.40 can make easier or harder** (printed beside the claim table):
  - Harder: C1, C3b, C7b, C7c and C8; C2 and C3a mostly harder.
  - Easier: C7a, through a slowed loop worker (CCD0, shared with outside load): later corrections move the within-run ratio
    toward 1.0. C7c does not bound this: its 2.0 s bar is 3.9-24 x the clean K14 worker target-latency p95 (0.08-0.51 s on 19
    of 20 seen rows with loop events; rosariov2_01 1.31 s).
  - Either way: C4, C5, C6 and the registered ratio. The only systematic path in our favour is that same worker, on rows where
    loop corrections cost accuracy (within-run ratio > 1). The competitors' numbers do not depend on load (ORB-SLAM3 offline,
    OKVIS2-X causal unpaced under K = 4 load, x1 lock-step), and the P1.2 references were taken under protocol v2 (d).
  - Every claim, through counting: no draw can end VOID-only for load any more, and fewer re-attempts mean fewer crash exposures
    for C3a and the system-failure rule.
- **Disclosure** (8.38 c and e extended; implemented with 8.38 after the paced rounds, before any board result is read):
  - per run of record: `outside_load_pct_max` and `harness_valid_raw`, printed beside B1 and the allow-list figures;
  - per table: the count of runs of record that protocol v2 (d) would have voided;
  - C7a, printed with C7c and, per draw: the worker target-latency p95, the queue-wait p95, the outside-load peaks, and a mark on
    every draw that protocol v2 (d) would have voided;
  - per row: the attempt count.
  - The report, website and paper describe the runs as "taken on the desktop with its normal background load, recorded per run;
    outside CPU load did not void a run (owner ruling 8.40)", and never as taken on an idle machine. C1 and C7c are stated with
    that sentence.
- Files:
  - board_block.sh v4.3 md5 1c9f0e0c (v4.2 87817c9c kept in v42_20261028_1610/);
  - board_eval.py md5 27c787cd (harness_valid, the k14_attempts key renamed harness_valid_raw, harness_valid_raw, outside_load_pct_max; d5887521 kept in v41_20261028_1230/ and
    v42_20261028_1610/);
  - fake harness (blocktest/fakesnap): "void" is a feeder reason, and the new "load" is a load reason (valid under 8.40).
  - Tests: blocktest/drytest840.sh 13/13 (real K14 and 12:14 dirs plus synthetic reasons); blocktest/selftest4 / selftest41 / selftest2v4 on v4.3 identical to v4.2 apart from one PID (blocktest/*_v43.out); drytest835 on v4.3: 11 PASS + 2 intended changes (12:14 a2 now valid; a3 VOID by the 8.35 witness); a stub test of the v4.3 wait_clean (clean / CPU load: no wait; a GPU compute app, a hold or no pre-flight json: waits, 90-min cap). The independent verification (workflow wf_47df9a33-bc6; audit_v42/VERIFY_840_CODE.md, VERIFY_840_REG.md): block and evaluator classifiers disagree 0 times on 301 real design-dir harness.json files; go with fixes, all applied (a fresh pre-flight json per call, no json = keep waiting; the VOID detail keeps the load reasons; this text).

## 9 (v4.3). Command lines unchanged (section 9 (v4.2)); board_block.sh is v4.3 (md5 1c9f0e0c), board_eval.py md5 27c787cd; lead, 2026-10-28 16:34 IST.

### 8 (v4.3) continued. 8.41: E: (USB) feeding of the 4Seasons rows; the registered E: -> D: copy (5.3); a disclosed overlap (lead, 2026-10-28 18:02 IST; no result read)
- **Event.** 17:29:35, the first 4Seasons row on E: (the USB drive), sealed row 4s_20200612_112 draw 1 attempt 1: VOID "feeder
  jitter/slope" (feeder_validity: replay slope 2.380, jitter p99 795 s; 1376 s wall for a 568-s stream).
- **The lead's first diagnosis was wrong.** It read the event as a throughput limit of E: for 30 Hz stereo PNGs and started the
  registered copy (5.3: "If a row is VOID in every attempt because its E: (USB) dataset feeds with jitter, it may be copied
  byte-identically (md5 checked) to D: and retried"), EARLY (after the first VOID, not all 6) and for all five 4Seasons rows on E:.
  Attempt 2 of the same row, still read from E: (17:30:46-17:40:26), was VALID, and 4s_20201007_144 attempt 1 from E: was VALID too.
  The likely cause is a COLD read of the USB disk: the first attempt read the files cold; attempt 2 found them in the Windows file
  cache; 144's files had just been read by the copy. Not proven; recorded as the working explanation.
- **Copies made** (runs/xclean/board/copy_e2d.py: every file's md5 read from E: == md5 re-read from D:; a manifest per sequence; the
  tag's "dataset" in runs/extra_rows.json re-pointed only after the whole sequence verified; the original registry, md5 b8ef2064, kept
  as board/v42_20261028_1610/extra_rows_before_841.json; extra_rows.json is outside every md5 of record):
  - ver2_4s_20200612_112: 34394 files, 5.00 GB, combined md5 34c9eb05cfe4d8f5eaa5c3388231fa05 -> D:/wm/datasets/4seasons_oak_e2d/
    fourseasons_2020-06-12_11-26-43 (switched ~17:38, after its attempt 2 had started from E:);
  - ver2_4s_20201007_144: 20066 files, 2.96 GB, combined md5 66605ac762cb8d3e9d7160c2f05b0256 -> .../fourseasons_2020-10-07_14-47-51
    (switched ~17:52, after its attempt 1 had run from E:).
  - Draw 1 of both rows was read from E:; their draws 2 and 3 read the byte-identical D: copies. Identical bytes: no effect on any
    result.
  - The copy of 4s_20201222_115 was stopped part-way (18:01:04); never switched; the partial dir is moved aside as
    D:/wm/datasets/4seasons_oak_e2d/_ABORTED_partial_fourseasons_2020-12-22_11-54-24 and is never used. 115, 131, 141 stay on E:.
- **Disclosed overlap (against 8.39 / 8.40's operating commitment: no lead job during the paced rounds).** The copy process
  (python.exe 8232, reading E: and writing D:) ran 17:32:38-18:01:04, during four paced runs: 4s_20200612_112 a2 (valid),
  madmax_H0 a1 (valid), 4s_20201007_144 a1 (valid) and madmax_A0 a1 (VOID "feeder jitter/slope"; the copy was reading the same USB
  disk, python.exe 8232 at 22.6 % of a core): that VOID is the lead's doing and stays a VOID of record (8.25). The three valid runs
  are runs of record; the overlap is printed beside them.
- **From here:** no copy or other lead job runs during the paced rounds. A 4Seasons or madmax row on E: whose first (cold) attempt
  VOIDs on the feeder is retried by the block as registered (5.3); if an E: row is VOID in every attempt, 5.3's copy is made with the
  block paused between runs.

### 8 (v4.3) continued. 8.42: the remaining 4Seasons rows on E: copied to D: with the block paused (lead, 2026-10-28 21:37 IST; no result read)
- **Evidence (feeder only, no result read):** every 30 Hz 4Seasons row on E: VOIDed on the feeder at its first (cold) read and passed
  once its files were in the Windows file cache: 4s_20200612_112 a1 (slope 2.38), 4s_20201222_115 a1 (slope 1.08, lag p99 30 s),
  4s_20210107_131 a1, 4s_20210225_141 a1; after the two session restarts of the evening (~20:16, ~20:45; the second cut 141 a2,
  INCOMPLETE) 141 a3 VOIDed cold again (21:00:04). Cold metadata reads of E: are slow (a `du` of one sequence dir did not finish
  in 5 min). The 15 Hz madmax rows on E: read cold in time (madmax_A1 a1 valid).
- **Action (prereg 5.3's registered copy, 8.41's commitment "with the block paused between runs"):** the block was stopped by TERM
  at 21:00:34 during its pause after 141 a3 (exit 143 at 21:01:04, PACED_ACTIVE removed, no run started); nothing else read E:;
  runs/xclean/board/copy_e2d.py copied ver2_4s_20210225_141, ver2_4s_20201222_115 and ver2_4s_20210107_131 to
  D:/wm/datasets/4seasons_oak_e2d/ (every file's md5 read from E: == re-read from D:; manifests; extra_rows.json re-pointed per tag
  after its sequence verified). Copies: ver2_4s_20210225_141 25086 files 3.72 GB md5 6cba5da9e60a443a2a0dbb527cf9dcee; ver2_4s_20201222_115 19556 files 2.74 GB md5 9b991d276d5902fff5ac2bc1940ca0ec; ver2_4s_20210107_131 24052 files 3.41 GB md5 7852ee484d56056c4c7c3bf17aac78d9 (all switched by 21:36:23; the earlier partial 115 dir stays aside, unused).
- **Early trigger, declared:** 5.3's trigger is "VOID in every attempt"; 115 and 131 already have valid draw-1 runs (warm retries)
  and 141 has 2 VOIDs + 1 INCOMPLETE of its budget. The copy is made now because every cold read of these rows fails (each power or
  session event and each new round starts cold) and a byte-identical copy cannot change a result. Draw-1 runs of 115 and 131 were
  read from E:; all later draws of all five 4Seasons rows read the D: copies. All 4Seasons board rows are now on D: (or C:).
- Then the block was relaunched (board_relaunch.sh): the ledger resumes; the opening sentinel re-checks the machine.

### 8 (v4.4). 8.43: the INCOMPLETE limit raised from 3 to 10 (lead, 2026-10-28 23:45 IST; between runs; no result read)
- **Why.** 5.3 leaves a draw without a run after 3 INCOMPLETE attempts (an attempt whose block died: no end line, no harness.json at
  the relaunch). The limit was meant to stop an endless loop on a row that keeps killing the block. On 2026-10-28 the owner's site had
  four power / session outages in about 3.5 h (~20:16, ~20:45, ~23:1x, ~23:4x; the owner reported each in chat and the lead
  relaunched with board_relaunch.sh). They cut 4s_20210225_141 a2 and madmax_B4 a1 and a2 mid-run: madmax_B4 draw 1 (an unseen row,
  an 18-min stream) had 2 of its 3 INCOMPLETEs, so one more outage during it would have left that draw without a run and failed every
  claim of the U table for a reason outside the system and the machine checks. Every INCOMPLETE so far coincides with an
  owner-reported outage.
- **Change (board_block.sh v4.4, md5 625e6745; v4.3 1c9f0e0c kept as v42_20261028_1610/board_block_v43.sh):** INC_MAX = 10 for the
  paced rounds and phase E alike (was 3); nothing else changes. INCOMPLETE stays "parked, not an attempt"; the VOID / FAIL budgets
  and the 2-FAIL system-failure rule are unchanged; the evaluator never applied the limit (it lists INCOMPLETE attempts).
- **Applied** with the block stopped by TERM right after the 23:43 opening sentinel (between runs), then relaunched; madmax_B4's
  INCOMPLETE a1 and a2 keep their records.
## 9 (v4.4). Command lines unchanged; board_block.sh is v4.4 (md5 625e6745); lead, 2026-10-28 23:45 IST.

- **8.38 implemented (2026-10-30, before any board result was read):** board_eval.py 27c787cd -> ad421f2d (builder, IMPL_838.md) -> **27bc3e9d** (lead: audit D1 + D2, AUDIT_838.md). Reporting only: claims, verdicts and thresholds byte-identical on the dry-test inputs (design rows, pre-fix K14 runs, sentinel witness dirs, fabricated scratch-ledger lines).

### 8 (v4.5). 8.44: an outside app that only HOLDS a GPU context is reported, never waits or voids (lead, 2026-10-30 ~09:4x IST; owner "do what you need"; no board result read)
- **What happened.** After the 10-30 ~03:2x power outage, Microsoft Copilot (copilotapp.exe, copilotapphost.exe) auto-started and held
  a GPU context. The relaunch (03:31:08) waited 90 min in the pre-flight, then the opening sentinel bsent1030033112 VOIDed 3 times on
  "non-allowed GPU apps" (05:03, 06:36, 08:09; pre-flight GPU utilisation 1-2 %); the block stopped (STOP_VOID, exit 3) at 08:10:27.
  No draw of record was parked (the last good sentinel was the 02:03:40 closing one; no paced run of record after it); 216 / 216 draws
  stay valid. The lead stopped the Copilot processes (owner: "do what you need and complete the board") and relaunched at 09:11:55;
  the opening sentinel bsent1030091159 was valid at 09:14:02 and phase E resumed at 09:15:56. ~5.5 h were lost. The lead's watcher
  did not read the pre-flight json (fixed: it now alerts on any non-allowed GPU app within ~60 s). The owner holds the lead
  responsible: a normal desktop app must not block the board.
- **Rule (8.44), board_block.sh v4.5 (md5 1843bb0b; v4.4 625e6745 archived as runs_archive_board_block_v44.sh at the swap):**
  - state_of(): the harness reason "non-allowed GPU apps [...]" is REPORTED, not a VOID, when the attempt's own pre-flight json
    (written just before the attempt, before our engine starts, so all GPU use it sees is outside work) shows GPU utilisation < 10 %;
    with no / unreadable json, or utilisation >= 10 %, the VOID stands.
  - wait_clean(): a non-allowed GPU app holds the attempt only when the pre-flight GPU utilisation is >= 10 %.
  - 10 % is a declared bound (class D): the desktop baseline at every recorded pre-flight was 1-4 %; real outside GPU work (a game,
    a render, a training job) runs far above it. Real outside GPU work still blocks, because it would distort latency.
- **Scope.** Applies from the next relaunch only (relaunch_v45.sh swaps v4.5 in when no block process is alive; the running v4.4 is never
  edited). Attempts classified under v4.4 keep their ledger class (8.25: the ledger end line is the class of record): the 3 Copilot
  sentinel VOIDs stay VOID and are disclosed. board_eval.py (27bc3e9d) is unchanged: no run of record has a GPU-app reason.
- **Dry test on real directories (real-output rule):** state_of() of v4.4 and v4.5, extracted from the scripts, on copies of
  board_void/bsent1030033112_mh_04_difficult__a3 with its real pre-flight json (1 %): v4.4 VOID, v4.5 valid (8.35 witness test passed);
  the same with utilisation edited to 55 %: VOID / VOID; with no json: VOID / VOID; the smoke run: valid / valid. wait_clean rule on
  the real jsons: Copilot 1 % -> v4.4 HOLD, v4.5 proceed; 55 % -> HOLD / HOLD; the clean 09:12 json -> proceed / proceed.

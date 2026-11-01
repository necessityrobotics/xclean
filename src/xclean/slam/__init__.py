# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""xclean P1.3: loop closure, global correction and relocalisation in a background WORKER (never in the frame path).

Modules: geom (SE(3)), packets (messages, recorded streams),
rich (rich ORB set + BoW + stereo 3D), kfdb (database, inverted file, retrieval), verify (rich-set PnP + information),
admit (chain, kappa(t), admission), headgraph (anchored head filter), merge (R1 segment merging), worker (WorkerCore,
the live process entry point), replay (the offline causal replay).  The frame side is xclean/live/p13_link.py.  CPU only; no learned weights; own code.
"""

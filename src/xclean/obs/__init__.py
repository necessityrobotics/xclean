# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Observability - the budget, the ledger, and the stamps that make a number
traceable.  Nothing here imports torch or cupy: the ledger has to be cheap
enough to leave switched on, and a dependency-light module is also one that
can be unit-tested on CPU in milliseconds.
"""

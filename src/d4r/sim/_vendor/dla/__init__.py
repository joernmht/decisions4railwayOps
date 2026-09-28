"""Vendored deadlock-avoidance heuristic (DLA) from flatland-baselines (MIT).

DLA is the strongest simple baseline on Flatland 4.2.x (0 deadlocks in our census). In this lab it
plays the part of the *interlocking*: it guarantees that a train only moves when the move cannot
lead to a head-on deadlock, and it supplies the default route. Dispatching engines then decide
among options that DLA has already judged safe.
"""

from .deadlock_avoidance_policy import DeadLockAvoidancePolicy

__all__ = ["DeadLockAvoidancePolicy"]

"""Finance-oriented Q preset aliases."""

from __future__ import annotations

import updatesupport as us


def q_portfolio_mix_shift(radius: float = 0.5) -> us.QPreset:
    """Constrain each hidden portfolio-cell mass around its observed mass."""

    return us.q_bounded_shift(radius)


def q_exposure_weighted_tv(
    radius: float,
    *,
    backend: str = "cvxpy",
) -> us.QPreset:
    """Constrain total hidden portfolio mass shift under the observed weights."""

    return us.q_tv_budget(radius, backend=backend)

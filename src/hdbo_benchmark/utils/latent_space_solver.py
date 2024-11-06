"""Implements an abstract latent space solver

A latent space solver takes the usual ingredients of a
poli-baselines solver (i.e. a black box, an initial guess x0,
and an initial observation y0), but also the class of a continuous
optimizer. This continuous optimizer is used to optimize in latent space
by encoding the history, querying the continuous optimizer for a new
latent space point, and then decoding this point to get a new candidate
point in the original space.
"""

from typing import Callable, Tuple, Type

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox  # type: ignore[import]
from poli_baselines.core.abstract_solver import AbstractSolver  # type: ignore[import]


class LatentSpaceSolver(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        continuous_optimizer_class: Type[AbstractSolver],
        encoder: Callable[[np.ndarray], np.ndarray],
        decoder: Callable[[np.ndarray], np.ndarray],
        **kwargs_for_continuous_optimizer,
    ):
        super().__init__(black_box, x0, y0)

        continuous_optimizer = continuous_optimizer_class(
            black_box=black_box,
            x0=encoder(x0),
            y0=y0,
            **kwargs_for_continuous_optimizer,
        )
        self.continuous_optimizer = continuous_optimizer
        self.encoder = encoder
        self.decoder = decoder

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the solver for one iteration.
        """
        z = self.continuous_optimizer.next_candidate()
        x = self.decoder(z)
        y = self.black_box(x)

        # TODO: in an ideal world, we would
        # only maintain a single history. We could
        # update history to be a property instead.

        # Updating this solver's history
        self.update(x, y)
        self.post_update(x, y)

        # Updating the continuous optimizer's history
        self.continuous_optimizer.update(z, y)
        self.continuous_optimizer.post_update(z, y)
        self.iteration += 1

        return x, y

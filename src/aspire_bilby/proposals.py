"""Proposal distributions backed by Bilby objects."""

from collections.abc import Sequence

import numpy as np
from bilby.core.prior import PriorDict


class BilbyPriorProposal:
    """Use a Bilby prior dictionary as an Aspire proposal.

    The proposal delegates both sampling and density evaluation to the same
    ``PriorDict`` used by Bilby. This preserves joint constraints and prior
    conversion functions while presenting samples in Aspire's parameter order.

    Parameters
    ----------
    priors : PriorDict
        Bilby priors used for sampling and density evaluation.
    parameters : sequence of str
        Ordered search parameters corresponding to columns in Aspire samples.
    """

    xp = np

    def __init__(self, priors: PriorDict, parameters: Sequence[str]):
        self.priors = priors
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("At least one proposal parameter is required.")

        missing = set(self.parameters) - set(priors)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"Parameters are missing from the priors: {missing_names}")

    @property
    def dims(self) -> int:
        """Number of sampled dimensions."""
        return len(self.parameters)

    def sample_and_log_prob(self, n_samples: int):
        """Draw samples from the prior and evaluate their log density."""
        if n_samples < 1:
            raise ValueError("n_samples must be at least one.")

        theta = self.priors.sample(size=n_samples)
        x = np.column_stack(
            [np.atleast_1d(theta[parameter]) for parameter in self.parameters]
        )
        return x, self.log_prob(x)

    def log_prob(self, x):
        """Evaluate the normalized joint prior density."""
        x = np.asarray(x)
        if x.ndim == 1:
            if self.dims == 1:
                x = x.reshape(-1, 1)
            elif x.shape == (self.dims,):
                x = x.reshape(1, self.dims)
        if x.ndim != 2 or x.shape[1] != self.dims:
            raise ValueError(
                f"x must have shape (n_samples, {self.dims}), got {x.shape}."
            )

        theta = dict(zip(self.parameters, x.T))
        return np.atleast_1d(self.priors.ln_prob(theta, axis=0))

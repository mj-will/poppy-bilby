"""Example plugin for using a sampler in bilby.

Here we demonstrate the how to implement the class.
"""
from functools import partial

import bilby
from bilby.core.utils.random import rng
from bilby.core.utils.log import logger
from bilby.core.sampler.base_sampler import Sampler
import os
import poppy

from .utils import (
    get_poppy_functions,
    get_prior_bounds,
    get_periodic_parameters,
    samples_from_bilby_result,
)


class Poppy(Sampler):
    """Bilby wrapper for poppy.

    Poppy: https://github.com/mj-will/bayesian-poppy
    """

    sampler_name = "poppy"
    """
    Name of the sampler.
    """

    @property
    def external_sampler_name(self) -> str:
        """The name of package that provides the sampler."""
        return "poppy"

    @property
    def default_kwargs(self) -> dict:
        """Dictionary of default keyword arguments."""
        return dict(
            n_samples=1000,
            initial_result_file=None,
            flow_matching=False,
            npool=None,
        )

    def run_sampler(self) -> dict:
        """Run the sampler."""
        self.kwargs.pop("resume", None)

        initial_result = bilby.core.result.read_in_result(
            self.kwargs.pop("initial_result_file")
        )
        initial_samples = samples_from_bilby_result(
            initial_result, self.search_parameter_keys
        )

        n_samples = self.kwargs.pop("n_samples")

        base_transforms = {
            p: "periodic" for p in get_periodic_parameters(self.priors)
        }
        transforms = self.kwargs.pop("transforms", {})
        base_transforms.update(transforms)

        funcs = get_poppy_functions(
            self.likelihood,
            self.priors,
            self.search_parameter_keys,
            use_ratio=self.use_ratio,
        )

        prior_bounds = get_prior_bounds(self.priors, self.search_parameter_keys)

        self._setup_pool()

        if self.pool:
            log_likelhood_fn = partial(funcs.log_likelihood, map_fn=self.pool.map)
        else:
            log_likelhood_fn = funcs.log_likelihood

        pop = poppy.Poppy(
            log_likelihood=log_likelhood_fn,
            log_prior=funcs.log_prior,
            dims=self.ndim,
            parameters=self.search_parameter_keys,
            prior_bounds=prior_bounds,
            transforms=base_transforms,
            **self.kwargs,
        )

        history = pop.fit(initial_samples)

        if self.plot:
            history.plot_loss().savefig(os.path.join(self.outdir + "loss.png"))

        samples = pop.sample_posterior(n_samples).to_numpy()
        iid_samples = samples.rejection_sample(rng=rng)

        self.result.samples = iid_samples.x
        self.result.nested_samples = samples.to_dataframe(flat=True)
        self.result.nested_samples["log_likelihood"] = samples.log_likelihood
        self.result.nested_samples["log_prior"] = samples.log_prior
        self.result.nested_samples["weights"] = samples.weights
        self.result.log_likelihood_evaluations = iid_samples.log_likelihood
        self.result.log_prior_evaluations = iid_samples.log_prior
        self.result.log_evidence = samples.log_evidence
        self.result.log_evidence_err = samples.log_evidence_error
        return self.result

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Get lists of the expected outputs directories and files.

        These are used by :code:`bilby_pipe` when transferring files via
        HTCondor. Both can be empty.

        Parameters
        ----------
        outdir : str
            The output directory.
        label : str
            The label for the run.

        Returns
        -------
        list
            List of file names.
        list
            List of directory names.
        """
        filenames = [os.path.join(outdir, "loss.png")]
        dirs = []
        return filenames, dirs

    def _verify_kwargs_against_default_kwargs(self):
        """Check for additional kwargs that are not included in the defaults.

        Since the arguments for poppy depend on the flow being used, arguments
        are not removed if they are not present in the defaults.
        """
        args = self.default_kwargs
        for user_input in self.kwargs.keys():
            if user_input not in args:
                logger.debug(
                    (
                        "Supplied argument '{user_input}' is not a default "
                        f"argument of '{self.__class__.__name__}'. "
                    )
                )
    def _translate_kwargs(self, kwargs):
        """Translate the keyword arguments"""
        if "npool" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["npool"] = kwargs.pop(equiv)
                    break
            # If nothing was found, set to npool but only if it is larger
            # than 1
            else:
                if self._npool > 1:
                    kwargs["npool"] = self._npool
        super()._translate_kwargs(kwargs)

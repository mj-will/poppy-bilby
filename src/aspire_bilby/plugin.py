"""Plugin for the aspire sampler in bilby."""

from functools import partial
from typing import Callable

import bilby
from bilby.core.utils.random import rng
from bilby.core.utils.log import logger
from bilby.core.sampler.base_sampler import Sampler
import copy
import os
import numpy as np
import aspire
from aspire.samples import Samples
from aspire.utils import configure_logger, PoolHandler

from .utils import (
    get_function_from_path,
    get_aspire_functions,
    get_prior_bounds,
    get_periodic_parameters,
    samples_from_bilby_result,
    samples_from_bilby_priors,
)


class Aspire(Sampler):
    """Bilby wrapper for aspire.

    Aspire: https://github.com/mj-will/bayesian-aspire

    Since aspire is designed to be called in multiple steps, specific keyword
    arguments are used for each step:
    - `fit_kwargs` for the fit step
    - `sample_kwargs` for the sampling step
    In addition, there are custom arguments for handling e.g. logging:
    - `aspire_log_level` for the logging level of aspire
    - `initial_conversion_function` for a function to convert the initial
    samples.
    - `sample_from_prior` to specify parameters that should be sampled from the
    prior regardless of whether they are in the initial result file.


    It also includes a method to read initial samples from a bilby result
    """

    sampler_name = "aspire"
    """
    Name of the sampler.
    """

    @property
    def external_sampler_name(self) -> str:
        """The name of package that provides the sampler."""
        return "aspire"

    @property
    def default_kwargs(self) -> dict:
        """Dictionary of default keyword arguments."""
        return dict(
            n_samples=1000,
            initial_result_file=None,
            flow_matching=False,
            npool=None,
        )

    def get_conversion_function(
        self, conversion_function: Callable | str = None
    ) -> Callable:
        """Get the conversion function from a string or return None."""
        if conversion_function is None:
            return None
        if isinstance(conversion_function, str):
            logger.debug(
                "Conversion function is a string, trying to get it from the path."
            )
            conversion_function = get_function_from_path(conversion_function)
        if not callable(conversion_function):
            raise TypeError(
                f"Conversion function {conversion_function} is not callable."
            )
        return conversion_function

    def read_initial_samples(
        self,
        initial_result_file: str,
        sample_from_prior: list[str] = None,
        conversion_function: Callable | str = None,
    ) -> Samples:
        """Read the initial samples from a bilby result file.

        If parameters are missing, they will be drawn from the prior.

        Parameters
        ----------
        initial_result_file : str
            The path to the initial result file.
        parameters_to_sample : list
            List of parameters to sample from the prior regardless of whether
            they are in the initial result file.
        sample_from_prior : list[str], optional
            List of parameters to sample from the prior regardless of whether
            they are in the initial result file. If None, all parameters will
            be sampled from the initial result file.
        conversion_function : Callable
            A function to convert the initial samples.
        """
        initial_result = bilby.core.result.read_in_result(initial_result_file)
        initial_samples = samples_from_bilby_result(
            initial_result,
            bilby_priors=self.priors,
            parameters=self.search_parameter_keys,
            sample_from_prior=sample_from_prior,
            conversion_function=conversion_function,
        )
        return initial_samples

    def run_sampler(self) -> dict:
        """Run the sampler."""

        kwargs = copy.deepcopy(self.kwargs)

        kwargs.pop("resume", None)
        n_samples = kwargs.pop("n_samples")
        n_initial_samples = kwargs.pop("n_initial_samples", 10_000)
        sample_from_prior = kwargs.pop("sample_from_prior", None)

        initial_result_file = kwargs.pop("initial_result_file", None)
        initial_samples = kwargs.pop("initial_samples", None)

        if initial_samples is not None and initial_result_file is not None:
            raise ValueError(
                "You cannot provide both `initial_samples` and "
                "`initial_result_file`. Please provide only one."
            )

        conversion_function = self.get_conversion_function(
            kwargs.pop("initial_conversion_function", None)
        )
        if initial_result_file is not None:
            logger.info(f"Initial samples will be read from {initial_result_file}.")

            initial_samples = self.read_initial_samples(
                initial_result_file,
                sample_from_prior=sample_from_prior,
                conversion_function=conversion_function,
            )
        elif initial_samples is not None:
            logger.info("Using provided initial samples.")
            if conversion_function is not None:
                logger.warning(
                    "Conversion function is ignored when initial samples are provided."
                )
        else:
            logger.info("Initial samples will be drawn from the prior.")
            initial_samples = samples_from_bilby_priors(
                self.priors, n_initial_samples, parameters=self.search_parameter_keys
            )

        disable_periodic_parameters = kwargs.pop("disable_periodic_parameters", False)

        if disable_periodic_parameters:
            periodic_parameters = []
        else:
            periodic_parameters = [p for p in get_periodic_parameters(self.priors)]

        funcs = get_aspire_functions(
            self.likelihood,
            self.priors,
            self.search_parameter_keys,
            use_ratio=self.use_ratio,
        )

        prior_bounds = get_prior_bounds(self.priors, self.search_parameter_keys)

        self._setup_pool()

        if self.pool:
            log_likelihood_fn = partial(funcs.log_likelihood, map_fn=self.pool.map)
        else:
            log_likelihood_fn = funcs.log_likelihood

        sample_kwargs = kwargs.pop("sample_kwargs", {})
        if n_final_samples := kwargs.pop("n_final_samples", None):
            sample_kwargs["n_final_samples"] = n_final_samples
        fit_kwargs = kwargs.pop("fit_kwargs", {})

        configure_logger(log_level=kwargs.pop("aspire_log_level", "INFO"))

        # Should handle these properly
        kwargs.pop("npool", None)
        kwargs.pop("pool", None)
        kwargs.pop("sampling_seed", None)

        logger.info(f"Creating aspire instance with kwargs: {kwargs}")
        pop = aspire.Aspire(
            log_likelihood=log_likelihood_fn,
            log_prior=funcs.log_prior,
            dims=self.ndim,
            parameters=self.search_parameter_keys,
            prior_bounds=prior_bounds,
            periodic_parameters=periodic_parameters,
            **kwargs,
        )

        logger.info(f"Fitting aspire with kwargs: {fit_kwargs}")
        history = pop.fit(initial_samples, **fit_kwargs)

        if self.plot:
            from aspire.plot import plot_comparison

            logger.debug("Plotting loss history")
            history.plot_loss().savefig(
                os.path.join(self.outdir, f"{self.label}_loss.png")
            )
            logger.debug("Plotting samples from flow")
            flow_samples = pop.sample_flow(10_000)

            fig = plot_comparison(
                initial_samples,
                flow_samples,
                per_samples_kwargs=[
                    dict(include_weights=False, color="C0"),
                    dict(include_weights=False, color="C1"),
                ],
                labels=["Initial samples", "Flow samples"],
            )
            fig.savefig(os.path.join(self.outdir, f"{self.label}_flow.png"))

        logger.info(f"Sampling from posterior with kwargs: {sample_kwargs}")

        self._setup_pool()

        with PoolHandler(pop, self.pool, close_pool=False):
            samples, sampling_history = pop.sample_posterior(
                n_samples, return_history=True, **sample_kwargs
            )
            samples = samples.to_numpy()

        self._close_pool()

        if self.plot and sampling_history is not None:
            sampling_history.plot().savefig(
                os.path.join(self.outdir, f"{self.label}_sampling_history.png")
            )

        if hasattr(samples, "log_w") and samples.log_w is not None:
            iid_samples = samples.rejection_sample(rng=rng)
        else:
            iid_samples = samples

        self.result.samples = iid_samples.x

        self.result.nested_samples = samples.to_dataframe(flat=True)
        self.result.nested_samples["log_likelihood"] = samples.log_likelihood
        self.result.nested_samples["log_prior"] = samples.log_prior
        if hasattr(samples, "weights") and samples.weights is not None:
            self.result.nested_samples["weights"] = samples.weights

        self.result.log_likelihood_evaluations = iid_samples.log_likelihood
        self.result.log_prior_evaluations = iid_samples.log_prior
        self.result.log_evidence = iid_samples.log_evidence or np.nan
        self.result.log_evidence_err = iid_samples.log_evidence_error or np.nan

        self.result.num_likelihood_evaluations = pop.n_likelihood_evaluations

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
        filenames = [
            os.path.join(outdir, f"{label}_loss.png"),
            os.path.join(outdir, f"{label}_sampling_history.png"),
            os.path.join(outdir, f"{label}_flow.png"),
        ]
        dirs = []
        return filenames, dirs

    def _verify_kwargs_against_default_kwargs(self):
        """Check for additional kwargs that are not included in the defaults.

        Since the arguments for aspire depend on the flow being used, arguments
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

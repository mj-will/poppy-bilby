from copy import deepcopy
from typing import Callable

from bilby.core.likelihood import Likelihood
from bilby.core.prior import PriorDict
from bilby.core.result import Result
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import os
import numpy as np
import pandas as pd

from bilby.core.utils.log import logger


Inputs = namedtuple(
    "Inputs",
    [
        "log_likelihood",
        "log_prior",
        "dims",
        "parameters",
        "prior_bounds",
        "periodic_parameters",
    ],
)
"""Container for the inputs to the aspire sampler."""

Functions = namedtuple("Functions", ["log_likelihood", "log_prior"])
"""Container for the log likelihood and log prior functions."""


@dataclass
class GlobalFunctions:
    """Dataclass to store global functions."""

    bilby_likelihood: Likelihood
    bilby_priors: PriorDict
    parameters: list
    use_ratio: bool


_global_functions = GlobalFunctions(None, None, [], False)


def update_global_functions(
    bilby_likelihood: Likelihood,
    bilby_priors: PriorDict,
    parameters: list[str],
    use_ratio: bool,
):
    """Update the global functions for log likelihood and log prior."""
    global _global_functions
    _global_functions.bilby_likelihood = bilby_likelihood
    _global_functions.bilby_priors = bilby_priors
    _global_functions.parameters = parameters
    _global_functions.use_ratio = use_ratio


def _global_log_likelihood(x):
    theta = dict(zip(_global_functions.parameters, x))
    _global_functions.bilby_likelihood.parameters.update(theta)

    if _global_functions.use_ratio:
        return _global_functions.bilby_likelihood.log_likelihood_ratio()
    else:
        return _global_functions.bilby_likelihood.log_likelihood()


def get_aspire_functions(
    bilby_likelihood,
    bilby_priors,
    parameters,
    use_ratio: bool = False,
    likelihood_dtype: str = "float64",
):
    """Get the log likelihood function for a bilby likelihood object."""

    update_global_functions(bilby_likelihood, bilby_priors, parameters, use_ratio)

    def log_likelihood(samples, map_fn=map):
        logl = -np.inf * np.ones(len(samples.x))
        # Only evaluate the log likelihood for finite log prior
        if samples.log_prior is None:
            raise RuntimeError("log-prior has not been evaluated!")
        mask = np.isfinite(samples.log_prior, dtype=bool)
        x = np.asarray(samples.x[mask, :], dtype=likelihood_dtype)
        logl[mask] = np.fromiter(
            map_fn(_global_log_likelihood, x),
            dtype=float,
        )
        return logl

    def log_prior(samples):
        x = dict(zip(parameters, np.array(samples.x).T))
        return bilby_priors.ln_prob(x, axis=0)

    return Functions(log_likelihood=log_likelihood, log_prior=log_prior)


def get_prior_bounds(
    bilby_priors: PriorDict, parameters: list[str]
) -> dict[str : np.ndarray]:
    """Get a dictionary of prior bounds."""
    return {
        p: np.array([bilby_priors[p].minimum, bilby_priors[p].maximum])
        for p in parameters
    }


def get_periodic_parameters(bilby_priors: PriorDict) -> list[str]:
    """Determine which parameters are periodic."""
    parameters = []
    for p in bilby_priors.keys():
        # Skip fixed parameters
        try:
            if bilby_priors[p].boundary == "periodic":
                parameters.append(p)
        except AttributeError:
            pass
    return parameters


def samples_from_bilby_result(
    result: Result,
    parameters: str = None,
    bilby_priors: PriorDict = None,
    sample_from_prior: list[str] = None,
    conversion_function: Callable | None = None,
):
    """Get samples from a bilby result object.

    Parameters
    ----------
    result : Result
        The bilby result object.
    parameters : str
        The parameters to read from the result. If None, all parameters will be read.
    bilby_priors : PriorDict
        The bilby prior object. If not specified, the initial result must contain
        all parameters.
    sample_from_prior : list[str]
        A list of parameters to explicitly sample from the prior rather reading
        from the result.
    """
    from aspire.samples import Samples
    # TODO: add option to load nested samples

    result = deepcopy(result)
    if conversion_function is not None:
        logger.info("Applying conversion function to the initial samples.")
        result.posterior = conversion_function(result.posterior)

    available_parameters = list(
        result.posterior.columns[result.posterior.nunique() > 1]
    )
    logger.info(f"Available parameters in result: {available_parameters}")

    if parameters is None:
        parameters = result.priors.non_fixed_keys
    elif (
        missing_parameters := set(parameters) - set(available_parameters)
        or sample_from_prior
    ):
        if bilby_priors is not None:
            # Sample the missing parameters
            samples_df = sample_missing_parameters(
                result,
                bilby_priors,
                parameters=parameters,
                parameters_to_sample=sample_from_prior,
            )
            # Check all parameters are present
            if not all(p in samples_df for p in parameters):
                raise ValueError("Not all parameters are present in the result.")
            samples = samples_df[parameters].to_numpy()
        else:
            raise RuntimeError(
                "Initial result does not contain all parameters and new priors "
                f"were not provided. Missing parameters: {missing_parameters}."
            )
    else:
        samples = result.posterior[parameters].to_numpy()
    return Samples(
        x=samples,
        parameters=parameters,
    )


def samples_from_bilby_priors(
    bilby_priors: PriorDict, n_samples: int, parameters: str = None
):
    """Get samples from a bilby prior object.

    Parameters
    ----------
    bilby_priors : PriorDict
        The bilby prior object.
    n_samples : int
        The number of samples to draw.
    parameters : str
        The parameters to sample. If None, all parameters will be sampled.
    """
    from aspire.samples import Samples

    if parameters is None:
        parameters = bilby_priors.non_fixed_keys
    theta = bilby_priors.sample(size=n_samples)
    x = np.array([theta[p] for p in parameters]).T
    return Samples(
        x=x,
        parameters=parameters,
    )


def sample_missing_parameters(
    bilby_result: Result,
    bilby_priors: PriorDict,
    parameters: list[str] = None,
    parameters_to_sample: list[str] = None,
) -> pd.DataFrame:
    """Sample the missing parameters from the bilby result.

    Parameters
    ----------
    bilby_result : Result
        The initial bilby result object.
    bilby_priors : PriorDict
        The bilby prior object.
    parameters : list[str]
        The parameters that should be present in the final set of samples.
        If not specified, the parameters will be inferred from the priors.
    parameters_to_sample : list[str]
        A list of parameters to explicitly sample from the prior rather reading
        from the result.

    Returns
    -------
    numpy.ndarray
        The samples from the bilby result and the missing parameters from the bilby priors.
        The order will be the same as :code:`parameters`.
    """
    if parameters is None:
        parameters = bilby_priors.non_fixed_keys

    initial_parameters = list(bilby_result.priors.non_fixed_keys)
    all_parameters = list(
        bilby_result.posterior.columns[bilby_result.posterior.nunique() > 1]
    )
    if parameters_to_sample is not None:
        logger.debug(f"Ignoring existing samples for: {parameters_to_sample}")
        for parameter in parameters_to_sample:
            # Remove the parameter from parameter lists if present
            # This ensures that the parameter is sampled from the prior
            if parameter in initial_parameters:
                initial_parameters.remove(parameter)
            if parameter in all_parameters:
                all_parameters.remove(parameter)
            else:
                logger.debug(f"{parameter} not in the initial result.")

    missing_parameters = list(set(parameters) - set(all_parameters))
    common_parameters = list(set(parameters) & set(all_parameters))
    extra_parameters = list(set(initial_parameters) - set(parameters))

    if extra_parameters:
        logger.warning(
            f"Extra parameters in the initial result: {extra_parameters}. "
            "These will be ignored when sampling."
        )

    logger.info(f"Found initial samples for: {common_parameters}")
    samples = bilby_result.posterior[common_parameters].copy()

    logger.info(f"Drawing samples for: {missing_parameters}")

    if missing_parameters:
        new_samples = bilby_priors.sample_subset(
            keys=missing_parameters, size=len(samples)
        )

        # Add the new samples to the initial samples
        for p in missing_parameters:
            samples[p] = new_samples[p]
    else:
        logger.info("No missing parameters to sample.")
    return samples


@contextmanager
def temporary_logger_level(logger, level: str | None):
    """Temporarily set the logger level.

    Example usage

    ```python
    with temporary_logger_level(logger, "DEBUG"):
        # Do something
    ```
    """
    initial_level = logger.level
    if level is not None:
        logger.setLevel(level)
    try:
        yield initial_level
    finally:
        logger.setLevel(initial_level)


def load_bilby_pipe_ini(
    config_file: str,
    data_dump_file: str,
    suppress_bilby_logger: bool = True,
):
    """Load a bilby_pipe ini file and return the likelihood and priors."""
    from bilby_pipe import data_analysis
    from bilby_pipe.utils import logger as bilby_pipe_logger
    from bilby.core.utils.log import logger as bilby_logger

    with (
        temporary_logger_level(bilby_pipe_logger, 0 if suppress_bilby_logger else None),
        temporary_logger_level(bilby_logger, 0 if suppress_bilby_logger else None),
    ):
        parser = data_analysis.create_analysis_parser()
        args, unknown_args = data_analysis.parse_args(
            [config_file, "--data-dump-file", data_dump_file], parser
        )
        analysis = data_analysis.DataAnalysisInput(args, unknown_args)
        likelihood, priors = analysis.get_likelihood_and_priors()
        priors.convert_floats_to_delta_functions()
        likelihood.parameters.update(priors.sample())
        return likelihood, priors


def get_inputs_from_bilby_pipe_ini(
    config_file: str,
    data_dump_file: str,
    use_ratio: bool = False,
    suppress_bilby_logger: bool = True,
):
    """Get the aspire inputs from a bilby_pipe ini file.

    Returns
    -------
    namedtuple
        A namedtuple with the log_likelihood and log_prior functions.
    """
    if not os.path.exists(config_file):
        raise OSError("Config file does not exist!")
    if not os.path.exists(data_dump_file):
        raise OSError("Data dump does not exist!")

    bilby_likelihood, bilby_priors = load_bilby_pipe_ini(
        config_file, data_dump_file, suppress_bilby_logger=suppress_bilby_logger
    )
    parameters = bilby_priors.non_fixed_keys
    funcs = get_aspire_functions(
        bilby_likelihood, bilby_priors, parameters, use_ratio=use_ratio
    )
    return Inputs(
        log_likelihood=funcs.log_likelihood,
        log_prior=funcs.log_prior,
        dims=len(parameters),
        parameters=parameters,
        prior_bounds=get_prior_bounds(bilby_priors, parameters),
        periodic_parameters=get_periodic_parameters(bilby_priors),
    )


def get_function_from_path(path: str):
    """Get a function from a module path.

    Parameters
    ----------
    path : str
        The path to the function, e.g. "module.submodule.function".

    Returns
    -------
    Callable
        The function object.
    """
    module_path, function_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)

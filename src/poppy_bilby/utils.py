from bilby.core.likelihood import Likelihood
from bilby.core.prior import PriorDict
from bilby.core.utils.log import logger as bilby_logger
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
import os
import numpy as np


Inputs = namedtuple(
    "Inputs", ["log_likelihood", "log_prior", "dims", "parameters", "prior_bounds", "periodic_parameters"]
)
"""Container for the inputs to the poppy sampler."""

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


def get_poppy_functions(
    bilby_likelihood,
    bilby_priors,
    parameters,
    use_ratio: bool = False,
):
    """Get the log likelihood function for a bilby likelihood object."""

    update_global_functions(
        bilby_likelihood, bilby_priors, parameters, use_ratio
    )

    def log_likelihood(samples, map_fn=map):
        logl = -np.inf * np.ones(len(samples.x))
        # Only evaluate the log likelihood for finite log prior
        if samples.log_prior is None:
            raise RuntimeError("log-prior has not been evaluated!")
        mask = np.isfinite(samples.log_prior, dtype=bool)
        logl[mask] = np.fromiter(
            map_fn(
                _global_log_likelihood,
                samples.x[mask, :],
            ), dtype=float,
        )
        return logl

    def log_prior(samples):
        x = dict(zip(parameters, np.array(samples.x).T))
        return bilby_priors.ln_prob(x, axis=0)

    return Functions(log_likelihood=log_likelihood, log_prior=log_prior)


def get_prior_bounds(bilby_priors: PriorDict, parameters: list[str]) -> dict[str: np.ndarray]:
    """Get a dictionary of prior bounds."""
    return {p: np.array([bilby_priors[p].minimum, bilby_priors[p].maximum]) for p in parameters}


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


def samples_from_bilby_result(result, parameters: str = None):
    """Get samples from a bilby result object."""
    from poppy.samples import Samples
    # TODO: add option to load nested samples
    if parameters is None:
        parameters = result.priors.non_fixed_keys
    return Samples(
        x=result.posterior[parameters].to_numpy(),
        parameters=parameters,
    )


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

    with temporary_logger_level(
        bilby_pipe_logger, 0 if suppress_bilby_logger else None
    ), temporary_logger_level(bilby_logger, 0 if suppress_bilby_logger else None):
        parser = data_analysis.create_analysis_parser()
        args, unknown_args = data_analysis.parse_args([config_file, "--data-dump-file", data_dump_file], parser)
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
    """Get the poppy inputs from a bilby_pipe ini file.

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
        config_file,
        data_dump_file,
        suppress_bilby_logger=suppress_bilby_logger
    )
    parameters = bilby_priors.non_fixed_keys
    funcs = get_poppy_functions(
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

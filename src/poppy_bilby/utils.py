from collections import namedtuple
from poppy.samples import from_numpy, to_numpy
import numpy as np


Inputs = namedtuple(
    "Inputs", ["log_likelihood", "log_prior", "dims", "parameters", "prior_bounds", "periodic_parameters"]
)


def get_poppy_log_likelihood(
    bilby_likelihood, parameters, use_ratio: bool = False, map_fn=map
):
    """Get the log likelihood function for a bilby likelihood object."""
    def fn(x):
        theta = dict(zip(parameters, x))
        bilby_likelihood.parameters.update(theta)
        if use_ratio:
            return bilby_likelihood.log_likelihood_ratio()
        else:
            return bilby_likelihood.log_likelihood()

    def log_likelihood(samples):
        logl = -np.inf * np.ones(len(samples.x))
        # Only evaluate the log likelihood for finite log prior
        mask = np.isfinite(to_numpy(samples.log_prior), dtype=bool)
        logl[mask] = np.fromiter(
            map_fn(fn, to_numpy(samples.x)[mask, :]), dtype=float,
        )
        return from_numpy(logl)

    return log_likelihood


def get_poppy_log_prior(bilby_priors, parameters):
    """Get the log prior function for a bilby prior object"""
    def log_prior(samples):
        x = dict(zip(parameters, to_numpy(samples.x).T))
        return from_numpy(bilby_priors.ln_prob(x, axis=0))

    return log_prior

def get_prior_bounds(bilby_priors, parameters):
    return {p: np.array([bilby_priors[p].minimum, bilby_priors[p].maximum]) for p in parameters}

def get_periodic_parameters(bilby_priors):
    return [p for p in bilby_priors.keys() if bilby_priors[p].boundary == "periodic"]


def samples_from_bilby_result(result, parameters: str = None):
    """Get samples from a bilby result object."""
    from poppy.samples.numpy import NumpySamples
    # TODO: add option to load nested samples
    if parameters is None:
        parameters = result.priors.non_fixed_keys
    return NumpySamples(
        x=result.posterior[parameters].to_numpy(),
        parameters=parameters,
    )


def load_bilby_pipe_ini(config_file: str, data_dump_file: str):
    """Load a bilby_pipe ini file and return the likelihood and priors."""
    from bilby_pipe import data_analysis

    parser = data_analysis.create_analysis_parser()
    args, unknown_args = data_analysis.parse_args([config_file, "--data-dump-file", data_dump_file], parser)
    analysis = data_analysis.DataAnalysisInput(args, unknown_args)
    likelihood, priors = analysis.get_likelihood_and_priors()
    priors.convert_floats_to_delta_functions()
    likelihood.parameters.update(priors.sample())
    return likelihood, priors


def get_inputs_from_bilby_pipe_ini(
    config_file: str, data_dump_file: str, use_ratio: bool = False, map_fn=map
):
    """Get the poppy inputs from a bilby_pipe ini file.

    Returns
    -------
    namedtuple
        A namedtuple with the log_likelihood and log_prior functions.
    """
    bilby_likelihood, bilby_priors = load_bilby_pipe_ini(config_file, data_dump_file)
    parameters = bilby_priors.non_fixed_keys
    log_likelihood = get_poppy_log_likelihood(
        bilby_likelihood, parameters, use_ratio=use_ratio, map_fn=map_fn
    )
    log_prior = get_poppy_log_prior(bilby_priors, parameters)
    return Inputs(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=len(parameters),
        parameters=parameters,
        prior_bounds=get_prior_bounds(bilby_priors, parameters),
        periodic_parameters=get_periodic_parameters(bilby_priors),
    )

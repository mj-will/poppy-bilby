from unittest.mock import patch

import bilby
import numpy as np
import pytest


def model(x, m, c):
    return m * x + c


def conversion_func(parameters):
    # d = |m| + |c|
    parameters["d"] = abs(parameters["m"]) + abs(parameters["c"])
    return parameters


@pytest.fixture()
def bilby_likelihood(rng):
    x = np.linspace(0, 10, 100)
    injection_parameters = dict(m=0.5, c=0.2)
    sigma = 1.0
    y = model(x, **injection_parameters) + rng.normal(0.0, sigma, len(x))
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma)
    return likelihood


@pytest.fixture()
def bilby_priors():
    priors = bilby.core.prior.PriorDict(conversion_function=conversion_func)
    priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
    priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
    # priors["d"] = bilby.core.prior.Constraint(name="d", minimum=0, maximum=5)
    return priors


@pytest.fixture()
def sampler_kwargs():
    return dict(
        n_samples=100,
        sample_kwargs=dict(
            sampler="smc",
            adaptive=True,
        ),
    )


@pytest.fixture(params=["zuko", "flowjax"])
def flow_backend(request):
    return request.param


@pytest.fixture(params=[None, "samples", "result"])
def existing_result(request, bilby_priors, tmp_path):
    if request.param is None:
        return {}
    elif request.param == "samples":
        from aspire.samples import Samples

        parameters = list(bilby_priors.non_fixed_keys)
        theta = bilby_priors.sample(500)
        theta_array = np.array([theta[p] for p in parameters]).T
        initial_samples = Samples(theta_array, parameters=parameters)
        return {"initial_samples": initial_samples}
    elif request.param == "result":
        import pandas as pd

        # Make a fake bilby result
        outdir = tmp_path / "existing_result"
        result_file = outdir / "existing_result.hdf5"

        samples = pd.DataFrame(bilby_priors.sample(100))
        result = bilby.core.result.Result(
            outdir=outdir,
            label="existing_result",
            priors=bilby_priors,
            posterior=samples,
            search_parameter_keys=list(bilby_priors.non_fixed_keys),
        )
        result.save_to_file(filename=result_file)
        return {"initial_result_file": result_file}


def test_run_sampler(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    sampler_kwargs,
    existing_result,
    flow_backend,
):
    outdir = tmp_path / "test_run_sampler"

    sampler_kwargs.update(**existing_result)

    bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        sampler="aspire",
        outdir=outdir,
        flow_backend=flow_backend,
        **sampler_kwargs,
    )


def test_run_sampler_pool(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    sampler_kwargs,
):
    from multiprocessing.dummy import Pool

    outdir = tmp_path / "test_run_sampler_pool"

    with patch("multiprocessing.Pool", new=Pool):
        bilby.run_sampler(
            likelihood=bilby_likelihood,
            priors=bilby_priors,
            sampler="aspire",
            outdir=outdir,
            npool=2,
            **sampler_kwargs,
        )

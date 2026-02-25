from unittest.mock import patch

import bilby
import numpy as np
import pytest


@pytest.fixture(params=["zuko", "flowjax"])
def flow_backend(request):
    return request.param


@pytest.fixture(params=["importance", "emcee", "smc"])
def sample_kwargs(request):
    """Kwargs for the sampler method in aspire"""
    if request.param == "importance":
        return dict(
            sampler="importance",
        )
    elif request.param == "emcee":
        return dict(
            sampler="emcee",
            nwalkers=10,
            nsteps=20,
        )
    elif request.param == "smc":
        return dict(
            sampler="smc",
        )
    else:
        raise ValueError(f"Unknown sampler: {request.param}")


@pytest.fixture()
def sampler_kwargs(flow_backend, sample_kwargs):
    if flow_backend == "zuko":
        fit_kwargs = dict(
            n_epochs=10,
        )
    elif flow_backend == "flowjax":
        fit_kwargs = dict(
            max_epochs=10,
        )
    else:
        raise ValueError(f"Unknown flow backend: {flow_backend}")
    return dict(n_samples=100, fit_kwargs=fit_kwargs, sample_kwargs=sample_kwargs)


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
    conversion_function,
):
    outdir = tmp_path / "test_run_sampler"
    outdir.mkdir(parents=True, exist_ok=True)

    sampler_kwargs.update(**existing_result)

    bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        sampler="aspire",
        outdir=outdir,
        flow_backend=flow_backend,
        conversion_function=conversion_function,
        **sampler_kwargs,
    )


def test_run_sampler_pool(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    sampler_kwargs,
    flow_backend,
):
    from multiprocessing.dummy import Pool

    outdir = tmp_path / "test_run_sampler_pool"
    outdir.mkdir(parents=True, exist_ok=True)

    with patch("multiprocessing.Pool", new=Pool):
        bilby.run_sampler(
            likelihood=bilby_likelihood,
            priors=bilby_priors,
            sampler="aspire",
            outdir=outdir,
            flow_backend=flow_backend,
            npool=2,
            **sampler_kwargs,
        )

import bilby
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def seed_bilby():
    import bilby

    bilby.core.utils.random.seed(42)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def model(x, m, c):
    # Assert d = |m| + |c| is satisfied and d is in the prior range
    d = abs(m) + abs(c)
    assert 0.5 <= d <= 5
    return m * x + c


def conversion_func(parameters, likelihood=None, priors=None):
    # d = |m| + |c|
    parameters["d"] = abs(parameters["m"]) + abs(parameters["c"])
    return parameters


@pytest.fixture()
def conversion_function():
    return conversion_func


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
    priors["d"] = bilby.core.prior.Constraint(name="d", minimum=0.5, maximum=5)
    return priors

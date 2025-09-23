import pytest
import numpy as np


@pytest.fixture(autouse=True)
def seed_bilby():
    import bilby

    bilby.core.utils.random.seed(42)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)

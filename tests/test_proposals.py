import bilby
import numpy as np
import pytest
from aspire.proposals import Proposal

from aspire_bilby.proposals import BilbyPriorProposal


def test_bilby_prior_proposal_preserves_order_and_constraints(bilby_priors):
    proposal = BilbyPriorProposal(bilby_priors, parameters=["c", "m"])

    x, log_prob = proposal.sample_and_log_prob(50)

    assert isinstance(proposal, Proposal)
    assert x.shape == (50, 2)
    assert log_prob.shape == (50,)
    derived = np.abs(x[:, 0]) + np.abs(x[:, 1])
    assert np.all((derived >= 0.5) & (derived <= 5.0))
    expected = bilby_priors.ln_prob(
        {"c": x[:, 0], "m": x[:, 1]},
        axis=0,
    )
    np.testing.assert_allclose(log_prob, expected)


def test_bilby_prior_proposal_handles_one_dimension():
    priors = bilby.core.prior.PriorDict({"x": bilby.core.prior.Uniform(-1.0, 1.0)})
    proposal = BilbyPriorProposal(priors, parameters=["x"])

    x, log_prob = proposal.sample_and_log_prob(1)

    assert x.shape == (1, 1)
    assert log_prob.shape == (1,)
    assert proposal.log_prob(np.array([0.0, 0.5])).shape == (2,)


def test_bilby_prior_proposal_validates_inputs():
    priors = bilby.core.prior.PriorDict({"x": bilby.core.prior.Uniform(-1.0, 1.0)})

    with pytest.raises(ValueError, match="At least one"):
        BilbyPriorProposal(priors, parameters=[])
    with pytest.raises(ValueError, match="missing"):
        BilbyPriorProposal(priors, parameters=["missing"])

    proposal = BilbyPriorProposal(priors, parameters=["x"])
    with pytest.raises(ValueError, match="n_samples"):
        proposal.sample_and_log_prob(0)
    with pytest.raises(ValueError, match="must have shape"):
        proposal.log_prob(np.zeros((2, 2)))

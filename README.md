# aspire-bilby

Interface between `aspire` and `bilby`.

## Installation

`aspire-bilby` is available via `pypi`:

```
pip install aspire-bilby
```

## Usage in bilby

`aspire` can be used just like any other sampler in `bilby` and supports
`multiprocessing` via the `n_pool` keyword argument.


```python
bilby.run_sampler(
    sampler="aspire",
    n_samples=1000,
    n_final_samples=None,  # Optional, final number of samples to produce
    sample_kwargs=dict(
        sampler="smc"
    ),
    fit_kwargs=dict(
        n_epochs=100,
    ),
    n_pool=4,
)
```

### Using a set of samples

```python

from aspire.samples import Samples

initial_samples = Samples(...)  # Define initial samples

bilby.run_sampler(
    sampler="aspire",
    initial_samples=initial_samples,
    ...
)
```

### Using a bilby result file

```python
bilby.run_sampler(
    sampler="aspire",
    initial_result_file="<path to bilby result file>"
    ...,
)
```

### Sampling from the prior

```python
bilby.run_sampler(
    sampler="aspire",
    n_initial_samples=5000,  # Number of samples to draw from the prior, defaults to 10,000 if not specified
    ...,
)
```


## Using bilby objects with aspire

`aspire-bilby` also provides functions for converting `bilby` likelihood and
prior objects into


```python
import bilby
from aspire import Aspire
from aspire_bilby.utils import samples_from_bilby_result, get_aspire_functions

likelihood = ...    # Define bilby likelihood
priors = ...        # Define bilby priors

result = bilby.core.utils.read_in_result(...)    # Read in bilby result

functions = get_aspire_functions(
    likelihood,
    priors,
    parameters=priors.non_fixed_keys,
)

initial_samples = samples_from_bilby_result(result)

aspire = Aspire(
    log_likelihood=functions.log_likelihood,
    log_prior=functions.log_prior,
    dims=len(initial_samples.parameters),
)

history = aspire.fit(initial_samples)
```

## Usage in bilby_pipe

`aspire` can be used with `bilby_pipe` as you would any other sampler:

```
sampler = "aspire"
sampler_kwargs = {
    "initial_result_file": "path_to_file",
    "sample_kwargs": {...},
    "fit_kwargs": {...},
}
```

If using transfer files, you may also need to add the initial result file to the `additional-transfer-paths`.

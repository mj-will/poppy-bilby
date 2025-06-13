# poppy-bilby

Interface between the `poppy` post-processing sampler and `bilby`

## Usage in bilby



## Using bilby objects with poppy

`poppy-bilby` also provides functions for converting `bilby` likelihood and
prior objects into


```
import bilby
from poppy import Poppy
from poppy_bilby.utils import samples_from_bilby_result, get_poppy_functions

likelihood = ...    # Define bilby likelihood
priors = ...        # Define bilby priors

result = bilby.core.utils.read_in_result(...)    # Read in bilby result

functions = get_poppy_functions(
    likelihood,
    priors,
    parameters=priors.non_fixed_keys,
)

initial_samples = samples_from_bilby_result(result)

poppy = Poppy(
    log_likelihood=functions.log_likelihood,
    log_prior=functions.log_prior,
    dims=len(initial_samples.parameters),
)

history = poppy.fit(initial_samples)
```
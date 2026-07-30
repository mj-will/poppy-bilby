Usage
=====

Run aspire as a bilby sampler
-----------------------------

``aspire`` can be used like any other sampler within ``bilby`` and supports
multiprocessing via the ``n_pool`` keyword argument. We most problems,
we recommend using SMC sampling (this requires installing ``minipcn``).

.. code-block:: python

    import bilby

    bilby.run_sampler(
        sampler="aspire",
        n_samples=1000,
        n_final_samples=None,  # Optional final number of samples
        sample_kwargs=dict(
            sampler="smc",
        ),
        fit_kwargs=dict(
            n_epochs=100,
        ),
        n_pool=4,
    )

Different ways to initialize aspire
-----------------------------------

In ``aspire-bilby``, you can initialize aspire sampling in multiple ways listed
below. Additional keyword arguments are omitted but should be specified in
``sample_kwargs`` and ``fit_kwargs``. See the example above and the ``aspire``
documentation for more details.


Starting from an existing bilby result file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can seed aspire using a bilby result file:

.. code-block:: python

    bilby.run_sampler(
        sampler="aspire",
        initial_result_file="<path to bilby result file>",
        sample_kwargs={...},
        fit_kwargs={...},
    )

Starting from precomputed samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from aspire.samples import Samples

    initial_samples = Samples(...)  # Define initial samples

    bilby.run_sampler(
        sampler="aspire",
        initial_samples=initial_samples,
        sample_kwargs={...},
        fit_kwargs={...},
    )

Using the prior as the proposal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Bilby prior can be used directly as Aspire's proposal distribution. This
avoids training a normalizing flow and does not require initial samples:

.. code-block:: python

    bilby.run_sampler(
        sampler="aspire",
        proposal="prior",
        n_samples=1000,
        sample_kwargs={
            "sampler": "importance",
        },
    )

Sampling and log-density evaluation are delegated to the complete Bilby
``PriorDict``, including its conversion function and constraints. With
importance sampling, using the prior as the proposal means the importance
weights are determined by the likelihood. This can be inefficient when the
posterior occupies only a small fraction of the prior volume.

Converting bilby objects for aspire
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities are provided to convert bilby likelihood/prior objects and results
into aspire-friendly functions and sample sets.

.. code-block:: python

    import bilby
    from aspire import Aspire
    from aspire_bilby.utils import samples_from_bilby_result, get_aspire_functions

    likelihood = ...    # bilby likelihood
    priors = ...        # bilby priors
    result = bilby.core.utils.read_in_result(...)

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


Checkpointing and Resuming
--------------------------

``aspire`` supports checkpointing during sampling once the flow is trained.
Checkpoints are stored in HDF5 files and can be used to resume sampling later.
In the ``billy`` integration, checkpoints are saved to
``<outdir>/<label>_aspire_checkpoint.h5`` by default if checkpointing is enabled
(via the ``enable_checkpointing`` keyword argument).

.. note::

    If the result file already exists and contains a checkpoint, sampling will
    resume from that checkpoint automatically. If you want to always start fresh,
    delete or rename the existing checkpoint file first.

When ``proposal="prior"``, the prior proposal is reconstructed from the live
Bilby ``PriorDict`` on resume. It is intentionally not serialized into the
Aspire checkpoint, ensuring custom prior conversion functions are preserved.

Usage in bilby_pipe
-------------------

``aspire`` can be used with ``bilby_pipe`` as you would any other sampler:

.. code-block:: ini

    sampler = "aspire"
    sampler_kwargs = {
        "initial_result_file": "path_to_file",
        "sample_kwargs": {...},
        "fit_kwargs": {...},
    }

If using transfer files, you may need to add the initial result file to
``additional-transfer-paths``.

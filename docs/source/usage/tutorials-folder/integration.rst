Data integration
================

Loading a model
---------------

.. warning::

    Currently, the model requires the data it was trained on for proper loading. **We are working on removing this in the next version.**

This is the recommended way to load a model:

.. code-block:: python

    import multiDGD
    import anndata as ad

    data = ad.read_h5ad("path/to/adata.h5ad")

    model = multiDGD.load(
        data=data,
        save_dir="path/to/save_dir",
        model_name="name-of-saved-model",
    )

Integrating new data
--------------------

In general, this is how new data can be integrated into an existing model loaded as shown above. It is important to note that the features (genes, peaks) have to match those the model was trained on.

.. code-block:: python

    # remember that this data needs to be prepared in the same way as the training data
    new_data = ad.read_h5ad("path/to/new_data.h5ad")

    model.predict_new(new_data, n_epochs=20, include_correction_error=True, indices_of_new_distribution=None, external=False)

This method is described in the `predict_new API section <https://multidgd.readthedocs.io/en/latest/api.html#multiDGD.DGD.predict_new>`_.
A detailed example is given in the following notebook:

.. button-link:: https://github.com/Center-for-Health-Data-Science/multiDGD/blob/main/tutorials/example_mapping_new_data.ipynb
    :color: info

    :octicon:`repo;2em` human bonemarrow example integration notebook

It also includes a demonstration of how to integrate a single modality.

If the data comes from a different dataset with different covariates, `indices_of_new_distribution` should be a list of indices of samples coming from new covariate classes (e.g. new batches). This currently only supports learning one new component.
`include_correction_error` determines whether the integration should take into account the covariates during inference. When integrating all new batches, it might be beneficial to set this to `False`.
We recommend to set `n_epochs` to a higher number (e.g. 100), especially for larger datasets.

Integrating a single modality / Predicting missing modalities
-------------------------------------------------------------

.. note::

    *See section above for tutorial.*

In order to integrate a single modality, the same method as above can be used. Just make sure that the number and order of features of the modality to be integrated matches the one the model was trained on.


Integrating novel covariates
----------------------------

.. note::

    *Documentation coming soon.*
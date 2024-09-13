Data integration
================

Loading a model
---------------

.. warning::

    Currently, the model requires the data it was trained on for proper loading. **We are working on removing this in the next version.**

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

.. code-block:: python

    # remember that this data needs to be prepared in the same way as the training data
    new_data = ad.read_h5ad("path/to/new_data.h5ad")

    model.predict_new(new_data)


Integrating a single modality / Predicting missing modalities
-------------------------------------------------------------

.. note::

    *Documentation coming soon.*


Integrating novel covariates
----------------------------

.. note::

    *Documentation coming soon.*
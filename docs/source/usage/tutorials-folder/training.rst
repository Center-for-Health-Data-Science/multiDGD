Model initialization and training
=================================

Initialization
--------------

.. code-block:: python

    import multiDGD

    save_dir = './models/'

    # initializing the model with default parameters
    model = multiDGD.DGD(
        data=data,
        save_dir=save_dir,
        model_name='model1',
    )

    model.view_data_setup()

.. note::

    This is an example of the default model initialization. You can find more information on the parameters in the :doc:`API reference <../../generated/multiDGD>`.

    ``view_data_setup()`` is a method that visualizes the data setup. This is useful to check if the data is correctly loaded and if the model is initialized correctly.
    It displays the data dimensions, number of celltypes, and the covariates chosen (if any).

Training
--------

.. code-block:: python

    model.train(
        n_epochs=500
    )

Save the model
--------------

.. code-block:: python

    model.save()
Data loading and preparation
============================

In this tutorial we walk through the steps to get your data ready for multiDGD.

.. tip::

    You can use multiDGD for multi-omics data and for data from just one modality.

    The multi-omics setting is currently only supported for 2 modalities (RNA and ATAC-seq). *Support for proteomics data is in the works.*

.. note::

    You can use multiDGD with both anndata and mudata objects. The preparation steps are slightly different for each. We will show you how to prepare your data for both types of objects.

Example data (anndata)
-----------------------

.. note::

    Running the following cell is only necessary to get the example data.

.. code-block:: python

    import requests, zipfile

    # Download
    file_name = 'human_bonemarrow.h5ad.zip'
    file_url = 'https://api.figshare.com/v2/articles/23796198/files/41740251'

    file_response = requests.get(file_url).json()
    file_download_url = file_response['download_url']
    response = requests.get(file_download_url, stream=True)
    with open(file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Unzip
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall('.')

.. note::
    
    Now we can start with the actual tutorial. You can modify the data directory ``data_dir`` to point to your own data.

.. code-block:: python

    import anndata as ad

    data_dir = './human_bonemarrow'

    data = ad.read_h5ad(data_dir+'.h5ad')


Anndata object preparation
--------------------------

.. note::

    Now that we have some data, we can get started with multiDGD.

.. code-block:: python

    import multiDGD

    data = multiDGD.functions.setup_data(
        data, 
        modality_key='feature_types', # adata.var column indicating which feature belongs to which modality
        observable_key='cell_type', # cell annotation key to initialize GMM components 
        covariate_keys=['Site'], # confounders
        train_fraction=0.9, # default, fraction of data to use for training
        include_test=False, # default is True, whether to make a test set in the train-val-test split
    )

.. note::

    You can read more about the details of this function in the :doc:`API reference <../../generated/multiDGD.functions>`. The most important arguments are:

    * ``modality_key``: When using anndata objects with multi-omics data, it is important to specify which variable column indicates the type of the modalities. This can be ignored for mudata objects.
    * ``observable_key``: This is the key in the cell annotation that indicates the cell type. This is used to initialize the GMM components. The model works better when having an estimate of the celltypes.
    * ``covariate_keys``: If you wish to remove batch effects or model other variables separately, you can specify them here with their column names in the anndata observables.


    * ``train_fraction``: multiDGD (like other ML models) needs a train and validation set (at least). Here you specify how much of the data will be used for training. The rest will be used for validation (and testing).
    * ``include_test``: Whether to include a test set in the train-val-test split. If set to False, the split will only contain a train and validation set.


Mudata object preparation
-------------------------

Let's look at how this would work for mudata objects.

.. tip::

    Working with mudata objects is a bit easier, as the data is already separating modalities.

.. code-block:: python

    import mudata as md

    data_dir = './data' # path to your data

    data = md.read(data_dir+'.h5mu', backed=False)

    data = multiDGD.functions.setup_data(
        data, 
        observable_key='...',
        covariate_keys=['...'],
        train_fraction=0.9,
        include_test=False,
    )

.. note::

    Here, we do not need a ``modality_key``.

Saving the prepared data
------------------------

.. tip::

    It is always a good idea to save the prepared data for the sake of reproducibility and so you can easily load it later.

.. code-block:: python

    import scanpy as sc

    data.write_h5ad('./example_data_prepared.h5ad')
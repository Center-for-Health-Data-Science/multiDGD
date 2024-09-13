Plotting
========

After training, you can plot the training history. This can help understand whether the model has converged or is overfitting:

.. code-block:: python

    model.plot_history()


Visualizing the representations (embeddings):

#. PCA of plain representations and the GMM (means and samples):

    .. code-block:: python

        model.plot_latent_space()

#. UMAP of representations using ``scanpy``:

    .. code-block:: python

        data.obsm['latent'] = model.get_representation()
        data.obs['cluster'] = model.clustering().astype(str)

        sc.pp.neighbors(data, use_rep='latent')
        sc.tl.umap(data, min_dist=1.0)
        sc.pl.umap(data, color='observable')
        sc.pl.umap(data, color='cluster')

#. Covariate representations (2D):

    .. code-block:: python

        cov_rep = model.get_covariate_representation()

        import seaborn as sns
        sns.scatterplot(x=cov_rep[:, 0], y=cov_rep[:, 1], hue=data.obs[data.obs["train_val_test"]=="train"]["Site"].values)


See this notebook for the example plots:

.. button-link:: https://github.com/Center-for-Health-Data-Science/multiDGD/blob/main/tutorials/example_adata_bonemarrow.ipynb

    :octicon:`repo;1em` human bonemarrow example notebook
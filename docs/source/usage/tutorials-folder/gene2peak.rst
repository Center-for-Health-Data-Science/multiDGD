Gene2Peak: *in silico* perturbations
====================================

This feature performs in silico perturbations on the specified gene and predicts the changes in prediction on all output features.

.. note::
    Right now the perturbations only consist of silencing of the given gene. Thus, a negative predicted change suggests a positive correlation.


First, we need to specify which gene and what data we want to look at.

.. code-block:: python

    # specify the gene we want to look at
    gene_name = "ID2"
    gene_location = "chr2:8678845-8684461"

    # and the samples we want to look at
    test_set = data[data.obs["train_val_test"] == "test",:].copy()

Now we can perform the *in silico* perturbations.

.. code-block:: python

    predicted_changes, samples_of_interest = model.gene2peak(
        gene_name=gene_name, testset=test_set
    )

    # in this example, RNA is the first modality and ATAC the second
    # we separate the predicted changes (delta) for each modality
    delta_gex = predicted_changes[0]
    delta_atac = predicted_changes[1]

You can also visualize the predicted changes with a celltype-sorted heatmap. For this, we recommend you check out the notebook:

.. button-link:: https://github.com/Center-for-Health-Data-Science/multiDGD/blob/main/tutorials/gene2peak.ipynb

    :octicon:`repo;1em` human bonemarrow example notebook (gene2peak)
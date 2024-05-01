# multiDGD

multiDGD is a new generative model for multi-omics data and provides the following functionalities:
- low-dimensional embedding
- data integration
- modality prediction
- gene2peak association (NEW :tada:)

Our base functionalities (embedding and data integration) come with some extreme upgrades in comparison to MultiVI thanks to our base method, the [Deep Generative Decoder](https://arxiv.org/abs/2110.06672) (which is in review for the plain transcriptomics application and available [here](https://github.com/Center-for-Health-Data-Science/scDGD)). The low-dimensional embedding is more structured and provides improved clustering. We are especially proud of having improved the data integration, by modelling covariates probabilistically, which enables users to integrate even data from unseen covariates without the need for architectural surgery.

In addition to a general improvement for data integration and clustering, we are proud to present gene2peak. This feature provides insight into associations between genes and peaks in single cells by performing in silico perturbations and accumulating cell-type specific changes in the transcriotion or chromatin landscape of multiome data.

## Installation

Since this is the alpha-version, the package should be installed by cloning the repository and installing it from source (from the project directory). Please note that the package requires Python version 3.8 or higher.

```
pip install .
```

## How to use it

Check out the notebooks showing examples of how to use multiDGD for anndata objects and mudata objects in the tutorials folder. There is also a preliminary tutorial on the gene2peak feature. This is still in development as a tool though and currently only supports a very specific use.

## We would love your feedback!

Feel free to create issues if you encounter problems or tell us what other functionalities you would like. This is the first version, so it might not all be super smooth yet.


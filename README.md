# multiDGD

multiDGD is a new generative model for multi-omics data and provides the following functionalities:
- low-dimensional embedding
- data integration
- modality prediction
- gene2peak association (NEW :tada:)

[Read the paper here.](https://www.nature.com/articles/s41467-024-53340-z)

Our base functionalities (embedding and data integration) come with some extreme upgrades in comparison to MultiVI thanks to our base method, the [Deep Generative Decoder](https://arxiv.org/abs/2110.06672) (which is in review for the plain transcriptomics application and available [here](https://github.com/Center-for-Health-Data-Science/scDGD)). The low-dimensional embedding is more structured and provides improved clustering. We are especially proud of having improved the data integration, by modelling covariates probabilistically, which enables users to integrate even data from unseen covariates without the need for architectural surgery.

In addition to a general improvement for data integration and clustering, we are proud to present gene2peak. This feature provides insight into associations between genes and peaks in single cells by performing in silico perturbations and accumulating cell-type specific changes in the transcriotion or chromatin landscape of multiome data.

## Documentation

Find out more and keep updated on our [documentation page](https://multidgd.readthedocs.io/en/latest/).

## Installation

Please note that the package requires Python version 3.8 or higher.

```
pip install multiDGD@git+https://github.com/Center-for-Health-Data-Science/multiDGD
```

*The model is compatible with scverse and will soon be installable via pip.*

## How to use it

Check out the notebooks showing examples of how to use multiDGD for anndata objects and mudata objects in the tutorials folder. There is also a preliminary tutorial on the gene2peak feature. This is still in development as a tool though and currently only supports a very specific use.

## We would love your feedback!

Feel free to create issues if you encounter problems or tell us what other functionalities you would like. This is the first version, so it might not all be super smooth yet.

## Citation

```
@article{schuster_multidgd_2024,
	title = {{multiDGD}: {A} versatile deep generative model for multi-omics data},
	volume = {15},
	issn = {2041-1723},
	url = {https://doi.org/10.1038/s41467-024-53340-z},
	doi = {10.1038/s41467-024-53340-z},
	number = {1},
	journal = {Nature Communications},
	author = {Schuster, Viktoria and Dann, Emma and Krogh, Anders and Teichmann, Sarah A.},
	month = nov,
	year = {2024},
	pages = {10031},
}
```


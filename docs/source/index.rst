.. multiDGD documentation master file, created by
   sphinx-quickstart on Wed Sep 11 12:48:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

multiDGD Documentation
====================================

multiDGD :cite:`Schuster2024` is a new generative model for multi-omics data and provides the following functionalities:

* low-dimensional embedding
* data integration
* modality prediction
* gene2peak association (NEW |:tada:|)

Our base functionalities (embedding and data integration) come with some extreme upgrades in comparison to MultiVI thanks to our base method, the `Deep Generative Decoder <https://academic.oup.com/bioinformatics/article/39/9/btad497/7241685>`_ :cite:`SchusterKrogh2023`. The low-dimensional embedding is more structured and provides improved clustering. We are especially proud of having improved the data integration, by modelling covariates probabilistically, which enables users to integrate even data from unseen covariates without the need for architectural surgery.

In addition to a general improvement for data integration and clustering, we are proud to present gene2peak. This feature provides insight into associations between genes and peaks in single cells by performing in silico perturbations and accumulating cell-type specific changes in the transcriotion or chromatin landscape of multiome data.

.. note::

   This package is still in beta-version. If you encounter any issues, please report them on our `GitHub <https://github.com/Center-for-Health-Data-Science/multiDGD/tree/main>`_ :octicon:`mark-github;1em`.

.. toctree::
   :hidden:
   :maxdepth: 2

   usage/installation
   usage/tutorials
   usage/intro
   api
   GitHub <https://github.com/Center-for-Health-Data-Science/multiDGD/tree/main>

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Installation :octicon:`plug;1em`
      :link: usage/installation.html

      Find out how to install multiDGD.
   
   .. grid-item-card:: Tutorials :octicon:`play;1em`
      :link: usage/tutorials.html

      The tutorials provide a brief guide to using multiDGD with examples.

   .. grid-item-card:: User Guide :octicon:`info;1em`
      :link: usage/intro.html

      The user guide provides detailed description on how multiDGD works and how you can tweak it to your needs. Here we connect the math to the code.

   .. grid-item-card:: API Reference :octicon:`book;1em`
      :link: api.html

      The API reference provides detailed information on the functions and classes in multiDGD.

.. rubric:: References

.. bibliography:: references.bib
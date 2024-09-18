Troubleshooting
===============

.. note::

    This page is under construction. We will add more guides as we go.

Here we list some options to troubleshoot common issues and low performance of multiDGD.

1. The count reconstruction performance is too low

* Increase the latent dimensionality: ``latent_dimension``
* Increase the depth of the decoder: ``n_hidden`` and ``n_hidden_modality``
* Increase the number of units in the hidden layers: ``n_units``

2. You observe overfitting (i.e. the model is likely too large for your dataset)

* Do the opposite of the previous point

3. There are too many / too few clusters

* Decrease / increase the number of components: ``n_components``
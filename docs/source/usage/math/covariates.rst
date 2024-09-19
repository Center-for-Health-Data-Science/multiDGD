Covariate model
===============

What we call the covariate model is an additional - importantly **supervised** - version of the GMM over a disentangled latent space. In the supervised scheme, the objective includes only probability densities of components assigned to a sample's label.

.. math::
    p(z_i\mid \phi, c_i) = \mathcal{N}_l(z_i\mid \mu_{k=c_i}, \Sigma_{k=c_i})
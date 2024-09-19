Gaussian mixture model
=======================

The Gaussian Mixture Model (GMM) presents the complex distribution over latent space in this model.

The GMM consists of a set of :math:`K` multivariate Gaussians with the same dimensionality as the corresponding representation. For the purpose of simplicity, we let multiDGD choose $K$ based on the number of unique annotated cell types. This parameter is of course flexible and allows for tailored latent spaces depending on the desired clustering resolution.

Trainable parameters include the means :math:`\mu` and covariances :math:`\Sigma` of the components and the mixture coefficients :math:`w`, which are transformed into mixture weights :math:`\pi`.

The means :math:`\mu` follow our *softball prior* similar to a mollified Uniform:

.. math::
    p(\mu) = \prod_k p_{\mathrm{Softball}}(\mu_k \mid \text{scale}, \text{sharpness})

Weights :math:`w` are described by a Dirichlet prior as is common in Bayesian Inference:

.. math::
    p(w) = \prod_k \mathrm{Dir}(\mu_k \mid \alpha)

Empirically, variances of Gaussian distributions follow the inverse Gamma distribution. As a result, the negative log covariance can be approximated by a Gaussian:

.. math::
    p(- \log \Sigma) = \prod_k \prod_l \mathcal{N}(- \log \Sigma_{k,l} \mid - \log 0.2 \times \frac{\text{scale}}{K}, 1)
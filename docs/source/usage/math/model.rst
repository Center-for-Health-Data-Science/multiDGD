Math
====

Notation
--------

+---------------------+--------------------------------------------+
| Symbol              | Representation                             |
+=====================+============================================+
| :math:`Z`           | representation                             |
+---------------------+--------------------------------------------+
| :math:`X`           | data                                       |
+---------------------+--------------------------------------------+
| :math:`\hat{X}`     | predicted/ reconstructed data              |
+---------------------+--------------------------------------------+
| mod                 | modality                                   |
+---------------------+--------------------------------------------+
| cov                 | covariate                                  |
+---------------------+--------------------------------------------+
| :math:`\theta`      | decoder parameters                         |
+---------------------+--------------------------------------------+
| :math:`\phi`        | GMM parameters                             |
+---------------------+--------------------------------------------+
| :math:`S`           | cell-specific scaling factor               |
+---------------------+--------------------------------------------+
| :math:`Y`           | decoder output (predicted normalized count)|
+---------------------+--------------------------------------------+
| :math:`i \in N`     | single sample :math:`i` among :math:`N`    |
|                     | total samples                              |
+---------------------+--------------------------------------------+
| :math:`k \in K`     | component :math:`k` among :math:`K`        |
|                     | components                                 |
+---------------------+--------------------------------------------+
| :math:`l`           | latent dimension                           |
+---------------------+--------------------------------------------+
| :math:`c \in C`     | class :math:`c` in :math:`C` covariate     |
|                     | classes                                    |
+---------------------+--------------------------------------------+
| :math:`\mu`         | GMM mean                                   |
+---------------------+--------------------------------------------+
| :math:`\Sigma`      | GMM covariance                             |
+---------------------+--------------------------------------------+
| :math:`w`           | component coefficient                      |
+---------------------+--------------------------------------------+
| :math:`\pi`         | component weight                           |
+---------------------+--------------------------------------------+
| :math:`\alpha`      | Dirichlet alpha                            |
+---------------------+--------------------------------------------+

Probabilistic formulation
-------------------------

.. note::

    The following section is (in most part) a direct excerpt from the multiDGD paper.

The training objective is given by the joint probability 

.. math::

    p(X,Z,\theta,\phi) = p(X\mid Z, \theta) \, p(Z\mid \phi)

which is maximized using Maximum a Posteriori estimation.

:math:`p(X\mid Z, \theta)` in this model is presented as the Negative Binomial distribution's mass of the observed count :math:`x_i` for cell :math:`i` given the predicted mean count and a learned dispersion parameter :math:`r_{j}` for each feature :math:`j`:

.. math::

    p( x_{i} \mid z_{i}, \theta , s_{i}) = \prod_{j=1}^D p(x_{ij}\mid z_{i},\theta,s_{i})

    \text{with } p(x_{ij}\mid z_{i},\theta,s_{i}) = \mathcal{NB}(x_{ij}\mid s_i y_{ij},r_j)

where :math:`\mathcal{NB}(x \mid y, r)` is the negative binomial distribution. Here we calculate the probability mass of the observed count :math:`x_{i,j}` given the negative binomial distribution with mean :math:`s_i y_{i,j}` and dispersion factor :math:`r_j`. The predicted mean :math:`s_i y_{i,j}` is given by the modality-specific total count :math:`s_i` of cell :math:`i` and the decoder output :math:`y_{i,j}`. This output :math:`y_{i,j}` describes the fraction of counts for cell :math:`i` and modality-specific feature :math:`j`, i.e. the predicted normalized count. These equations are valid for each modality (RNA and ATAC) separately, as we have a total count :math:`s` per modality.

The joint probability further contains the objective for the representation to follow the latent distribution, :math:`p(Z \mid \phi)`. Since :math:`\phi` is a GMM, this results in the weighted multivariate Gaussian probability density

.. math::

   p(z_i \mid \phi) = \sum_{k=1}^{K} \pi_k \mathcal{N}_L(z_i \mid \mu_k, \Sigma_k)

with :math:`K` as the number of GMM components and :math:`\mathcal{N}_L(z_i \mid \mu, \Sigma)` is a multivariate Gaussian distribution with dimension :math:`L` (the latent dimension), mean vector :math:`\mu` and covariance matrix :math:`\Sigma`.

For new data points, the representation is found by maximizing :math:`p(x_i \mid z_i, \theta, s) p(z_i \mid \phi)` only with respect to :math:`z_i`, as all other model parameters are fixed.

Modules
-------

.. toctree::
    :maxdepth: 2

    dgd
    reps
    gmm
    covariates
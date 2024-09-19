Model customization
===================

multiDGD hyperparameters
-------------------------

Hyperparameters for multiDGD are provided in the form of a dictionary. The package has a default set of hyperparameters, which can be overwritten by the user (by providing a dictionary to the model initialization, see below).

.. code-block:: python

    import multiDGD

    custom_parameters = {
        # custom hyperparameters
    }

    # initializing the model with default and custom parameters
    model = multiDGD.DGD(
        data=data,
        parameter_dictionary=custom_parameters,
    )

The following table lists the hyperparameters that can be customized, as well as their default values:

.. list-table::
   :header-rows: 1

   * - Argument
     - Type
     - Default
     - Description
   * - ``latent_dimension``
     - *int*
     - ``20``
     - Dimensionality of the latent space.
   * - ``n_components``
     - *int*
     - ``1``
     - Number of components in the mixture model.
   * - ``n_hidden``
     - *int*
     - ``2``
     - Number of hidden layers in the shared decoder (:math:`\theta_{h}`).
   * - ``n_hidden_modality``
     - *int*
     - ``3``
     - Number of hidden layers in the modality-specific decoder (:math:`\theta_{h_m}`).
   * - ``n_units``
     - *int*
     - ``100``
     - Number of units in the hidden layers (except the last layer, which is the maximum of :math:`\{100, \sqrt{|features|}\}`).
   * - ``value_init``
     - *str*
     - ``'zero'``
     - Initialization of the weights. Options are ``'zero'`` or an array of values.
   * - ``softball_scale``
     - *float*
     - ``2``
     - Scale parameter of the Softball prior (see manuscript). It determines the scale of the sphere of the (mollified uniform) prior over component means.
   * - ``softball_hardness``
     - *float*
     - ``5``
     - Hardness parameter of the Softball prior (see manuscript). It determines how *not* smooth the transition from probability 1 to 0 is.
   * - ``sd_sd``
     - *float*
     - ``1``
     - Standard deviation of the Gaussian prior over the negative log covariance. It is pretty irrelevant and can just stay at 1. The mean of this prior is determined by the number of components and the softball scale.
   * - ``softball_scale_corr``
     - *float*
     - ``2``
     - Same as ``softball_scale`` for the covariate models.
   * - ``softball_hardness_corr``
     - *float*
     - ``5``
     - Same as ``softball_hardness`` for the covariate models.
   * - ``sd_sd_corr``
     - *float*
     - ``1``
     - Same as ``sd_sd`` for the covariate models.
   * - ``dirichlet_a``
     - *float*
     - ``1``
     - Concentration parameter of the Dirichlet prior over the mixture weights. Higher values means stronger enforcement of equal probabilities.
   * - ``batch_size``
     - *int*
     - ``128``
     - Batch size for training.
   * - ``learning_rates``
     - *list*
     - ``[1e-4, 1e-2, 1e-2]``
     - Learning rates for the three sets of parameters: decoder, representation, GMM.
   * - ``betas``
     - *list*
     - ``[0.5,0.7]``
     - Betas for the Adam optimizer.
   * - ``weight_decay``
     - *float*
     - ``1e-4``
     - Weight decay for the Adam optimizer.
   * - ``decoder_width``
     - *int*
     - ``1``
     - Multiplies all hidden units by its factor (to bypass the last layer width rule).
   * - ``log_wandb``
     - *list*
     - ``['username', 'projectname']``
     - List of strings to log to wandb.
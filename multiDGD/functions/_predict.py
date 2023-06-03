import torch
import numpy as np
from multiDGD.latent import RepresentationLayer
from multiDGD.latent import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reshape_scaling_factor(x, o_dim):
    """
    the scaling factor needs to be of the same dimensionality as the predictions
    which can increase when searching for new representations.

    Given the desired dimensionality (i.e. that of the predictions), the
    scaling factors are extended. The first dimension keeps the number of features.
    """
    start_dim = len(x.shape)
    for _ in range(o_dim - start_dim):
        x = x.unsqueeze(1)
    return x


def prepare_potential_reps(sample_list):
    """
    takes a list of samples drawn from the DGD's distributions.
    The length gives the number of distributions which defines
    the dimensionality of the output tensor.
    If the list of samples is longer than 1, we will create representations
    from the combination of each GMM's samples.
    """
    if len(sample_list) > 1:
        if (
            len(sample_list) == 2
        ):  # currently only supporting 1-2 GMMs for simplicity in case we don't keep this approach
            samples_0 = sample_list[0].unsqueeze(1).expand(-1, sample_list[1].shape[0], -1)
            samples_1 = sample_list[1].unsqueeze(0).expand(sample_list[0].shape[0], -1, -1)
            return torch.cat([samples_0, samples_1], axis=-1).view(-1, samples_0.shape[-1] + samples_1.shape[-1])
    else:
        return sample_list[0]


def learn_new_representations(
    gmm,
    decoder,
    data_loader,
    n_samples_new,
    correction_model=None,
    n_epochs=10,
    lrs=[0.01, 0.01],
    resampling_type="mean",
    resampling_samples=1,
    include_correction_error=True,
    indices_of_new_distribution=None
):
    """
    this function creates samples from the trained GMM for each new point,
    returns the best representation for each sample,
    and trains the representations with all remaining parameters fixed

    gmm: the trained GMM
    decoder: the trained decoder
    data_loader: the data loader for the data to be predicted
    n_samples_new: the number of samples in the new data
    correction_model: the trained correction model (if available)
    n_epochs: the number of epochs to train the representations
    lrs: the learning rates for the representations
    resampling_type: the type of resampling to use for the GMM (can be mean or sample)
    resampling_samples: the number of samples to draw from the GMM for each new point (is ignored for resampling_type='mean')
    """

    # check if there are unknown distributions in the data
    #if gmm.n_mix_comp < len(data_loader.dataset.meta.unique()):
    #    print("WARNING: there are unknown distributions in the data\nWill learn extra components in the GMM")
    #    print(gmm.n_mix_comp, len(data_loader.dataset.meta.unique())) # apparently site1 had ha whole cell type
    correction_hook = False
    if correction_model is not None:
        if correction_model.n_mix_comp < data_loader.dataset.correction_classes:
            print("WARNING: there are unknown distributions in the data\nWill learn extra components in the batch GMM")
            n_correction_classes_old = correction_model.n_mix_comp
            correction_model = find_new_component(
                data_loader,
                decoder,
                correction_model,
                indices_of_new_distribution,
                other_gmm=gmm)
            correction_hook = True

    # make temporary representations with samples from each component per data point
    if correction_model is not None:
        potential_reps = prepare_potential_reps(
            [
                gmm.sample_new_points(resampling_type, resampling_samples),
                correction_model.sample_new_points(resampling_type, resampling_samples),
            ]
        )
    else:
        potential_reps = prepare_potential_reps(
            [gmm.sample_new_points(resampling_type, resampling_samples)]
        )

    print("making potential reps")
    print("   all potential reps: ", potential_reps.shape)

    decoder.eval()

    ############################
    # first match potential reps to samples
    ############################
    print("calculating losses for each new sample and potential reps")
    # creating a storage tensor into which the best reps are copied
    rep_init_values = torch.zeros((n_samples_new, potential_reps.shape[-1]))
    # compute predictions for all potential reps
    predictions = decoder(potential_reps.to(device))
    # go through data loader to calculate losses batch-wise
    for x, lib, i in data_loader:
        x = x.unsqueeze(1).to(device)
        lib = lib.to(device)
        if data_loader.dataset.modality_switch is not None:
            recon_loss_x = decoder.loss(
                [predictions[comp].unsqueeze(0) for comp in range(len(predictions))],
                [x[:, :, : data_loader.dataset.modality_switch], x[:, :, data_loader.dataset.modality_switch :]],
                scale=[reshape_scaling_factor(lib[:, xxx], 3) for xxx in range(decoder.n_out_groups)],
                reduction="sample",
            )
        best_fit_ids = torch.argmin(recon_loss_x, dim=-1).detach().cpu()
        rep_init_values[i, :] = potential_reps.clone()[best_fit_ids, :]

    ############################
    # create new initial representation
    ############################
    # create a new representation from the best components
    new_rep = RepresentationLayer(n_rep=gmm.dim, n_sample=n_samples_new, value_init=rep_init_values[:, : gmm.dim]).to(
        device
    )
    newrep_optimizer = torch.optim.Adam(new_rep.parameters(), lr=lrs[0], weight_decay=1e-4, betas=(0.5, 0.7))
    if correction_model is not None:
        test_correction_rep = RepresentationLayer(
            n_rep=2, n_sample=n_samples_new, value_init=rep_init_values[:, gmm.dim :]
        ).to(device)
        correction_rep_optim = torch.optim.Adam(
            test_correction_rep.parameters(), lr=lrs[1], weight_decay=1e-4, betas=(0.5, 0.7)
        )
        if correction_hook:
            correction_model_optim = torch.optim.Adam(
            correction_model.parameters(), lr=lrs[0], weight_decay=0, betas=(0.5, 0.7)
        )
    rep_init_values = None

    #####################
    # training reps (only)
    #####################
    print("training selected reps for ", n_epochs, " epochs")
    for epoch in range(n_epochs):
        newrep_optimizer.zero_grad()
        if correction_model is not None:
            correction_rep_optim.zero_grad()
        corr_loss = 0
        e_loss = 0
        for x, lib, i in data_loader:
            if correction_hook:
                correction_model_optim.zero_grad()
            x = x.to(device)
            lib = lib.to(device)
            if correction_model is not None:
                z = new_rep(i)
                z_correction = test_correction_rep(i)
                y = decoder(torch.cat((z, z_correction), dim=1))
            else:
                z = new_rep(i)
                y = decoder(z)
            # compute losses
            if data_loader.dataset.modality_switch is not None:
                recon_loss_x = decoder.loss(
                    y,
                    [x[:, : data_loader.dataset.modality_switch], x[:, data_loader.dataset.modality_switch :]],
                    scale=[lib[:, xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)],
                )
            else:
                recon_loss_x = decoder.loss(
                    y, [x], scale=[lib[:, xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)]
                )
            # gmm_error = gmm.forward_split(gmm,z).sum()
            gmm_error = gmm(z).sum()
            correction_error = torch.zeros(1).to(device)
            if correction_model is not None:
                correction_error += correction_model(z_correction).sum()
                corr_loss += correction_error.item()
            if include_correction_error:
                loss = recon_loss_x.clone() + gmm_error.clone() + correction_error.clone()
            else:
                loss = recon_loss_x.clone() + gmm_error.clone()
            loss.backward()
            if correction_hook:
                correction_model.mean.grad[:n_correction_classes_old,:] = 0
                correction_model.neglogvar.grad[:n_correction_classes_old,:] = 0
                correction_model.weight.grad[:n_correction_classes_old] = 0
                correction_model_optim.step()
            e_loss += loss.item()

        newrep_optimizer.step()
        if correction_model is not None:
            correction_rep_optim.step()
        e_loss /= (len(data_loader.dataset)*data_loader.dataset.n_features)
        print("epoch: ", epoch, " loss: ", e_loss)
    
    if correction_hook:
        return new_rep, test_correction_rep, correction_model
    else:
        return new_rep, test_correction_rep, None

def find_new_component(data_loader,
                       decoder,
                       gmm_model,
                       idx_of_new_distribution,
                       rep_type="correction",
                       other_gmm=None,
                       resampling_type="mean",
                       resampling_samples=1
                       ):
    ###
    # sample new GMM components and select the one that fits the unseen data best
    ###
    n_samples_newdist = len(idx_of_new_distribution)
    # create new GMM model
    n_classes_new = 20
    gmm_model_new = GaussianMixture(
        n_mix_comp=n_classes_new,
        dim=gmm_model.dim,
        mean_init=(gmm_model._mean_prior.radius,gmm_model._mean_prior.sharpness),
        sd_init=gmm_model._sd_init,
        weight_alpha=2).to(device)
    # make all potential representations
    if rep_type == "correction":
        potential_reps = prepare_potential_reps(
                [
                    other_gmm.sample_new_points(resampling_type, resampling_samples),
                    gmm_model_new.sample_new_points(resampling_type, resampling_samples),
                ]
            )
    else:
        if other_gmm is not None:
            potential_reps = prepare_potential_reps(
                    [
                        gmm_model_new.sample_new_points(resampling_type, resampling_samples),
                        other_gmm.sample_new_points(resampling_type, resampling_samples)
                    ]
                )
        else:
            potential_reps = gmm_model.sample_new_points(resampling_type, resampling_samples)
    # compute losses for all potential representations
    rep_init_values = torch.zeros((n_samples_newdist, potential_reps.shape[-1]))
    predictions = decoder(potential_reps.to(device))
    for x, lib, i in data_loader:
        keep = [i_elem for i_elem, loader_idx in enumerate(i.numpy()) if loader_idx in idx_of_new_distribution]
        i_newdist = [a for a,b in enumerate(idx_of_new_distribution) if b in i.numpy()]
        x = x[keep,:].unsqueeze(1).to(device)
        lib = lib[keep,:].to(device)
        if data_loader.dataset.modality_switch is not None:
            recon_loss_x = decoder.loss(
                [predictions[comp].unsqueeze(0) for comp in range(len(predictions))],
                [x[:, :, : data_loader.dataset.modality_switch], x[:, :, data_loader.dataset.modality_switch :]],
                scale=[reshape_scaling_factor(lib[:, xxx], 3) for xxx in range(decoder.n_out_groups)],
                reduction="sample",
            )
        best_fit_ids = torch.argmin(recon_loss_x, dim=-1).detach().cpu()
        rep_init_values[i_newdist, :] = potential_reps.clone()[best_fit_ids, :]
    # count how often each new component has been chosen
    best_component = [0,0]
    for c in range(gmm_model_new.n_mix_comp):
        n_times_chosen = len(np.unique(torch.where(rep_init_values[:,-2:] == gmm_model_new.mean.detach()[c,:])[0].numpy()))
        if n_times_chosen > best_component[1]:
            best_component = [c, n_times_chosen]
    best_new_mean = gmm_model_new.mean.detach().cpu()[best_component[0],:]
    best_new_nlv = gmm_model_new.neglogvar.detach().cpu()[best_component[0],:]
    best_new_weight = gmm_model_new.weight.detach().cpu()[best_component[0]]
    ###
    # take the best component and add it to the existing correction model
    ###
    n_classes_old = gmm_model.n_mix_comp
    n_classes_new = data_loader.dataset.correction_classes
    print("adding {} new correction classes".format(n_classes_new - n_classes_old))
    gmm_model_new = GaussianMixture(
        n_mix_comp=n_classes_new,
        dim=gmm_model.dim,
        mean_init=(gmm_model._mean_prior.radius,gmm_model._mean_prior.sharpness),
        sd_init=gmm_model._sd_init,
        weight_alpha=2).to(device)
    with torch.no_grad():
        gmm_model_new.mean[:n_classes_old,:] = gmm_model.mean.clone().detach()
        gmm_model_new.neglogvar[:n_classes_old,:] = gmm_model.neglogvar.clone().detach()
        gmm_model_new.weight[:n_classes_old] = gmm_model.weight.clone().detach()
        gmm_model_new.mean[-1,:] = best_new_mean
        gmm_model_new.neglogvar[-1,:] = best_new_nlv
        gmm_model_new.weight[-1] = best_new_weight
    print("old correction model: ", gmm_model, gmm_model.mean)
    print("new correction model: ", gmm_model_new)
    gmm_model = gmm_model_new

    return gmm_model

"""
def learn_new_representations(
    gmm, decoder, data_loader, rep_shape, correction_model, resampling_epoch=5, n_epochs=10, lrs=[0.01, 0.01]
):
    '''this function creates samples from the trained GMM for each sample,
    trains the representations with all remaining parameters fixed,
    and returns the best representation for each sample'''

    # make temporary representations with samples from each component per data point
    rep_init_values = (
        gmm.mean.clone()
        .cpu()
        .detach()
        .unsqueeze(1)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(rep_shape[0], -1, -1, correction_model.n_mix_comp, -1)
        .clone()
    )
    new_rep = RepresentationLayer(n_rep=rep_shape[1], n_sample=rep_shape[0], value_init=rep_init_values).to(device)
    newrep_optimizer = torch.optim.Adam(new_rep.parameters(), lr=lrs[0], weight_decay=1e-4, betas=(0.5, 0.7))

    if correction_model is not None:
        rep_init_values = (
            correction_model.mean.clone()
            .cpu()
            .detach()
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(rep_shape[0], -1, gmm.n_mix_comp, -1, -1)
            .clone()
        )
        test_correction_rep = RepresentationLayer(
            n_rep=2,
            n_sample=rep_shape[0],  # *correction_model.n_mix_comp,
            value_init=rep_init_values,  # .reshape(rep_shape[0]*correction_model.n_mix_comp,2)
        ).to(device)
        correction_rep_optim = torch.optim.Adam(
            test_correction_rep.parameters(), lr=lrs[1], weight_decay=1e-4, betas=(0.5, 0.7)
        )
    else:
        test_correction_rep = None
    rep_init_values = None
    torch.cuda.empty_cache()

    decoder.eval()

    newrep_optimizer.zero_grad()
    if correction_model is not None:
        correction_rep_optim.zero_grad()

    ############################
    # first train all new reps for a little bit
    ############################
    print("   training all potential reps")
    for resample_epoch in range(resampling_epoch):
        newrep_optimizer.zero_grad()
        for x, lib, i in data_loader:
            x = x.to(device)
            lib = lib.to(device)
            # x has shape (n,d)
            # z is chosen with i by intermediately viewing it as (n,m,c,l) and then shaping it to (n*m*c,l)
            if correction_model is not None:
                z = new_rep(i)  # .expand(-1,-1,-1,correction_model.n_mix_comp,-1)
                z_correction = test_correction_rep(i)  # .expand(-1,-1,gmm.n_mix_comp,-1,-1)
                out_nsample = i.shape[0] * z.shape[1] * gmm.n_mix_comp * correction_model.n_mix_comp
                y = decoder(torch.cat((z, z_correction), dim=-1).view(-1, z.shape[-1] + z_correction.shape[-1]))
            if data_loader.dataset.modality_switch is not None:
                recon_loss_x = decoder.loss(
                    [
                        y[comp].view(i.shape[0], z.shape[1], gmm.n_mix_comp, correction_model.n_mix_comp, -1)
                        for comp in range(len(y))
                    ],
                    [
                        x[:, : data_loader.dataset.modality_switch]
                        .unsqueeze(1)
                        .unsqueeze(1)
                        .unsqueeze(1)
                        .expand(-1, z.shape[1], gmm.n_mix_comp, correction_model.n_mix_comp, -1),
                        x[:, data_loader.dataset.modality_switch :]
                        .unsqueeze(1)
                        .unsqueeze(1)
                        .unsqueeze(1)
                        .expand(-1, z.shape[1], gmm.n_mix_comp, correction_model.n_mix_comp, -1),
                    ],
                    scale=[reshape_scaling_factor(lib[:, xxx], len(z.shape)) for xxx in range(decoder.n_out_groups)],
                    mask=data_loader.dataset.get_mask(i),
                )
            gmm_error = gmm(z.view(-1, rep_shape[1])).sum()
            correction_error = torch.zeros(1).to(device)
            if correction_model is not None:
                correction_error += correction_model(z_correction.view(-1, 2)).sum()
            loss = recon_loss_x.clone() + gmm_error.clone() + correction_error.clone()
            # print('      loss {}'.format(loss.item()))
            loss.backward()
        newrep_optimizer.step()
        if correction_model is not None:
            correction_rep_optim.step()  # leave this out and rather use same origin for all
    x, lib, z, y = None, None, None, None
    torch.cuda.empty_cache()

    ############################
    # now select best for each sample
    ############################
    print("   selecting best representations")
    rep_new_values = torch.empty(rep_shape)
    batch_new_values = torch.empty((rep_shape[0], 2))
    for x, lib, i in data_loader:
        x = x.to(device)
        lib = lib.to(device)
        # z is again chosen with i by intermediately viewing it as (n,m,c,l) and then shaping it to (n*m*c,l)
        if correction_model is not None:
            z = new_rep(i)  # .expand(-1,-1,-1,correction_model.n_mix_comp,-1)
            z_correction = test_correction_rep(i)  # .expand(-1,-1,gmm.n_mix_comp,-1,-1)
            out_nsample = i.shape[0] * z.shape[1] * gmm.n_mix_comp * correction_model.n_mix_comp
            y = decoder(torch.cat((z, z_correction), dim=-1).view(-1, z.shape[-1] + z_correction.shape[-1]))
        if data_loader.dataset.modality_switch is not None:
            recon_loss_x = decoder.loss(
                [
                    y[comp].view(i.shape[0], z.shape[1], gmm.n_mix_comp, correction_model.n_mix_comp, -1)
                    for comp in range(len(y))
                ],
                [
                    x[:, : data_loader.dataset.modality_switch]
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .expand(-1, z.shape[1], gmm.n_mix_comp, correction_model.n_mix_comp, -1),
                    x[:, data_loader.dataset.modality_switch :]
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .expand(-1, z.shape[1], gmm.n_mix_comp, correction_model.n_mix_comp, -1),
                ],
                scale=[reshape_scaling_factor(lib[:, xxx], len(z.shape)) for xxx in range(decoder.n_out_groups)],
                mask=data_loader.dataset.get_mask(i),
                reduction="sample",
            )
        gmm_error = gmm(z)
        # here, the recon loss has to be reshaped from (n,m,c) to (n*m*c) to match the gmm error
        if correction_model is None:
            loss = gmm.reshape_targets(recon_loss_x.clone(), y_type="reverse") + gmm_error.clone()
        else:
            loss = recon_loss_x.clone() + gmm_error.clone()
        # now we can easily get the best representations for i with just z and the loss
        best_sample = torch.argmin(loss.view(-1, gmm.n_mix_comp * correction_model.n_mix_comp), dim=1)
        rep_new_values[i] = (
            z.view(-1, gmm.n_mix_comp * correction_model.n_mix_comp, new_rep.n_rep)[range(i.shape[0]), best_sample]
            .detach()
            .cpu()
        )
        if correction_model is not None:
            batch_new_values[i] = (
                z_correction.view(-1, gmm.n_mix_comp * correction_model.n_mix_comp, test_correction_rep.n_rep)[
                    range(i.shape[0]), best_sample
                ]
                .detach()
                .cpu()
            )

    # create a new representation from the best components
    new_rep = RepresentationLayer(n_rep=rep_shape[1], n_sample=rep_shape[0], value_init=rep_new_values).to(device)
    newrep_optimizer = torch.optim.Adam(new_rep.parameters(), lr=lrs[0], weight_decay=1e-4, betas=(0.5, 0.7))
    rep_new_values = None
    if correction_model is not None:
        test_correction_rep = RepresentationLayer(
            n_rep=2,
            n_sample=rep_shape[0],  # *correction_model.n_mix_comp,
            value_init=batch_new_values,  # .reshape(rep_shape[0]*correction_model.n_mix_comp,2)
        ).to(device)
        correction_rep_optim = torch.optim.Adam(
            test_correction_rep.parameters(), lr=lrs[1], weight_decay=1e-4, betas=(0.5, 0.7)
        )

    #####################
    print("   further training selected reps")
    #####################
    for epoch in range(n_epochs):
        newrep_optimizer.zero_grad()
        if correction_model is not None:
            correction_rep_optim.zero_grad()
        corr_loss = 0

        for x, lib, i in data_loader:
            x = x.to(device)
            lib = lib.to(device)

            if correction_model is not None:
                z = new_rep(i)
                z_correction = test_correction_rep(i)
                y = decoder(torch.cat((z, z_correction), dim=1))
            else:
                z = new_rep(i)
                y = decoder(z)

            # compute losses
            if data_loader.dataset.modality_switch is not None:
                recon_loss_x = decoder.loss(
                    y,
                    [x[:, : data_loader.dataset.modality_switch], x[:, data_loader.dataset.modality_switch :]],
                    scale=[lib[:, xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)],
                    mask=data_loader.dataset.get_mask(i),
                )
            else:
                recon_loss_x = decoder.loss(
                    y,
                    [x],
                    scale=[lib[:, xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)],
                    mask=data_loader.dataset.get_mask(i),
                )
            # gmm_error = gmm.forward_split(gmm,z).sum()
            gmm_error = gmm(z).sum()
            correction_error = torch.zeros(1).to(device)
            if correction_model is not None:
                # correction_error += correction_model.forward_split(correction_model, z_correction).sum()
                correction_error += correction_model(z_correction).sum()
                corr_loss += correction_error.item()
            loss = recon_loss_x.clone() + gmm_error.clone() + correction_error.clone()
            loss.backward()
        # print(corr_loss)

        newrep_optimizer.step()
        if correction_model is not None:
            correction_rep_optim.step()

    return new_rep, test_correction_rep
"""

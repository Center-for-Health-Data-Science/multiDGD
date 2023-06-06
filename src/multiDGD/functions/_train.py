import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from multiDGD.latent import RepresentationLayer
from multiDGD.functions._metrics import clustering_metric, rep_triangle_loss
from sklearn.metrics import silhouette_score

#import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_l_regularization(mod, reg_type):
    w = 0
    for name, param in mod.named_parameters():
        if 'weight' in name:
            if reg_type == 'l1':
                w += param.abs().sum()
            elif reg_type == 'l2':
                w += param.square().sum()
    return w

def train_dgd(
    decoder,
    gmm,
    representation,
    validation_rep,
    train_loader,
    validation_loader,
    correction_gmm,
    correction_rep,
    correction_val_rep,
    n_epochs,
    learning_rates,
    adam_betas,
    wd,
    stop_method,
    stop_len,
    train_minimum,
    save_dir='./',
    developer_mode=False
    ):

    '''main train function'''

    print('training for ',n_epochs,' epochs with early stopping (',stop_method,')')

    ###
    # prepare optimizers and correction models (if applicable, e.g. batch correction)
    ###

    if developer_mode:
        import wandb

    # get number of samples
    n_samples = representation.n_sample
    # see if validation set is included
    if validation_loader is not None:
        validation = True
        n_samples_val = validation_rep.n_sample
    else: 
        validation = False
        n_samples_val = 0
    # get modality switch
    modality_switch = train_loader.dataset.modality_switch

    # create directory for saving files if not already existing
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # initialize optimizers
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rates[0], weight_decay=wd,betas=(adam_betas[0],adam_betas[1]))
    rep_optimizer = torch.optim.Adam(representation.parameters(), lr=learning_rates[1], weight_decay=wd,betas=(adam_betas[0],adam_betas[1]))
    gmm_optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rates[2], weight_decay=wd,betas=(adam_betas[0],adam_betas[1]))
    
    if validation:
        testrep_optimizer = torch.optim.Adam(validation_rep.parameters(), lr=learning_rates[1], weight_decay=wd,betas=(adam_betas[0],adam_betas[1]))
    
    # initialize correction representation and optimizers (if applicable)
    if correction_gmm is not None:
        correction_optim = torch.optim.Adam(correction_gmm.parameters(), lr=learning_rates[1]/10, weight_decay=wd,betas=(adam_betas[0],adam_betas[1]))
        correction_rep_optim = torch.optim.Adam(correction_rep.parameters(), lr=learning_rates[1], weight_decay=wd, betas=(adam_betas[0],adam_betas[1]))
        if validation:
            correction_val_rep_optim = torch.optim.Adam(correction_val_rep.parameters(), lr=learning_rates[1], weight_decay=wd, betas=(adam_betas[0],adam_betas[1]))

    # in case of mosaic data, add a representation for the 3 reps of the same cell
    if train_loader.dataset.mosaic_train_idx is not None:
        representation_mosaic = RepresentationLayer(n_rep=representation.n_rep,
                                                    n_sample=train_loader.dataset.data_triangle.shape[0],
                                                    value_init='zero').to(device)
        mosaic_optimizer = torch.optim.Adam(representation_mosaic.parameters(), lr=learning_rates[1], weight_decay=wd, betas=(adam_betas[0],adam_betas[1]))
    
    ###
    # define metrics to observe during training
    ###
    total_loss = []
    reconstruction_loss = []
    distribution_loss = []
    if validation:
        total_loss_val = []
        reconstruction_loss_val = []
        distribution_loss_val = []


    # early stopping
    early_stopping_queue = []
    early_stopping_metric = [] # has to be 1-ari for clustering

    ###
    # start training
    ###

    if not developer_mode:
        from tqdm import trange
        progress_bar = trange(n_epochs, desc="Training", unit="epochs")
    else:
        progress_bar = np.arange(n_epochs)

    for epoch in progress_bar:

        early_stopping_metric.append(0)
        total_loss.append(0)
        reconstruction_loss.append(0)
        distribution_loss.append(0)
        if developer_mode:
            if decoder.n_out_groups > 1:
                summed_gradient_ratios = [0]*decoder.n_out_groups

        # set model train state
        decoder.train()

        # set representation gradients to zero
        rep_optimizer.zero_grad()
        if correction_gmm is not None:
            correction_rep_optim.zero_grad()
        
        ###
        # train run
        ###

        ###
        # in the case of mosaic data, add a run where distances between reps for the same cell
        # are minimized
        if train_loader.dataset.mosaic_train_idx is not None:
            mosaic_batch_size = 64
            n_split_samples = int(train_loader.dataset.data_triangle.shape[0]/3)
            # set representation gradients to zero
            mosaic_optimizer.zero_grad()
            triangle_loss = 0
            for i in range(0, n_split_samples, mosaic_batch_size):
                decoder_optimizer.zero_grad()
                #gmm_optimizer.zero_grad()
                # get batch
                idx_paired = np.arange(i,min(i+mosaic_batch_size,n_split_samples))
                idx = np.concatenate((idx_paired, idx_paired+n_split_samples, idx_paired+2*n_split_samples))
                x = train_loader.dataset.data_triangle[idx,:].to(device)
                lib = train_loader.dataset.library[idx_paired,:].repeat(3,1).to(device)
                mask = [m[idx] for m in train_loader.dataset.modality_mask_triangle]
                # get representations
                z = representation_mosaic(idx)
                if correction_gmm is not None:
                    correction_optim.zero_grad()
                    z_correction = correction_rep(idx_paired).detach().clone().repeat(3,1)
                    y = decoder(torch.cat((z, z_correction), dim=1))
                else:
                    y = decoder(z)
                # compute losses
                recon_loss_x = decoder.loss(
                    y,
                    [x[:,:modality_switch], x[:,modality_switch:]],
                    scale=[lib[:,xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)],
                    mask=mask
                )
                gmm_error = gmm(z).sum()
                # add representation trianlge loss to mosaic loss
                mosaic_loss = rep_triangle_loss(z,idx_paired).sum()
                triangle_loss += mosaic_loss.item()/len(idx_paired) # for logging
                loss = recon_loss_x.clone() + gmm_error.clone() + mosaic_loss.clone()
                # backpropagate
                loss.backward()
                #gmm_optimizer.step()
                decoder_optimizer.step()
                # update parameters
            mosaic_optimizer.step()
        ###

        for x,lib,i in train_loader:
            x = x.to(device)
            lib = lib.to(device)

            # set optimizer gradients to zero
            decoder_optimizer.zero_grad()
            gmm_optimizer.zero_grad()

            # get representation and pass through model
            if correction_gmm is not None:
                correction_optim.zero_grad()
                z = representation(i)
                z_correction = correction_rep(i)
                #z_correction = torch.cat(tuple([correction_models[corr_id][1](i) for corr_id in range(len(correction_models))]), dim=1)
                y = decoder(torch.cat((z, z_correction), dim=1))
            else:
                z = representation(i)
                y = decoder(z)
            
            # compute losses
            if modality_switch is not None:
                recon_loss_x = decoder.loss(
                    y,
                    [x[:,:modality_switch], x[:,modality_switch:]],
                    scale=[lib[:,xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)],
                    mask=train_loader.dataset.get_mask(i)
                )
            else:
                recon_loss_x = decoder.loss(
                    y,
                    [x],
                    scale=[lib[:,xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)],
                    mask=train_loader.dataset.get_mask(i)
                )
            gmm_error = gmm(z).sum()
            correction_error = torch.zeros(1).to(device)
            if correction_gmm is not None:
                supervision_idx = train_loader.dataset.get_correction_labels(i)
                #correction_error += correction_gmm(z_correction[:,(corr_id*2):(corr_id*2+2)],supervision_idx).sum()
                correction_error += correction_gmm(z_correction,supervision_idx).sum()
            loss = recon_loss_x.clone() + gmm_error.clone() + correction_error.clone()
            # include l1 regularization
            loss += get_l_regularization(mod=decoder, reg_type='l1')

            # backward pass
            loss.backward()

            # store gradient metrics in developer mode
            if developer_mode:
                if decoder.n_out_groups > 1:
                    summed_gradients = []
                    gradient_norm = 0
                    for out_mod_id in range(decoder.n_out_groups):
                        grad_temp = decoder.out_modules[out_mod_id].fc[0].weight.grad.clone().detach().cpu().abs().sum(0)
                        summed_gradients.append(grad_temp.sum())
                        gradient_norm += grad_temp.sum()
                    gradient_norm = gradient_norm.sum()
                    for out_mod_id in range(decoder.n_out_groups):
                        summed_gradient_ratios[out_mod_id] = (summed_gradients[out_mod_id]/gradient_norm).item()
                    summed_gradients = None
                    gradient_norm = None
                

            # make updates
            gmm_optimizer.step()
            if correction_gmm is not None:
                correction_optim.step()
            decoder_optimizer.step()
            
            # store metrics
            if (stop_method == 'loss') and not validation:
                #early_stopping_metric[-1] += recon_loss_x.item()
                early_stopping_metric[-1] += loss.item()
            total_loss[-1] += loss.item()
            reconstruction_loss[-1] += recon_loss_x.item()
            distribution_loss[-1] += gmm_error.item()
        
        total_loss[-1] /= (train_loader.dataset.n_features*n_samples)
        reconstruction_loss[-1] /= (train_loader.dataset.n_features*n_samples)
        distribution_loss[-1] /= (train_loader.dataset.n_features*n_samples)
        
        # make updates on representation(s)
        rep_optimizer.step()
        if correction_gmm is not None:
            correction_rep_optim.step()
        
        if stop_method == 'clustering':
            early_stopping_metric[-1] += (1 - clustering_metric(representation, gmm, train_loader.dataset.get_labels()))
        #else:
        #    early_stopping_metric[-1] /= n_samples
        #total_loss[-1] /= n_samples
        #reconstruction_loss[-1] /= n_samples
        #distribution_loss[-1] /= n_samples
        
        if not developer_mode:
            progress_bar.set_postfix(loss=total_loss[-1], reconstruction_loss=reconstruction_loss[-1])
        
        ###
        # validation run (in case)
        ###
        if validation:

            total_loss_val.append(0)
            reconstruction_loss_val.append(0)
            distribution_loss_val.append(0)

            decoder.eval()
            testrep_optimizer.zero_grad()
            if correction_gmm is not None:
                correction_val_rep_optim.zero_grad()
            
            for x,lib,i in validation_loader:
                x = x.to(device)
                lib = lib.to(device)

                if correction_gmm is not None:
                    z = validation_rep(i)
                    #z_correction = torch.cat(tuple([correction_models[corr_id][2](i) for corr_id in range(len(correction_models))]), dim=1)
                    z_correction = correction_val_rep(i)
                    y = decoder(torch.cat((z, z_correction), dim=1))
                else:
                    z = validation_rep(i)
                    y = decoder(z)
                
                # compute losses
                if modality_switch is not None:
                    recon_loss_x = decoder.loss(y,[x[:,:modality_switch], x[:,modality_switch:]],scale=[lib[:,xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)])
                else:
                    recon_loss_x = decoder.loss(y,[x],scale=[lib[:,xxx].unsqueeze(1) for xxx in range(decoder.n_out_groups)])
                gmm_error = gmm(z).sum()
                correction_error = torch.zeros(1).to(device)
                if correction_gmm is not None:
                    #supervision_idx = validation_loader.dataset.get_correction_labels(corr_id,i)
                    #correction_error += correction_models[corr_id][0](z_correction[:,(corr_id*2):(corr_id*2+2)],supervision_idx).sum()
                    correction_error += correction_gmm(z_correction).sum()
                loss = recon_loss_x.clone() + gmm_error.clone() + correction_error.clone()

                # backward pass
                loss.backward()

                if stop_method == 'loss':
                    early_stopping_metric[-1] += recon_loss_x.item()
                    #early_stopping_metric[-1] += loss.item()
                total_loss_val[-1] += loss.item()
                reconstruction_loss_val[-1] += recon_loss_x.item()
                distribution_loss_val[-1] += gmm_error.item()
            
            testrep_optimizer.step()
            if correction_gmm is not None:
                correction_val_rep_optim.step()
            
            #if stop_method == 'loss':
            #    early_stopping_metric[-1] /= n_samples_val
            total_loss_val[-1] /= (train_loader.dataset.n_features*n_samples_val)
            reconstruction_loss_val[-1] /= (train_loader.dataset.n_features*n_samples_val)
            distribution_loss_val[-1] /= (train_loader.dataset.n_features*n_samples_val)
        
        if developer_mode:
            log_dict = {'total_loss_train': total_loss[-1]/(n_samples*train_loader.dataset.n_features),
                'reconstruction_loss_train': reconstruction_loss[-1]/(n_samples*train_loader.dataset.n_features),
                'gmm_loss_train': distribution_loss[-1]/(n_samples*gmm.n_mix_comp*gmm.dim)}
            if validation:
                log_dict['total_loss_validation'] = total_loss_val[-1]/(n_samples_val*train_loader.dataset.n_features)
                log_dict['reconstruction_loss_validation'] = reconstruction_loss_val[-1]/(n_samples_val*train_loader.dataset.n_features)
                log_dict['gmm_loss_validation'] = distribution_loss_val[-1]/(n_samples_val*gmm.n_mix_comp*gmm.dim)
            # add metrics depending on availability
            if decoder.n_out_groups > 1:
                for out_mod_id in range(decoder.n_out_groups):
                    log_dict['gradient_contribution_modality'+str(out_mod_id)] = summed_gradient_ratios[out_mod_id]#/n_samples
            if train_loader.dataset.meta is not None:
                log_dict['AdjustedRandIndex_fromMetaLabel'] = clustering_metric(representation, gmm, train_loader.dataset.get_labels())
            if train_loader.dataset.correction is not None:
                log_dict['correction_AdjustedSilhouetteWidth'] = silhouette_score(representation.z.detach().cpu(), train_loader.dataset.correction)
            if train_loader.dataset.mosaic_train_idx is not None:
                log_dict['triangle_rep_loss'] = triangle_loss

            wandb.log(log_dict)
        
        # early stopping
        if epoch >= train_minimum:
            if epoch >= (train_minimum+stop_len):
                if any(early_stopping_metric[-1] <= i for i in early_stopping_queue):
                    # continue
                    early_stopping_queue.pop(0)
                    early_stopping_queue.append(early_stopping_metric[-1])
                else:
                    # stop
                    print('finished training after ',epoch,' epochs according to early stopping based on ',stop_method,' for ',stop_len,' epochs')
                    print(early_stopping_queue)
                    break
            else:
                early_stopping_queue.append(early_stopping_metric[-1])
    
    history = pd.DataFrame({
        'epoch': np.arange(len(total_loss)),
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'distribution_loss': distribution_loss,
        'split': 'train'
    })
    if validation:
        temp_history = pd.DataFrame({
            'epoch': np.arange(len(total_loss_val)),
            'total_loss': total_loss_val,
            'reconstruction_loss': reconstruction_loss_val,
            'distribution_loss': distribution_loss_val,
            'split': 'validation'
        })
        history = pd.concat([history, temp_history])

    return decoder, gmm, representation, validation_rep, correction_gmm, correction_rep, correction_val_rep, history
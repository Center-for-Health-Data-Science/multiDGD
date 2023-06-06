import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kstest
#from multiDGD.dataset import omicsDataset
#from multiDGD.latent import RepresentationLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_gene_in_peaks(gene_name, ref):
    """check if a gene is in the list of closest genes to a peak"""
    a, b = False, False
    if gene_name in ref["closest_gene_ID"].values:
        a = True
        if gene_name in ref[ref["in_promoter"]]["closest_gene_ID"].values:
            b = True
    return a, b


def check_gene_in_connected_features(gene_name, connected_features, ref, type="peak"):
    """check if a gene is in the list of connected features"""
    threshold = len(ref)
    connected_features = list(filter(lambda num: num < threshold, connected_features))
    if type == "peak":
        if gene_name in ref["closest_gene_ID"].values[connected_features]:
            return True
        else:
            return False
    elif type == "gene":
        if gene_name in ref["gene_id"].values[connected_features]:
            return True
        else:
            return False


def get_last_output_modality_name(out_weight_param_names, modality_id):
    return list(filter(lambda x: "decoder.out_modules." + str(modality_id) + ".fc" in x, out_weight_param_names))[-1]


def get_modality_id_of_perturbation(mod, feature_id):
    modality_id = 0
    if mod.train_set.modality_switch is not None:
        #if not isinstance(mod.train_set.modality_switch, list):
        #    modality_switch = mod.train_set.modality_switch.tolist()
        #else:
        modality_switch = mod.train_set.modality_switch
        #modality_id = np.where(np.array(modality_switch) > feature_id)[0][0]
        if feature_id >= modality_switch:
            modality_id = 1
        else:
            modality_id = 0
    if modality_id > 0:
        feature_id = feature_id - mod.train_set.modality_switch
    return modality_id, feature_id


def predict_perturbations(model, rep, correction, dataset, feature_id, fold_change, perturbation_type="silencing",n_epochs=1):
    """predict the effect of upregulating or silencing a feature fold_change times"""
    # n_samples = dataset.data.shape[0]
    # batch_size = 128
    # define rep optimizer
    rep_optimizer = torch.optim.Adam(rep.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.5, 0.7))
    #rep_optimizer.zero_grad()
    # get loss for changed feature and update z once
    # for i in range(int(n_samples/batch_size)+1):
    #    print(round(i/(int(n_samples/batch_size))*100),'%')
    #    start = i*batch_size
    #    end = min((i+1)*batch_size,n_samples)
    #    indices = np.arange(start,end,1)
    x_perturbed = dataset.data.clone().to(device)
    if perturbation_type == "silencing":
        x_perturbed[:, feature_id] = 0
    elif perturbation_type == "upregulation":
        x_perturbed[:, feature_id] = x_perturbed[:, feature_id] * fold_change
    else:
        raise ValueError("perturbation_type must be either silencing or upregulation")
    for i in range(n_epochs):
        rep_optimizer.zero_grad()
        y = model.predict_from_representation(rep, correction)
        modality_id, feature_id = get_modality_id_of_perturbation(model, feature_id)
        if model.train_set.modality_switch is not None:
            recon_loss_x = model.decoder.loss(
                y[modality_id],
                [x_perturbed[:, : model.train_set.modality_switch], x_perturbed[:, model.train_set.modality_switch :]][
                    modality_id
                ],
                scale=dataset.library[:, modality_id].unsqueeze(1).to(device),
                mod_id=modality_id,
                gene_id=feature_id,
            )
        else:
            recon_loss_x = model.decoder.loss(y, x_perturbed, scale=dataset.library, gene_id=feature_id)
        loss = recon_loss_x
        loss.backward()
        rep_optimizer.step()
    # return new predictions
    preds = model.predict_from_representation(rep, correction)
    return [x.detach().cpu() for x in preds]


def compute_p_values(mod, predictions_perturbed, predictions_original, lib, ids=None, alpha=0.05):
    """compute p values for each feature in the dataset given the original predictions and the perturbed predictions"""
    p_vals = []
    rejected = []
    padj = []
    # print((predictions_original[0])*(lib[:,0].unsqueeze(1)).shape)
    for mod_id in range(mod.decoder.n_out_groups):
        if ids is not None:
            x_original = predictions_original[mod_id][ids,:].clone() * lib[ids, mod_id].unsqueeze(1)
            p_vals.append(
                mod.decoder.out_modules[mod_id]
                .p_vals(
                    x=x_original.to(device),
                    mhat=predictions_original[mod_id][ids,:].to(device),
                    M=lib[ids, mod_id].unsqueeze(1).to(device),
                    mhat_case=predictions_perturbed[mod_id][ids,:].to(device),
                )
                .detach()
                .cpu()
            )
        else:
            x_original = predictions_original[mod_id].clone() * lib[:, mod_id].unsqueeze(1)
            p_vals.append(
                mod.decoder.out_modules[mod_id]
                .p_vals(
                    x=x_original.to(device),
                    mhat=predictions_original[mod_id].to(device),
                    M=lib[:, mod_id].unsqueeze(1).to(device),
                    mhat_case=predictions_perturbed[mod_id].to(device),
                )
                .detach()
                .cpu()
            )
        rejected_temp, padj_temp = mod.decoder.out_modules[mod_id].fdr(p_vals[-1],alpha=alpha)
        rejected.append(rejected_temp)
        padj.append(padj_temp)
    return p_vals, rejected, padj

def compute_p_values_from_fold_changes(mod, predictions_original, predictions_perturbed, predictions_pertubed2, test='mwu', ids=None, alpha=0.05):
    """compute p values for each feature in the dataset given the original predictions and the perturbed predictions
    the test can be mwu for Mann-Whittney U test or ks for Kolmogorov-Smirnov test"""
    
    p_vals = []
    rejected = []
    padj = []

    n_mods = len(predictions_original)
    if ids is None:
        ids = np.arange(predictions_original[0].shape[0])

    # create lambda function that computes p values for each prediction column depending on the test
    for mod_id in range(n_mods):
        folds_1 = predictions_perturbed[mod_id][ids,:]/predictions_original[mod_id][ids,:]
        folds_2 = predictions_pertubed2[mod_id][ids,:]/predictions_original[mod_id][ids,:]
        if test == 'mwu':
            p = mannwhitneyu(folds_1,folds_2,axis=0)[1]
        else:
            raise ValueError('test must be mwu')
        p_vals.append(p)
        rejected_temp, padj_temp = mod.decoder.out_modules[mod_id].fdr(p_vals[-1],alpha=alpha)
        rejected.append(rejected_temp)
        padj.append(padj_temp)
    return p_vals, rejected, padj

def compute_p_values_from_predictions(mod, predictions_original, predictions_perturbed, test='mwu', ids=None, alpha=0.05):
    """compute p values for each feature in the dataset given the original predictions and the perturbed predictions
    the test can be mwu for Mann-Whittney U test or ks for Kolmogorov-Smirnov test"""
    
    p_vals = []
    rejected = []
    padj = []

    n_mods = len(predictions_original)
    if ids is None:
        ids = np.arange(predictions_original[0].shape[0])

    # create lambda function that computes p values for each prediction column depending on the test
    for mod_id in range(n_mods):
        if test == 'mwu':
            p = mannwhitneyu(predictions_original[mod_id][ids,:],predictions_perturbed[mod_id][ids,:],axis=0)[1]
        else:
            raise ValueError('test must be mwu')
        p_vals.append(p)
        rejected_temp, padj_temp = mod.decoder.out_modules[mod_id].fdr(p_vals[-1],alpha=alpha)
        rejected.append(rejected_temp)
        padj.append(padj_temp)
    return p_vals, rejected, padj

def feature_linkage(model, test_set, feature_id, rna_ref, peak_ref, fold_change=10, alpha=0.05):
    """general function for predicting associated features in the model (and data)"""
    # save a copy of the original representation
    # because the reps have to be updated for up- and downregulation
    reps_original = model.test_rep.z.detach().clone().cpu().numpy()
    # also get indices of samples where the value of the feature is not 0
    indices_of_interest = np.where(test_set.data[:, feature_id] != 0)[0]

    predictions_original = model.decoder_forward(rep_shape=model.test_rep.z.shape[0])
    predictions_original = [x.detach().cpu() for x in predictions_original]
    torch.cuda.empty_cache()

    # run upregulation (1 step) and save predictions
    predictions_upregulated = predict_perturbations(
        model,
        model.test_rep,
        model.correction_test_rep,
        test_set,
        feature_id,
        fold_change,
        perturbation_type="upregulation",
    )
    with torch.no_grad():
        model.test_rep.z[:,:] = torch.tensor(reps_original[:,:], device=device)
    p_val, rejected_up, padj = compute_p_values(
        model,
        predictions_upregulated,
        predictions_original,
        test_set.library,
        ids=indices_of_interest,
        alpha=alpha)
    # rejected_* is a list of boolean arrays, one for each modality
    # that tells us which features are rejected (i.e. associated with the feature of interest)
    connected_features_up = [np.where(x)[0] for x in rejected_up]
    predictions_upregulated, rejected_up, padj, p_val = None, None, None, None
    torch.cuda.empty_cache()

    # run silencing (1 step) and save predictions
    predictions_silenced = predict_perturbations(
        model,
        model.test_rep,
        model.correction_test_rep,
        test_set,
        feature_id,
        fold_change,
        perturbation_type="silencing",
    )
    with torch.no_grad():
        model.test_rep.z[:,:] = torch.tensor(reps_original[:,:], device=device)
    # get p values for each feature (for up and down)
    p_val, rejected_sil, padj = compute_p_values(
        model,
        predictions_silenced,
        predictions_original,
        test_set.library,
        ids=indices_of_interest,
        alpha=alpha)
    connected_features_down = [np.where(x)[0] for x in rejected_sil]
    predictions_silenced, rejected_sil, padj, p_val = None, None, None, None
    torch.cuda.empty_cache()

    # check whether the promoter peak is in the list of connected features
    # careful, there can be multiple promoter peaks
    #print(rna_ref["gene_id"].values[feature_id])
    a = check_gene_in_connected_features(
        rna_ref["gene_id"].values[feature_id], connected_features_up[1], peak_ref, type="peak"
    )
    b = check_gene_in_connected_features(
        rna_ref["gene_id"].values[feature_id], connected_features_down[1], peak_ref, type="peak"
    )
    promoter_peak_in_regulated = "not"
    if a and b:  # needs to go first because a or b returns True if both are True
        promoter_peak_in_regulated = "both"
    elif a or b:
        if a:
            promoter_peak_in_regulated = "up"
        else:
            promoter_peak_in_regulated = "down"
    a,b = False,False
    gene_in_regulated = "not"
    if feature_id in connected_features_up[0]:
        a = True
    if feature_id in connected_features_down[0]:
        b = True
    if a and b:
        gene_in_regulated = "both"
    elif a or b:
        if a:
            gene_in_regulated = "up"
        else:
            gene_in_regulated = "down"
    
    df_out = pd.DataFrame(
        {
            "feature_id": [feature_id],
            "n_upregulated_rna": [len(connected_features_up[0])],
            "n_upregulated_atac": [len(connected_features_up[1])],
            "n_silenced_rna": [len(connected_features_down[0])],
            "n_silenced_atac": [len(connected_features_down[1])],
            "gene_in_regulated": [gene_in_regulated],
            "promoter_peak_in_regulated": [promoter_peak_in_regulated],
            "n_testsamples_relevant": [len(indices_of_interest)],
        }
    )
    return df_out

def feature_linkage2(model, test_set, feature_id, rna_ref, peak_ref, fold_change=10, alpha=0.05, test_statistic="mwu"):
    """general function for predicting associated features in the model (and data)
    this version calculates p values based on the distribution differences
    of up- and downregulated fold changes"""
    # save a copy of the original representation
    # because the reps have to be updated for up- and downregulation
    reps_original = model.test_rep.z.detach().clone().cpu().numpy()
    # also get indices of samples where the value of the feature is not 0
    indices_of_interest = np.where(test_set.data[:, feature_id] != 0)[0]

    predictions_original = model.decoder_forward(rep_shape=model.test_rep.z.shape[0])
    predictions_original = [x.detach().cpu() for x in predictions_original]
    torch.cuda.empty_cache()

    # run upregulation (1 step) and save predictions
    predictions_upregulated = predict_perturbations(
        model,
        model.test_rep,
        model.correction_test_rep,
        test_set,
        feature_id,
        fold_change,
        perturbation_type="upregulation",
    )
    with torch.no_grad():
        model.test_rep.z[:,:] = torch.tensor(reps_original[:,:], device=device)
    p_val_up, rejected_up, padj = compute_p_values_from_predictions(
        model,
        predictions_original,
        predictions_upregulated,
        ids=indices_of_interest,
        alpha=alpha,test=test_statistic)
    # rejected_* is a list of boolean arrays, one for each modality
    # that tells us which features are rejected (i.e. associated with the feature of interest)
    connected_features_up = [np.where(x)[0] for x in rejected_up]
    #predictions_upregulated, rejected_up, padj, p_val = None, None, None, None
    torch.cuda.empty_cache()

    # run silencing (1 step) and save predictions
    predictions_silenced = predict_perturbations(
        model,
        model.test_rep,
        model.correction_test_rep,
        test_set,
        feature_id,
        fold_change,
        perturbation_type="silencing",
    )
    with torch.no_grad():
        model.test_rep.z[:,:] = torch.tensor(reps_original[:,:], device=device)
    # get p values for each feature (for up and down)
    p_val_sil, rejected_sil, padj = compute_p_values_from_predictions(
        model,
        predictions_original,
        predictions_silenced,
        ids=indices_of_interest,
        alpha=alpha,test=test_statistic)
    connected_features_down = [np.where(x)[0] for x in rejected_sil]
    #predictions_silenced, rejected_sil, padj, p_val = None, None, None, None
    torch.cuda.empty_cache()

    # check whether the promoter peak is in the list of connected features
    # careful, there can be multiple promoter peaks
    #print(rna_ref["gene_id"].values[feature_id])
    a = check_gene_in_connected_features(
        rna_ref["gene_id"].values[feature_id], connected_features_up[1], peak_ref, type="peak"
    )
    b = check_gene_in_connected_features(
        rna_ref["gene_id"].values[feature_id], connected_features_down[1], peak_ref, type="peak"
    )
    promoter_peak_in_regulated = "not"
    if a and b:  # needs to go first because a or b returns True if both are True
        promoter_peak_in_regulated = "both"
    elif a or b:
        if a:
            promoter_peak_in_regulated = "up"
        else:
            promoter_peak_in_regulated = "down"
    a,b = False,False
    gene_in_regulated = "not"
    if feature_id in connected_features_up[0]:
        a = True
    if feature_id in connected_features_down[0]:
        b = True
    if a and b:
        gene_in_regulated = "both"
    elif a or b:
        if a:
            gene_in_regulated = "up"
        else:
            gene_in_regulated = "down"
    
    df_out = pd.DataFrame(
        {
            "feature_id": [feature_id],
            "n_upregulated_rna": [len(connected_features_up[0])],
            "n_upregulated_atac": [len(connected_features_up[1])],
            "n_silenced_rna": [len(connected_features_down[0])],
            "n_silenced_atac": [len(connected_features_down[1])],
            "gene_in_regulated": [gene_in_regulated],
            "promoter_peak_in_regulated": [promoter_peak_in_regulated],
            "n_testsamples_relevant": [len(indices_of_interest)],
            "n_pvals_under_alpha_up_rna": [np.sum(p_val_up[0] < alpha)],
            "n_pvals_under_alpha_up_atac": [np.sum(p_val_up[1] < alpha)],
            "n_pvals_under_alpha_down_rna": [np.sum(p_val_sil[0] < alpha)],
            "n_pvals_under_alpha_down_atac": [np.sum(p_val_sil[1] < alpha)]
        }
    )

    p_val, rejected, padj = compute_p_values_from_fold_changes(
        model,
        predictions_original,
        predictions_upregulated,
        predictions_silenced,
        ids=indices_of_interest,
        alpha=alpha,test=test_statistic)
    connected_features = [np.where(x)[0] for x in rejected]
    peak_in_fold_changes = check_gene_in_connected_features(
        rna_ref["gene_id"].values[feature_id], connected_features[1], peak_ref, type="peak"
    )
    gene_in_fold_changes = feature_id in connected_features[0]
    df_out["n_pvals_under_alpha_from-foldchange_rna"] = [np.sum(p_val[0] < alpha)]
    df_out["n_pvals_under_alpha_from-foldchange_atac"] = [np.sum(p_val[1] < alpha)]
    df_out["n_connected_rna"] = [len(connected_features[0])]
    df_out["n_connected_atac"] = [len(connected_features[1])]
    df_out["promoter_peak_in_foldchanges"] = [peak_in_fold_changes]
    df_out["gene_in_foldchanges"] = [gene_in_fold_changes]

    return df_out

def perturbation_stats(model, test_set, feature_id, rna_ref, peak_ref, fold_change=10, already_trained=False, alpha=0.05, n_epochs=1):
    """function to understand the effect of a perturbation on the model"""
    reps_original = model.test_rep.z.detach().clone().cpu().numpy()
    
    # get ids for gene and peak of interest
    gene_id = [feature_id] # int
    #peak_ids = np.where(peak_ref["closest_gene_ID"].values == rna_ref["gene_id"].values[feature_id])[0] # array
    peak_ids = peak_ref[peak_ref["in_promoter"] & (peak_ref["closest_gene_ID"] == rna_ref["gene_id"].values[feature_id])]["idx"].values
    print(peak_ids)
    
    # get original predictions
    output_original = model.decoder_forward(rep_shape=model.test_rep.z.shape[0])
    predictions_original = [ # keep only the predictions for the feature of interest
        output_original[0].detach().cpu()[:,gene_id],
        output_original[1].detach().cpu()[:,peak_ids],
    ]
    output_original = None
    torch.cuda.empty_cache()
    # get original probability masses
    probmass_original = [
        model.decoder.out_modules[0].group_probs(
            test_set.data[:,gene_id],predictions_original[0],
            test_set.library[:,0].unsqueeze(1),
            return_style='all',feature_ids=gene_id),
        model.decoder.out_modules[1].group_probs(
            test_set.data[:,peak_ids+test_set.modality_switch],predictions_original[1],
            test_set.library[:,1].unsqueeze(1),
            return_style='all',feature_ids=peak_ids)
    ]
    torch.cuda.empty_cache()
    # get scaled original predictions
    predictions_original_scaled = [
        predictions_original[0] * test_set.library[:,0].unsqueeze(1),
        predictions_original[1] * test_set.library[:,1].unsqueeze(1)
    ]

    # run upregulation (1 step) and save predictions
    output_upregulated = predict_perturbations(
        model,
        model.test_rep,
        model.correction_test_rep,
        test_set,
        feature_id,
        fold_change,
        perturbation_type="upregulation",
        n_epochs=n_epochs
    )
    with torch.no_grad():
        model.test_rep.z[:,:] = torch.tensor(reps_original[:,:], device=device)
    predictions_upregulated = [
        output_upregulated[0].detach().cpu()[:,gene_id],
        output_upregulated[1].detach().cpu()[:,peak_ids],
    ]
    output_upregulated = None
    torch.cuda.empty_cache()
    probmass_upregulated = [
        model.decoder.out_modules[0].group_probs(
            predictions_original[0]*test_set.library[:,0].unsqueeze(1),predictions_upregulated[0],
            test_set.library[:,0].unsqueeze(1),
            return_style='all',feature_ids=gene_id),
        model.decoder.out_modules[1].group_probs(
            predictions_original[1]*test_set.library[:,1].unsqueeze(1),predictions_upregulated[1],
            test_set.library[:,1].unsqueeze(1),
            return_style='all',feature_ids=peak_ids)
    ]
    torch.cuda.empty_cache()
    probmass_upregulated_wrtX = [
        model.decoder.out_modules[0].group_probs(
            test_set.data[:,gene_id],predictions_upregulated[0],
            test_set.library[:,0].unsqueeze(1),
            return_style='all',feature_ids=gene_id),
        model.decoder.out_modules[1].group_probs(
            test_set.data[:,peak_ids+test_set.modality_switch],predictions_upregulated[1],
            test_set.library[:,1].unsqueeze(1),
            return_style='all',feature_ids=peak_ids)
    ]
    torch.cuda.empty_cache()
    predictions_upregulated_scaled = [
        predictions_upregulated[0] * test_set.library[:,0].unsqueeze(1),
        predictions_upregulated[1] * test_set.library[:,1].unsqueeze(1)
    ]
    # same for down-regulation
    output_silenced = predict_perturbations(
        model,
        model.test_rep,
        model.correction_test_rep,
        test_set,
        feature_id,
        fold_change,
        perturbation_type="silencing",
        n_epochs=n_epochs
    )
    with torch.no_grad():
        model.test_rep.z[:,:] = torch.tensor(reps_original[:,:], device=device)
    predictions_silenced = [
        output_silenced[0].detach().cpu()[:,gene_id],
        output_silenced[1].detach().cpu()[:,peak_ids],
    ]
    output_silenced = None
    torch.cuda.empty_cache()
    probmass_silenced = [
        model.decoder.out_modules[0].group_probs(
            predictions_original[0]*test_set.library[:,0].unsqueeze(1),predictions_silenced[0],
            test_set.library[:,0].unsqueeze(1),
            return_style='all',feature_ids=gene_id),
        model.decoder.out_modules[1].group_probs(
            predictions_original[1]*test_set.library[:,1].unsqueeze(1),predictions_silenced[1],
            test_set.library[:,1].unsqueeze(1),
            return_style='all',feature_ids=peak_ids) 
    ]
    torch.cuda.empty_cache()
    probmass_silenced_wrtX = [
        model.decoder.out_modules[0].group_probs(
            test_set.data[:,gene_id],predictions_silenced[0],
            test_set.library[:,0].unsqueeze(1),
            return_style='all',feature_ids=gene_id),
        model.decoder.out_modules[1].group_probs(
            test_set.data[:,peak_ids+test_set.modality_switch],predictions_silenced[1],
            test_set.library[:,1].unsqueeze(1),
            return_style='all',feature_ids=peak_ids) 
    ]
    torch.cuda.empty_cache()
    predictions_silenced_scaled = [
        predictions_silenced[0] * test_set.library[:,0].unsqueeze(1),
        predictions_silenced[1] * test_set.library[:,1].unsqueeze(1)
    ]

    # get dispersion parameter
    gene_dispersion = model.decoder.out_modules[0].distribution.dispersion[:,gene_id[0]].detach().item()
    peak_dispersions = model.decoder.out_modules[1].distribution.dispersion[:,peak_ids].detach().flatten().numpy()
    
    df_out = pd.DataFrame(
        {
            "target_gene_value": test_set.data[:,gene_id[0]].detach().numpy(),
            "original_prediction_gene": predictions_original[0].detach().flatten().numpy(),
            "original_prediction_gene_scaled": predictions_original_scaled[0].detach().flatten().numpy(),
            "upregulated_prediction_gene": predictions_upregulated[0].detach().flatten().numpy(),
            "upregulated_prediction_gene_scaled": predictions_upregulated_scaled[0].detach().flatten().numpy(),
            "silenced_prediction_gene": predictions_silenced[0].detach().flatten().numpy(),
            "silenced_prediction_gene_scaled": predictions_silenced_scaled[0].detach().flatten().numpy(),
            "original_probmass_gene": probmass_original[0].detach().flatten().numpy(),
            "upregulated_probmass_gene": probmass_upregulated[0].detach().flatten().numpy(),
            "upregulated_probmass_gene_wrtX": probmass_upregulated_wrtX[0].detach().flatten().numpy(),
            "silenced_probmass_gene": probmass_silenced[0].detach().flatten().numpy(),
            "silenced_probmass_gene_wrtX": probmass_silenced_wrtX[0].detach().flatten().numpy()
        }
    )
    df_out["gene_dispersion"] = gene_dispersion
    for i,peak_id in enumerate(peak_ids):
        df_out["target_peak_value_{}".format(peak_id)] = test_set.data[:,peak_id+test_set.modality_switch].detach().numpy()
        df_out["original_prediction_peak_{}".format(peak_id)] = predictions_original[1][:,i].detach().flatten().numpy()
        df_out["original_prediction_peak_scaled_{}".format(peak_id)] = predictions_original_scaled[1][:,i].detach().flatten().numpy()
        df_out["upregulated_prediction_peak_{}".format(peak_id)] = predictions_upregulated[1][:,i].detach().flatten().numpy()
        df_out["upregulated_prediction_peak_scaled_{}".format(peak_id)] = predictions_upregulated_scaled[1][:,i].detach().flatten().numpy()
        df_out["silenced_prediction_peak_{}".format(peak_id)] = predictions_silenced[1][:,i].detach().flatten().numpy()
        df_out["silenced_prediction_peak_scaled_{}".format(peak_id)] = predictions_silenced_scaled[1][:,i].detach().flatten().numpy()
        df_out["original_probmass_peak_{}".format(peak_id)] = probmass_original[1][:,i].detach().flatten().numpy()
        df_out["upregulated_probmass_peak_{}".format(peak_id)] = probmass_upregulated[1][:,i].detach().flatten().numpy()
        df_out["upregulated_probmass_peak_wrtX{}".format(peak_id)] = probmass_upregulated_wrtX[1][:,i].detach().flatten().numpy()
        df_out["silenced_probmass_peak_{}".format(peak_id)] = probmass_silenced[1][:,i].detach().flatten().numpy()
        df_out["silenced_probmass_peak_wrtX{}".format(peak_id)] = probmass_silenced_wrtX[1][:,i].detach().flatten().numpy()
        df_out["peak_dispersion_{}".format(peak_id)] = peak_dispersions[i].item()
    return df_out

def feature_linkage3(model, test_set, feature_id, rna_ref, peak_ref, fold_change=10, alpha=0.05):
    """general function for predicting associated features in the model (and data)"""
    # save a copy of the original representation
    # because the reps have to be updated for up- and downregulation
    reps_original = model.test_rep.z.detach().clone().cpu().numpy()
    # also get indices of samples where the value of the feature is not 0
    #indices_of_interest = np.where(test_set.data[:, feature_id] != 0)[0]

    predictions_original = model.decoder_forward(rep_shape=model.test_rep.z.shape[0])
    predictions_original = [x.detach().cpu() for x in predictions_original]
    torch.cuda.empty_cache()

    # run upregulation (1 step) and save predictions
    modality_id, feature_id_in_mod = get_modality_id_of_perturbation(model, feature_id)
    predictions_upregulated = predict_perturbations2(
        model,
        model.test_rep,
        model.correction_test_rep,
        test_set,
        predictions_original,
        modality_id,
        feature_id_in_mod,
        fold_change,
    )
    with torch.no_grad():
        model.test_rep.z[:,:] = torch.tensor(reps_original[:,:], device=device)
    p_val, rejected_up, padj = compute_p_values_from_correlations(
        model,
        predictions_upregulated,
        predictions_original,
        feature_id=feature_id,
        alpha=alpha)
    # rejected_* is a list of boolean arrays, one for each modality
    # that tells us which features are rejected (i.e. associated with the feature of interest)
    connected_features_up = [np.where(x)[0] for x in rejected_up]
    torch.cuda.empty_cache()

    # check whether the promoter peak is in the list of connected features
    # careful, there can be multiple promoter peaks
    if modality_id == 0:
        promoter_peak_in_regulated = check_gene_in_connected_features(
            rna_ref["gene_id"].values[feature_id], connected_features_up[1], peak_ref, type="peak"
        )
        gene_in_regulated = feature_id in connected_features_up[0]
    else:
        promoter_peak_in_regulated = feature_id in connected_features_up[1]
        gene_id = peak_ref["closest_gene_ID"].values[feature_id_in_mod]
        gene_in_regulated = gene_id in rna_ref["gene_id"].values[connected_features_up[0]]
    
    df_out = pd.DataFrame(
        {
            "feature_id": [feature_id],
            "n_upregulated_rna": [len(connected_features_up[0])],
            "n_upregulated_atac": [len(connected_features_up[1])],
            "gene_in_regulated": [gene_in_regulated],
            "promoter_peak_in_regulated": [promoter_peak_in_regulated],
            #"n_testsamples_relevant": [len(indices_of_interest)],
        }
    )
    return df_out

def predict_perturbations2(model, rep, correction, dataset, predictions, modality_id, feature_id, fold_change,n_epochs=1):
    """predict the effect of upregulating or silencing a feature fold_change times"""
    # n_samples = dataset.data.shape[0]
    # batch_size = 128
    # define rep optimizer
    rep_optimizer = torch.optim.Adam(rep.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.5, 0.7))
    x_perturbed = dataset.data.clone().to(device)
    #x_perturbed[:, feature_id] = predictions[modality_id][:, feature_id] * fold_change
    x_perturbed[:, feature_id] = x_perturbed[:, feature_id] * fold_change
    for i in range(n_epochs):
        rep_optimizer.zero_grad()
        y = model.predict_from_representation(rep, correction)
        if model.train_set.modality_switch is not None:
            recon_loss_x = model.decoder.loss(
                y[modality_id],
                [x_perturbed[:, : model.train_set.modality_switch], x_perturbed[:, model.train_set.modality_switch :]][
                    modality_id
                ],
                scale=dataset.library[:, modality_id].unsqueeze(1).to(device),
                mod_id=modality_id,
                gene_id=feature_id,
            )
        else:
            recon_loss_x = model.decoder.loss(y, x_perturbed, scale=dataset.library, gene_id=feature_id)
        loss = recon_loss_x
        loss.backward()
        rep_optimizer.step()
    # return new predictions
    preds = model.predict_from_representation(rep, correction)
    return [x.detach().cpu() for x in preds]

from scipy.spatial.distance import cdist
def compute_p_values_from_correlations(mod, predictions_perturbed, predictions_original, feature_id, alpha=0.05):
    """compute p values from correlation coefficients of fold changes"""
    
    # get fold changes
    modality_switch = predictions_original[0].shape[1]
    modality_id, feature_id = get_modality_id_of_perturbation(mod, feature_id)
    fold_changes_ref = (predictions_perturbed[modality_id][:, feature_id].detach().cpu() / predictions_original[modality_id][:, feature_id].detach().cpu()).unsqueeze(1)
    fold_changes = [predictions_perturbed[0].detach().cpu() / predictions_original[0].detach().cpu(),
                    predictions_perturbed[1].detach().cpu() / predictions_original[1].detach().cpu()]
    # compute all against all correlation coefficients
    #correlations_of_feature = torch.zeros((fold_changes.shape[1]))
    # try to get one-vs-many correlations always
    correlations_of_feature = [
        torch.tensor(1 - cdist(fold_changes_ref.T, fold_changes[0].T, metric='correlation')[0]),
        torch.tensor(1 - cdist(fold_changes_ref.T, fold_changes[1].T, metric='correlation')[0])
    ]
    
    p_vals = []
    p_vals_unadjusted = [torch.zeros(fold_changes[0].shape[1]),
                         torch.zeros(fold_changes[1].shape[1])]
    norms = [torch.zeros(fold_changes[0].shape[1]),
            torch.zeros(fold_changes[1].shape[1])]
    
    for mod_id in range(len(predictions_original)):
        if fold_changes[mod_id].shape[1] > 20000:
            # use the quick and dirty version
            batch_size = 1000
            for i in range(0, fold_changes[mod_id].shape[1], batch_size):
                temp_corrs_higher = torch.where(
                        torch.abs(correlations_of_feature[mod_id]).unsqueeze(1)[i:i+batch_size,:] > torch.abs(correlations_of_feature[mod_id]),
                        torch.abs(correlations_of_feature[mod_id].clone()).unsqueeze(1)[i:i+batch_size,:], 0)
                p_vals_unadjusted[mod_id] += temp_corrs_higher.sum(0)
            p_vals.append(p_vals_unadjusted[mod_id] / torch.abs(correlations_of_feature[mod_id]).sum(0))
        else:
            batch_size = 1000
            for i in range(0, fold_changes[mod_id].shape[1], batch_size):
                temp_corrs = torch.tensor(1 - cdist(fold_changes[mod_id][:,i:i+batch_size].T, fold_changes[mod_id].T, metric='correlation'))
                norms[mod_id][i:i+batch_size] = torch.abs(temp_corrs).sum(1) - 1
                temp_corrs_higher = torch.where(torch.abs(temp_corrs) > torch.abs(correlations_of_feature[mod_id]), torch.abs(temp_corrs.clone()), 0)
                p_vals_unadjusted[mod_id] += temp_corrs_higher.sum(0)
            p_vals.append(p_vals_unadjusted[mod_id] / norms[mod_id])
    
    #correlations = torch.zeros((fold_changes.shape[1], fold_changes.shape[1]))
    #for i in range(0, correlations.shape[0], batch_size):
    #    for j in range(i, correlations.shape[0], batch_size):
    #        temp_size = min(batch_size, correlations.shape[0] - i)
    #        if i == j:
    #            correlations[i:i+batch_size, j:j+batch_size] = torch.corrcoef(fold_changes[:,i:i+batch_size].T)
    #        else:
    #            temp_corrs = torch.tensor(np.corrcoef(fold_changes[:,i:i+batch_size].T, fold_changes[:,j:j+batch_size].T))
    #            correlations[i:i+batch_size, j:j+batch_size] = temp_corrs[:temp_size,temp_size:]
    #            correlations[j:j+batch_size, i:i+batch_size] = temp_corrs[temp_size:,:temp_size]
    # save the correlation coefficients with the feature that was perturbed
    #correlations_of_feature = correlations[feature_id,:].clone()
    # now keep only those correlation coefficients that are larger than the one of interest per row
    #for i in range(0, fold_changes.shape[0], batch_size):
    #    correlations[i:i+batch_size,:] = torch.where(torch.abs(correlations[i:i+batch_size,:]) > torch.abs(correlations_of_feature), torch.abs(correlations[i:i+batch_size,:].clone()), 0)
    # from there calculate p values
    #p_vals_unadjusted = correlations.sum(0) - 1 # the 1 is substracted so I don't have to remove the diagonal
    #p_vals = p_vals_unadjusted / (torch.abs(correlations).sum(0)-1)
    
    #rejected = [x < alpha for x in p_vals.numpy()]
    rejected = [[x < alpha for x in p_vals[0].numpy()],
                [x < alpha for x in p_vals[1].numpy()]]
    return p_vals, rejected, None
import torch
import numpy as np

from multiDGD.latent import RepresentationLayer
from multiDGD.data import omicsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gene_ids(gene_name):
    gene_id = rna_ref_df[rna_ref_df["gene_name"] == gene_name]["gene_id"].values[0]
    gene_idx = np.where(rna_ref_df["gene_name"] == gene_name)[0][0]
    return gene_id, gene_idx

def find_closest_peak(gene_location):
    chrom = int(gene_location.split(":")[0].split("chr")[1])
    start = gene_location.split(":")[1].split("-")[0]
    end = gene_location.split(":")[1].split("-")[1]
    sub_df = ref_df[ref_df["Chromosome"] == chrom]
    # now find the peaks that contain the gene
    sub_df = sub_df[
        ((sub_df["Start"] <= int(start)) & (sub_df["End"] >= int(end))) | # peak contains gene
        ((sub_df["Start"] >= int(start)) & (sub_df["End"] <= int(end))) | # gene contains peak
        ((sub_df["Start"] <= int(start)) & (sub_df["End"] >= int(start))) | # peak starts before gene
        ((sub_df["Start"] <= int(end)) & (sub_df["End"] >= int(end))) # peak ends after gene
        ]
    closest_peak = list(sub_df['idx'].values)
    return closest_peak

def find_closest_start_and_end(chrom, window_start, window_end):
    sub_df = ref_df[ref_df["Chromosome"] == int(chrom.split("chr")[-1])]
    start_distances = np.abs(sub_df["Start"] - window_start)
    end_distances = np.abs(sub_df["End"] - window_end)
    start_idx = np.argmin(start_distances)
    end_idx = np.argmin(end_distances)
    return sub_df.iloc[start_idx]["idx"].item(), sub_df.iloc[end_idx]["idx"].item()

def predict_perturbations2(model, rep, correction, dataset, modality_id, feature_id, n_epochs=1):
    """predict the effect of upregulating or silencing a feature fold_change times"""
    # n_samples = dataset.data.shape[0]
    # batch_size = 128
    # define rep optimizer
    rep_optimizer = torch.optim.Adam(rep.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.5, 0.7))
    x_perturbed = dataset.data.clone().to(device)
    #x_perturbed[:, feature_id] = predictions[modality_id][:, feature_id] * fold_change
    #x_perturbed[:, feature_id] = x_perturbed[:, feature_id] * fold_change
    #x_perturbed[:, feature_id] = (x_perturbed[:, feature_id] + 1) * fold_change
    if modality_id == 1:
        x_perturbed[:, feature_id+model.train_set.modality_switch] = 0
        #x_perturbed[:, feature_id+model.train_set.modality_switch] = x_perturbed[:, feature_id+model.train_set.modality_switch] * fold_change
    else:
        x_perturbed[:, feature_id] = 0
        #x_perturbed[:, feature_id] = x_perturbed[:, feature_id] * fold_change
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

def predict_perturbations_hidden(model, testset, feature_id, step=1):
    test_set = omicsDataset(
        testset[::step, :],
        model.param_dict["modality_switch"],
        model.param_dict["scaling"],
        model.param_dict["clustering_variable(meta)"],
        model.param_dict["correction_variable"],
        model.param_dict["modalities"],
    )
    test_set.data_to_tensor()

    if step > 1:
        with torch.no_grad():
            new_reps = model.test_rep.z[::step, :].detach().cpu()
        model.test_rep = RepresentationLayer(n_rep=new_reps.shape[1], n_sample=new_reps.shape[0], value_init=new_reps).to(
            device
        )
    if not hasattr(model, "correction_test_rep"):  # so that code works for all my models
        if model.correction_gmm is None:
            model.correction_test_rep = None
    else:
        if step > 1:
            with torch.no_grad():
                new_reps = model.correction_test_rep.z[::step, :].detach().cpu()
            model.correction_test_rep = RepresentationLayer(
                n_rep=new_reps.shape[1], n_sample=new_reps.shape[0], value_init=new_reps
            ).to(device)

    
    # get up- and downregulated predictions for our given gene
    reps_original = model.test_rep.z.detach().clone().cpu().numpy()
    # also get indices of samples where the value of the feature is not 0
    #indices_of_interest = np.where(test_set.data[:,feature_id+model.train_set.modality_switch] != 0)[0]
    indices_of_interest_gene = np.where(test_set.data[:,feature_id] != 0)[0]
    #indices_of_interest_gene = np.arange(len(test_set.data))
    print("using {} samples".format(len(indices_of_interest_gene)))
    #print("using {} samples".format(len(indices_of_interest)))

    predictions_original_gene = model.decoder_forward(rep_shape=model.test_rep.z.shape[0])
    predictions_original_gene = [x.detach() for x in predictions_original_gene]
    torch.cuda.empty_cache()

    #from omicsdgd.functions._gene_to_peak_linkage import predict_perturbations2
    # run upregulation (1 step) and save predictions
    predictions_upregulated_gene = predict_perturbations2(
        model,
        model.test_rep,
        model.correction_test_rep,
        test_set,
        modality_id=0,
        feature_id=feature_id,
        n_epochs=1
    )
    predictions_original_gene = [x[indices_of_interest_gene,:] for x in predictions_original_gene]
    predictions_upregulated_gene = [x.detach()[indices_of_interest_gene,:] for x in predictions_upregulated_gene]
    with torch.no_grad():
        model.test_rep.z[:,:] = torch.tensor(reps_original[:,:], device=device)
    torch.cuda.empty_cache()
    return predictions_original_gene, predictions_upregulated_gene, test_set, indices_of_interest_gene


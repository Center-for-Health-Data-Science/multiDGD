import torch
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import adjusted_rand_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clustering_metric(r, gmm, labels):
    # transform categorical labels into numerical
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    true_labels = le.transform(labels)
    # compute probabilities per sample and component (n_sample,n_mix_comp)
    probs_per_sample_and_component = gmm.sample_probs(r.z.detach())
    # get index (i.e. component id) of max prob per sample
    cluster_labels = torch.max(probs_per_sample_and_component, dim=-1).indices.cpu().detach()
    # compute adjusted Rand index of the gmm clustering vs true labels
    radj = adjusted_rand_score(true_labels, np.asarray(cluster_labels))
    return radj

def rep_triangle_loss(r, idx):
    '''compute the circumfrance of the triangle spanned by the pairwise distances 
    between the representations of the same cell in the paired, rna and atac modalities'''
    n_split = len(idx)
    d_1 = torch.cdist(r[:n_split,:],r[n_split:(2*n_split),:])
    d_2 = torch.cdist(r[n_split:(2*n_split),:],r[(2*n_split):,:])
    d_3 = torch.cdist(r[(2*n_split):,:],r[:n_split,:])
    return d_1+d_2+d_3
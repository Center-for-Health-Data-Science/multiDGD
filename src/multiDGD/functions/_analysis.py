import pandas as pd
import anndata as ad
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import torch
from sklearn.metrics import confusion_matrix

from multiDGD import DGD
from multiDGD.latent import GaussianMixture
from multiDGD.nn import Decoder

# make model given a model name and the data
def init_model_from_name(data, name, meta, corr, d_name, model=None):
    name = name.split('.')[0]
    model_name_split = name.split('_')
    param_dict = {
        'latent_dimension': int(model_name_split[2][1:]),
        'n_hidden': int(model_name_split[3][1]),
        'n_hidden_modality': int(model_name_split[3][3])
    }
    restore_default_n_components = False
    if len(model_name_split) > 4:
        if model_name_split[4][0] == 'c':
            param_dict['n_components'] = int(model_name_split[4][1:3])
        elif model is not None:
            restore_default_n_components = True
    elif model is not None:
        restore_default_n_components = True
    if restore_default_n_components:    
        param_dict['n_components'] = int(len(list(set(model.train_set.meta))))
    
    is_train_df = pd.read_csv('data/'+d_name+'/train_val_test_split.csv')
    train_val_split = [
        list(is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values),
        list(is_train_df[is_train_df['is_train'] == 'test']['num_idx'].values)
    ]

    if model is None:
        if type(data) == ad.AnnData:
            model = DGD(data=data,
                    parameter_dictionary=param_dict,
                    train_validation_split=train_val_split,
                    #modalities='feature_types',
                    modalities=['GEX','ATAC'],
                    meta_label=meta,
                    correction=corr,
                    save_dir='./results/trained_models/'+d_name+'/',
                    model_name=name)
        else:
            model = DGD(data=data, 
                    parameter_dictionary=param_dict,
                    train_validation_split=train_val_split,
                    meta_label=meta,
                    correction=corr,
                    save_dir='./results/trained_models/'+d_name+'/',
                    model_name=name)
    else:
        model._model_name = name
        model._save_dir = './results/trained_models/'+d_name+'/'
        # update parameter dict
        for k in list(param_dict.keys()):
            model.param_dict[k] = param_dict[k]
        model.param_dict['sd_mean'] = round((2*model.param_dict['softball_scale'])/(10*model.param_dict['n_components']),2)
        # update decoder
        if model.correction_models is not None:
            updated_latent_dim = model.param_dict['latent_dimension']+len(model.correction_models)*2
        else:
            updated_latent_dim = model.param_dict['latent_dimension']
        model.decoder = Decoder(in_features=updated_latent_dim,parameter_dictionary=model.param_dict)
        # update gmm
        model.gmm = GaussianMixture(n_mix_comp=model.param_dict['n_components'],
            dim=model.param_dict['latent_dimension'],
            mean_init=(model.param_dict['softball_scale'],model.param_dict['softball_hardness']),
            sd_init=(model.param_dict['sd_mean'],model.param_dict['sd_sd']),
            weight_alpha=model.param_dict['dirichlet_a'])
    model.load_parameters()
    return model

# create a diverse enough color palette given the data
def make_palette_from_meta(name):
    meta = pd.read_csv('data/'+name+'/plotting_metadata.csv',sep=';')
    meta = meta.sort_values('lineage')
    classes_neworder = list(meta['cell_type'].values)
    n_classes = len(classes_neworder)

    palette_names = ['colorblind', 'dark', 'bright', 'deep', 'pastel']
    cluster_palette = []
    for palette_name in palette_names:
        n_colors_to_add = min(10, (n_classes-(len(cluster_palette))))
        cluster_palette.extend(sns.color_palette(palette_name, n_colors_to_add))
        if len(cluster_palette) == n_classes:
            break
    return classes_neworder, cluster_palette

"""
test_matrix = np.asarray([
    [0,0,0,0,5],
    [0,0,0,100,0],
    [0,20,0,0,0],
    [0,0,90,0,0],
    [10,0,0,30,0],
    [0,10,0,0,0]
])

import operator
def order_matrix_by_max_per_class(mtrx, class_labels):
    max_id_per_class = np.argmax(mtrx, axis=1)
    print(max_id_per_class)
    max_coordinates = list(zip(np.arange(mtrx.shape[0]), max_id_per_class))
    print(max_coordinates)
    max_coordinates.sort(key=operator.itemgetter(1))
    print(max_coordinates)
    new_class_order = [x[0] for x in max_coordinates]
    print(new_class_order)
    mtrx = mtrx[new_class_order,:]
    print(mtrx)
    print([class_labels[x] for x in new_class_order])
    #return mtrx[new_class_order,:], [class_labels[x] for x in new_class_order]


order_matrix_by_max_per_class(test_matrix, ['zero','one','two','three','four','five'])
"""

###############
# multiple functions for arriving at a sorted clustering heatmap
###############

import operator
def order_matrix_by_max_per_class(mtrx, class_labels, comp_order=None):
    if comp_order is not None:
        temp_mtrx = np.zeros(mtrx.shape)
        for i in range(mtrx.shape[1]):
            temp_mtrx[:,i] = mtrx[:,comp_order[i]]
        mtrx = temp_mtrx
    max_id_per_class = np.argmax(mtrx, axis=1)
    max_coordinates = list(zip(np.arange(mtrx.shape[0]), max_id_per_class))
    max_coordinates.sort(key=operator.itemgetter(1))
    new_class_order = [x[0] for x in max_coordinates]
    new_mtrx = np.zeros(mtrx.shape)
    # reindexing mtrx worked on test but not in application, reverting to stupid safe for-loop
    for i in range(mtrx.shape[0]):
        new_mtrx[i,:] = mtrx[new_class_order[i],:]
    #mtrx = mtrx[new_class_order,:]
    return new_mtrx, [class_labels[i] for i in new_class_order]

def gmm_clustering(r, gmm, labels):
    # transform categorical labels into numerical
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    true_labels = le.transform(labels)
    # compute probabilities per sample and component (n_sample,n_mix_comp)
    probs_per_sample_and_component = gmm.sample_probs(r.z.detach())
    # get index (i.e. component id) of max prob per sample
    cluster_labels = torch.max(probs_per_sample_and_component, dim=-1).indices.cpu().detach()
    return cluster_labels

import sklearn.metrics
def compute_distances(mtrx):
    distances = sklearn.metrics.pairwise.euclidean_distances(mtrx)
    return distances

def get_connectivity_from_threshold(mtrx, threshold):
    connectivity_mtrx = np.zeros(mtrx.shape)
    idx = np.where(mtrx <= threshold)
    connectivity_mtrx[idx[0],idx[1]] = 1
    np.fill_diagonal(connectivity_mtrx, 0)
    return connectivity_mtrx

def rank_distances(mtrx):
    ranks = np.argsort(mtrx, axis=-1)
    #testing advanced indexing for ranking
    m,n = mtrx.shape
    # Initialize output array
    out = np.empty((m,n),dtype=int)
    # Use sidx as column indices, while a range array for the row indices
    # to select one element per row. Since sidx is a 2D array of indices
    # we need to use a 2D extended range array for the row indices
    out[np.arange(m)[:,None], ranks] = np.arange(n)
    return out
    #return ranks

def get_node_degrees(mtrx):
    return np.sum(mtrx, 1)

def get_secondary_degrees(mtrx):
    out = np.zeros(mtrx.shape[0])
    for i in range(mtrx.shape[0]):
        direct_neighbors = np.where(mtrx[i] == 1)[0]
        out[i] = mtrx[direct_neighbors,:].sum()
    return out

def find_start_node(d1, d2):
    minimum_first_degree = np.where(d1 == d1.min())[0]
    if len(minimum_first_degree) > 1:
        minimum_second_degree_subset = np.where(d2[minimum_first_degree] == d2[minimum_first_degree].min())[0][0]
        return minimum_first_degree[minimum_second_degree_subset]
    else:
        return minimum_first_degree[0]

import operator
def find_next_node(c, r, i):
    connected_nodes = np.where(c[i,:] == 1)[0]
    if len(connected_nodes) > 0:
        connected_nodes_paired = list(zip(connected_nodes, r[i,connected_nodes]))
        connected_nodes_paired.sort(key=operator.itemgetter(1))
        connected_nodes = [connected_nodes_paired[x][0] for x in range(len(connected_nodes))]
    return connected_nodes

def traverse_through_graph(connectiv_mtrx, ranks, first_degrees, second_degrees):
    # create 2 lists of node ids
    # the first one keeps track of the nodes we have used already (and stores them in the desired order)
    # the second one keeps track of the nodes we still have to sort
    node_order = []
    nodes_to_be_distributed = list(np.arange(connectiv_mtrx.shape[0]))

    start_node = find_start_node(first_degrees, second_degrees)
    node_order.append(start_node)
    nodes_to_be_distributed.remove(start_node)

    count_turns = 0
    while len(nodes_to_be_distributed) > 0:
        next_nodes = find_next_node(connectiv_mtrx, ranks, node_order[-1])
        next_nodes = list(set(next_nodes).difference(set(node_order)))
        if len(next_nodes) < 1:
            next_nodes = [nodes_to_be_distributed[find_start_node(first_degrees[nodes_to_be_distributed], second_degrees[nodes_to_be_distributed])]]
        for n in next_nodes:
            if n not in node_order:
                node_order.append(n)
                nodes_to_be_distributed.remove(n)
                count_turns = 0
        count_turns += 1
        if count_turns >= 10:
            break
    
    return node_order

def order_components_as_graph_traversal(gmm):
    distance_mtrx = compute_distances(gmm.mean.detach().cpu().numpy())
    threshold = round(np.percentile(distance_mtrx.flatten(), 30), 2)

    connectivity_mtrx = get_connectivity_from_threshold(distance_mtrx, threshold)
    rank_mtrx = rank_distances(distance_mtrx)

    node_degrees = get_node_degrees(connectivity_mtrx)
    secondary_node_degrees = get_secondary_degrees(connectivity_mtrx)

    new_node_order = traverse_through_graph(connectivity_mtrx, rank_mtrx, node_degrees, secondary_node_degrees)
    return new_node_order

def gmm_make_confusion_matrix(mod, norm=True):
    classes = list(mod.train_set.meta.unique())
    true_labels = np.asarray([classes.index(i) for i in mod.train_set.meta])
    cluster_labels = gmm_clustering(mod.representation, mod.gmm, mod.train_set.meta)
    # get absolute confusion matrix
    cm1 = confusion_matrix(true_labels, cluster_labels)

    class_counts = [np.where(true_labels == i)[0].shape[0] for i in range(len(classes))]
    cm2 = cm1.astype(np.float64)
    for i in range(len(class_counts)):
        #percent_sum = 0
        for j in range(mod.gmm.n_mix_comp):
            if norm:
                cm2[i,j] = (cm2[i,j]*100 / class_counts[i])
            else:
                cm2[i,j] = cm2[i,j]
            #percent_sum += (cm2[i,j]*100 / class_counts[i])
    cm2 = cm2.round()
    print('cm2 calculated')

    # get an order of components based on connectivity graph
    component_order = order_components_as_graph_traversal(mod.gmm)

    # take the non-empty entries
    cm2 = cm2[:len(classes),:mod.gmm.n_mix_comp]

    cm3, classes_reordered = order_matrix_by_max_per_class(cm2, classes, component_order)
    #out = pd.DataFrame(data=cm3, index=classes_reordered, columns=np.arange(mod.gmm.n_mix_comp))
    out = pd.DataFrame(data=cm3, index=classes_reordered, columns=component_order)
    return out

#############
# functions for reconstruction performance comparison of multiVI and DGD
#############

def binarize(x, threshold=0.5):
    x[x >= threshold] = 1
    x[x < threshold] = 0
    return x

def classify_binary_output(target, mod, scaling_factor, switch, threshold, batch_size=5000, feature_indices=None):
    '''calculating true positives, false positives, true negatives and false negatives'''
    print('classifying binary output')
    
    n_samples = target.shape[0]
    true_positives = torch.zeros((n_samples))
    false_positives = torch.zeros((n_samples))
    true_negatives = torch.zeros((n_samples))
    false_negatives = torch.zeros((n_samples))
    
    for i in range(int(n_samples/batch_size)+1):
        print(round(i/(int(n_samples/batch_size))*100),'%')
        start = i*batch_size
        end = min((i+1)*batch_size,n_samples)
        indices = np.arange(start,end,1)
        x_accessibility = binarize(torch.Tensor(target.X[indices,switch:].todense())).int()
        y_accessibility = mod.get_accessibility_estimates(target, indices=indices)
        if type(y_accessibility) is not torch.Tensor:
            if type(y_accessibility) == pd.core.frame.DataFrame:
                y_accessibility = torch.from_numpy(y_accessibility.values)
                y_accessibility = y_accessibility.detach().cpu()
        else:
            y_accessibility = y_accessibility.detach().cpu()*scaling_factor[indices]
        y_accessibility = binarize(y_accessibility, threshold).int()
        if feature_indices is not None:
            x_accessibility = x_accessibility[:,feature_indices]
            y_accessibility = y_accessibility[:,feature_indices]
        p = (x_accessibility == 1)
        pp = (y_accessibility == 1)
        true_positives[indices] = torch.logical_and(p,pp).sum(-1).float()
        true_negatives[indices] = torch.logical_and(~p,~pp).sum(-1).float()
        false_positives[indices] = (y_accessibility > x_accessibility).sum(-1).float()
        false_negatives[indices] = (y_accessibility < x_accessibility).sum(-1).float()
    
    return true_positives, false_positives, true_negatives, false_negatives

def binary_output_scores(target, mod, scaling_factor, switch, threshold, batch_size=5000, feature_indices=None):
    '''returns FPR, FNR, balanced accuracy, LR+ and LR-'''
    tp, fp, tn, fn = classify_binary_output(target, mod, scaling_factor, switch, threshold, batch_size, feature_indices)
    tp = tp.sum()
    fp = fp.sum()
    tn = tn.sum()
    fn = fn.sum()
    tpr = tp / (tp + fn) # sensitivity
    tnr = tn / (tn + fp) # specificity
    fpr = 1 - tnr
    fnr = 1 - tpr
    balanced_accuracy = (tpr + tnr) / 2
    positive_likelihood_ratio = tpr/fpr
    negative_likelihood_ratio = fnr/tnr

    return tpr.item(), tnr.item(), balanced_accuracy.item(), positive_likelihood_ratio.item(), negative_likelihood_ratio.item()

def balanced_accuracies(target, mod, scaling_factor, switch, threshold, batch_size=5000):
    '''returns FPR, FNR, balanced accuracy, LR+ and LR-'''
    tp, fp, tn, fn = classify_binary_output(target, mod, scaling_factor, switch, threshold, batch_size)
    tpr = tp / (tp + fn) # sensitivity
    tnr = tn / (tn + fp) # specificity
    fpr = 1 - tnr
    fnr = 1 - tpr
    balanced_accuracy = (tpr + tnr) / 2
    return balanced_accuracy

def compute_error_per_sample(target, output, reduction_type='ms'):
    '''compute sample-wise error
    It can be of type `ms` (mean squared) or `ma` (mean absolute)
    '''
    error = target - output
    if reduction_type == 'ms':
        return torch.mean(error**2, dim=-1)
    elif reduction_type == 'ma':
        return torch.mean(torch.abs(error), dim=-1)
    else:
        raise ValueError('invalid reduction type given. Can only be `ms` or `ma`.')

def compute_expression_error(target, mod, scaling_factor, switch, batch_size=5000, error_type='rmse', feature_indices=None):
    '''computes expression error for target (given as anndata object)'''
    n_samples = target.shape[0]

    errors = torch.zeros((n_samples))

    for i in range(int(n_samples/batch_size)+1):
        print('   ',round(i/(int(n_samples/batch_size))*100),'%')
        start = i*batch_size
        end = min((i+1)*batch_size,n_samples)
        indices = np.arange(start,end,1)
        #target.n_vars = switch # because of multivi
        y_expression = mod.get_normalized_expression(target, indices=indices)
        if type(y_expression) is not torch.Tensor:
            if type(y_expression) == pd.core.frame.DataFrame:
                y_expression = torch.from_numpy(y_expression.values)
        y_expression *= scaling_factor[indices]
        if feature_indices is not None:
            y_expression = y_expression[:,feature_indices]
            x_expression = torch.Tensor(target.X[indices,:switch].todense())[:,feature_indices]
        else:
            x_expression = torch.Tensor(target.X[indices,:switch].todense())
        #print(y_expression[:10,:10])
        #print(torch.Tensor(target.X[indices,:switch].todense())[:10,:10])
        if error_type == 'rmse':
            errors[indices] = compute_error_per_sample(x_expression, y_expression, reduction_type='ms')
        elif error_type == 'mae':
            errors[indices] = compute_error_per_sample(x_expression, y_expression, reduction_type='ma')
        elif error_type == 'mae_sample':
            errors[indices] = compute_error_per_sample(x_expression, y_expression, reduction_type='ma')
        else:
            raise ValueError('incorrect error_type submitted. Can only be `rmse` or `mae`.')
    
    # print where the 5 highest errors are
    if error_type == 'rmse':
        out_error = torch.sqrt(torch.mean(errors))
        return out_error.item()
    elif error_type == 'mae':
        out_error = torch.mean(errors)
        return out_error.item()
    else:
        return errors

def testset_reconstruction_evaluation(testdata, mod, modality_switch, library, thresholds=[0.5], batch_size=5000, feature_indices=[None,None]):
    '''at the moment only valid for multiome'''

    # RMSE of expression values
    #rmse = compute_expression_error(testdata[:,:modality_switch], mod, library[:,0].unsqueeze(1), modality_switch)
    rmse = compute_expression_error(testdata, mod, library[:,0].unsqueeze(1), modality_switch, batch_size=batch_size, feature_indices=feature_indices[0])
    print(rmse)
    print('rmse done')

    # MAE of expression values
    #mae = compute_expression_error(testdata[:,:modality_switch], mod, library[:,0].unsqueeze(1), modality_switch, error_type='mae')
    mae = compute_expression_error(testdata, mod, library[:,0].unsqueeze(1), modality_switch, error_type='mae', batch_size=batch_size, feature_indices=feature_indices[0])
    print(mae)
    print('mae done')

    # metrics for binary accessibility output
    #tpr, tnr, balanced_accuracy, positive_likelihood_ratio, negative_likelihood_ratio = binary_output_scores(testdata[:,modality_switch:], mod, library[:,1].unsqueeze(1), modality_switch)
    for count, threshold in enumerate(thresholds):
        tpr, tnr, balanced_accuracy, positive_likelihood_ratio, negative_likelihood_ratio = binary_output_scores(testdata, mod, library[:,1].unsqueeze(1), modality_switch, threshold, batch_size=batch_size, feature_indices=feature_indices[1])
        print(threshold, tpr, balanced_accuracy, positive_likelihood_ratio, negative_likelihood_ratio)
        df_temp = pd.DataFrame({
            'RMSE (rna)': rmse,
            'MAE (rna)': mae,
            'TPR (atac)': tpr,
            'TNR (atac)': tnr,
            'balanced accuracy': balanced_accuracy,
            'LR+': positive_likelihood_ratio,
            'LR-': negative_likelihood_ratio,
            'binary_threshold': threshold
        }, index=[0])
        if count == 0:
            df_metrics = df_temp
        else:
            df_metrics = df_metrics.append(df_temp)
    print('confusion matrix done')

    return df_metrics

def knn_adjacency(dist_mtrx, k=10):
    '''
    returns the adjacency matrix based on k nearest neighbors of the input distance matrix

    this uses advanced indexing found here:
    https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
    I don' quite get it yet, but I have checked the output and it is correct
    '''
    ###
    # rank matrix
    ###
    # Sort the distance matrix (but this is not row-wise ranking yet)
    sidx = np.argsort(dist_mtrx, axis=-1)
    # Store shape info
    m,n = dist_mtrx.shape
    # Initialize output array
    out = np.empty((m,n),dtype=int)
    # Use sidx as column indices, while a range array for the row indices
    # to select one element per row. Since sidx is a 2D array of indices
    # we need to use a 2D extended range array for the row indices
    out[np.arange(m)[:,None], sidx] = np.arange(n)
    ###
    # adjacency matrix
    ###
    adjacency_mtrx = np.zeros((m,n),dtype=int)
    adjacency_mtrx[np.where((out<=k)&(out>0))] = 1
    return adjacency_mtrx

def compute_knn(np_data_array, k=10):
    '''
    computes the adjacency matrix of the k nearest neighbors
    '''
    # adjacency matrix of the representation for k nearest neighbors
    distance_matrix = compute_distances(np_data_array)
    print('distance matrix computed')
    adjacency_mtrx = knn_adjacency(distance_matrix,k=k)
    return adjacency_mtrx

def compute_knn_batch_effect(rep, data_set, k=10):
    '''
    computes batch effects for a given representation and dataset 
    based on kNN and the null hypothesis of conditional independence
    returns the average over all samples and the average over all classes
    '''
    # adjacency matrix of the representation for k nearest neighbors
    distance_matrix = compute_distances(rep)
    print('distance matrix computed')
    adjacency_mtrx = knn_adjacency(distance_matrix,k=k)
    
    # correction factors used for class identification
    correction_factors = data_set.correction_labels.numpy()

    ###
    # compute class frequencies (~p(i))
    ###
    class_freq = np.zeros((data_set.correction_classes))
    for i in range(data_set.correction_classes):
        class_freq[i] = np.sum(correction_factors==i)/len(correction_factors)
    
    ###
    # compute conditional frequencies per sample (~p(i|i))
    ###
    # expand correction factors to adjacency matrix shape
    correction_factors_expanded = torch.IntTensor(correction_factors).unsqueeze(0).expand(adjacency_mtrx.shape[0],-1)
    # get identifiers of neighbors
    indices = np.where(adjacency_mtrx == 1)
    # get the labels of k nearest neighbors per sample
    neighbor_labels = correction_factors_expanded[indices[0],indices[1]].view(adjacency_mtrx.shape[0],k).numpy()

    ###
    # compute average errors
    ###
    # per sample
    sample_error = np.zeros((len(correction_factors)))
    for i, node in enumerate(correction_factors):
        sample_error[i] = (np.sum(neighbor_labels[i] == node)/k) - class_freq[node]
    average_sample_error = sample_error.sum()/len(correction_factors)

    # per sample per class
    average_class_error = np.zeros((data_set.correction_classes))
    for i in range(data_set.correction_classes):
        average_class_error[i] = np.sum(sample_error[correction_factors == i])/np.sum(correction_factors == i)
    average_class_error = np.nan_to_num(average_class_error, 0).mean()
    
    return average_sample_error, average_class_error

# get kullback leibler divergence
def compute_distribution_frequencies(a, b):
    '''
    calculate the frequencies of observations in two discrete distributions a and b
    returns numpy arrays of the frequencies
    '''
    if isinstance(a, torch.Tensor):
        a = a.numpy()
        b = b.numpy()
    a = np.round(a)
    b = np.round(b)
    unique_values = np.sort(np.unique(np.concatenate([a,b])))
    a_frequencies = np.array([np.sum(a==value) for value in unique_values])
    b_frequencies = np.array([np.sum(b==value) for value in unique_values])
    a_frequencies = a_frequencies / np.sum(a_frequencies)
    b_frequencies = b_frequencies / np.sum(b_frequencies)
    return a_frequencies, b_frequencies

def discrete_kullback_leibler(p, q):
    '''
    Compute the discrete Kullback-Leibler divergence between two distributions.
    It can be seen as the information gain 
    achieved by using the real distribution p instead of the approximate distribution q.
    '''
    p, q = compute_distribution_frequencies(p, q)
    p = p + 1e-10
    q = q + 1e-10
    return (p * np.log(p / q)).sum()
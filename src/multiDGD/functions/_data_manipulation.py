import os
import mudata as md
import numpy as np
import anndata as ad
import pandas as pd
import scipy
import scipy.sparse
import torch
import collections

def sc_feature_selection(data, modalities, feature_selection):
    if isinstance(feature_selection, list):
        selection_method = 'variance'
    elif feature_selection > 1:
        selection_method = 'variance'
    elif (feature_selection > 0) & (feature_selection < 1):
        selection_method = 'percentage'
        percent_threshold = int(feature_selection * data.shape[0])
    else:
        raise ValueError('no valid feature selection mode chosen')
    
    if isinstance(data, md.MuData):
        modalities = list(data.mod.keys())
        new_data = []
        if selection_method == 'percentage':
            for mod in modalities:
                nonzero_id, nonzero_count = np.unique(data[mod].X.copy().tocsr().nonzero()[1], return_counts=True)
                selected_features = nonzero_id[np.where(nonzero_count >= percent_threshold)[0]]
                new_data.append(data[mod][:,selected_features])
        else:
            for i in range(len(modalities)):
                mod = modalities[i]
                var_sorted = np.argsort(np.squeeze(np.asarray(data[mod].X.copy().tocsr()[::10,:].todense()).var(axis=0)))
                selected_features = var_sorted[-feature_selection[i]:].tolist()
                new_data.append(data[mod][:,selected_features])
        data = md.MuData(dict(zip(modalities, new_data)))
        if len(modalities) > 1:
            switch = data[modalities[0]].X.shape[1]
        else:
            switch = None
    elif isinstance(data, ad.AnnData):
        import difflib
        modality_name = difflib.get_close_matches('modality', list(data.var_keys()))[0]
        modalities = list(data.var[modality_name].unique())
        if len(modalities) > 1:
            switch = np.where(data.var[modality_name] == modalities[1])[0][0]
            selected_features = []
            if selection_method == 'percentage':
                for i in range(len(modalities)):
                    if i == 0:
                        nonzero_id, nonzero_count = np.unique(data.X[:,:switch].copy().tocsr().nonzero()[1], return_counts=True)
                    else:
                        nonzero_id, nonzero_count = np.unique(data.X[:,switch:].copy().tocsr().nonzero()[1], return_counts=True)
                    selected_features.extend(list(nonzero_id[np.where(nonzero_count >= percent_threshold)[0]])+(i*switch))
                    if i == 0:
                        new_switch = len(selected_features)
            else:
                
                for i in range(len(modalities)):
                    if i == 0:
                        var_sorted = np.argsort(np.squeeze(np.asarray(data.X[:,:switch].copy().tocsr()[::10,:].todense()).var(axis=0)))
                    else:
                        var_sorted = np.argsort(np.squeeze(np.asarray(data.X[:,switch:].copy().tocsr()[::10,:].todense()).var(axis=0)))
                    selected_features.extend(var_sorted[-feature_selection[i]:].tolist()+(i*switch))
                    if i == 0:
                        new_switch = len(selected_features)
            data = data[:,selected_features]
            switch = new_switch
        else:
            if isinstance(feature_selection, list):
                feature_selection = feature_selection[0]
            if selection_method == 'percentage':
                nonzero_id, nonzero_count = np.unique(data.X.copy().tocsr().nonzero()[1], return_counts=True)
                selected_features = list(nonzero_id[np.where(nonzero_count >= percent_threshold)[0]])
            else:
                var_sorted = np.argsort(np.squeeze(np.asarray(data.X.copy().tocsr()[::10,:].todense()).var(axis=0)))
                selected_features = var_sorted[-feature_selection:].tolist()
            data = data[:,selected_features]
            switch = None
    return selection_method, data, switch

def load_data_from_name(name, dir_prefix='data/'):
    file_name = [x for x in os.listdir(dir_prefix+name) if '.h5' in x]
    if len(file_name) > 1:
        raise ValueError(f'only expected to find 1 `h5` file in the folder, but found {len(file_name)}')
    else:
        if '.h5mu' in file_name[0]:
            data = md.read(dir_prefix+name+'/'+file_name[0])
        elif '.h5ad' in file_name[0]:
            data = ad.read_h5ad(dir_prefix+name+'/'+file_name[0])
        else:
            raise ValueError('there is a problem either with the folder name provided or with the data organization')
    if name == 'mouse_gastrulation':
        data.obs['stage'] = data['atac'].obs['stage']
        data.obs['celltype'] = data['rna'].obs['celltype']
    return data

def load_testdata_as_anndata(name, dir_prefix='data/'):
    '''
    takes a dataname and returns the data's train and external test sets as anndata objects,
    as well as the modality switch and the library of the test set
    '''

    data = load_data_from_name(name)

    if type(data) is md.MuData:
        # transform to anndata
        modality_switch = data['rna'].X.shape[1]
        adata = ad.AnnData(scipy.sparse.hstack((data['rna'].X,data['atac'].X)))
        adata.obs = data['rna'].obs
        adata.var = pd.DataFrame(index=data['rna'].var_names.tolist()+data['atac'].var_names.tolist(),
                                 data={'name': data['rna'].var['gene'].values.tolist()+data['atac'].var['idx'].values.tolist(),
                                       'feature_types': ['rna']*modality_switch+['atac']*(adata.shape[1]-modality_switch)})
        #adata.var['feature_types'] = 'atac'
        #adata.var['feature_types'][:modality_switch] = 'rna'
        data = None
        data = adata
    else:
        if hasattr(data, 'layers'):
            data.X = data.layers['counts']
        modality_switch = np.where(data.var['feature_types'].values != data.var['feature_types'].values[0])[0][0]
    
    # make train and test subsets
    is_train_df = pd.read_csv(dir_prefix+name+'/train_val_test_split.csv')
    train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values
    test_indices = is_train_df[is_train_df['is_train'] == 'iid_holdout']['num_idx'].values
    if not isinstance(data.X, scipy.sparse.csc_matrix):# type(data.X) is not scipy.sparse._csc.csc_matrix:
        data.X = data.X.tocsr()
    trainset = data.copy()[train_indices]
    testset = data.copy()[test_indices]

    library = torch.cat(
        (torch.sum(torch.Tensor(testset.X.todense())[:,:modality_switch], dim=-1).unsqueeze(1),
        torch.sum(torch.Tensor(testset.X.todense())[:,modality_switch:], dim=-1).unsqueeze(1))
        ,dim=1
    )

    return trainset, testset, modality_switch, library

def get_column_name_from_unique_values(df, column_uniques):
    for colname in df:
        uniques = df[colname].unique()
        if collections.Counter(uniques) == collections.Counter(column_uniques):
            return colname
    raise ValueError('could not find the column for the modalities')
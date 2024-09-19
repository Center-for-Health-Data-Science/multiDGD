from typing import Union, Optional, Sequence, Tuple, List, Dict, Any
from mudata import MuData
from anndata import AnnData

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def setup_data(data: Union[MuData, AnnData], modality_key: str=None, observable_key: str =None, layer: str = None, 
               covariate_keys: List[str] = None, train_fraction: float =0.8, include_test: bool = True, reference = None) -> Union[MuData, AnnData]:
    '''
    This function will prepare the data for the model. Input formats can be both anndata and mudata objects.

    Arguments
    ----------
    data : anndata or mudata object
    modality_key: str, optional
        If the object is not a mudata object, this key will be used to define the modalities of the data.
    observable_key: str, optional
        Key of the 'observable' factor that will be used to define the number of GMM components in the prior distribution.
        If None, the model initialization will ask for a definition of the numberof components. The default is None.
    layer : str, optional
        Layer of the data to use. If None, use X. The default is None.
    covariate_keys: list, optional
        List of keys of the 'nuisance' factors that should be excluded from the biological representation,
        i.e. batch, donor, disease state, ... The default is None.
    train_fraction: float, optional
        Fraction of the data that will be used for training. The default is 0.8.
    include_test: bool, optional
        If True, the test set will be included in the data object. The default is True. 
        Otherwise, the split will only be train and validation.
        For integrating a new data set, set the train_fraction to 1.0.

    Returns
    -------
    Anndata or mudata object with the data prepared for the model.

    '''

    n_samples = data.shape[0]

    # if there is no reference, the data is not external and matches or defines the model structure
    if reference is None:

        # Check if data is anndata or mudata
        if isinstance(data, ad.AnnData):
            if modality_key is None:
                print('Warning: No modality key was provided. Assuming that the data is single modality and of type `rna`.')
                data.var['modality'] = 'rna'
            else:
                if modality_key not in data.var.keys():
                    raise ValueError('Modality key not found in data object. Please make sure it is in data.var')
                else:
                    modalities = data.var[modality_key].values
                    data.var['modality'] = modalities

        # if layer is defined, move its content to X
        if layer is not None:
            if isinstance(data, ad.AnnData):
                data.X = data.layers[layer]
            elif isinstance(data, md.MuData):
                for mod in list(data.mod.keys()):
                    data.mod[mod].X = data.mod[mod].layers[layer]
            else:
                raise ValueError('Data object is not an anndata or mudata object.')

        # if observable_key is defined, add it to the obersables
        if observable_key is not None:
            data.obs['observable'] = data.obs[observable_key]
        else:
            print('Warning: No observable key was provided. The model will ask for the number of components.')

        # if covariate_keys is defined, add it to the covariates
        if covariate_keys is not None:
            for cov in covariate_keys:
                data.obs['covariate_'+cov] = data.obs[cov]

        # add train-val-test split
        if 'train_val_test' not in data.obs.keys():
            if train_fraction < 1.0:
                if include_test:
                    if observable_key is not None: # stratified split
                        train_indices, test_indices = train_test_split(np.arange(n_samples), test_size=(1.0-train_fraction)/2, stratify=data.obs['observable'].values)
                        train_indices, val_indices = train_test_split(train_indices, test_size=(((1.0-train_fraction)/2)/(1.0-(1.0-train_fraction)/2)), stratify=data.obs['observable'].values[train_indices])
                    else:
                        train_indices, test_indices = train_test_split(np.arange(n_samples), test_size=(1.0-train_fraction)/2)
                        train_indices, val_indices = train_test_split(train_indices, test_size=(((1.0-train_fraction)/2)/(1.0-(1.0-train_fraction)/2)))
                    train_val_test = [''] * n_samples
                    train_val_test = ['train' if i in train_indices else train_val_test[i] for i in range(n_samples)]
                    train_val_test = ['validation' if i in val_indices else train_val_test[i] for i in range(n_samples)]
                    train_val_test = ['test' if i in test_indices else train_val_test[i] for i in range(n_samples)]
                else:
                    if observable_key is not None: # stratified split
                        train_indices, val_indices = train_test_split(np.arange(n_samples), test_size=(1.0-train_fraction), stratify=data.obs['observable'].values)
                    else:
                        train_indices, val_indices = train_test_split(np.arange(n_samples), test_size=(1.0-train_fraction))
                    train_val_test = [''] * n_samples
                    train_val_test = ['train' if i in train_indices else train_val_test[i] for i in range(n_samples)]
                    train_val_test = ['validation' if i in val_indices else train_val_test[i] for i in range(n_samples)]
            else:
                train_val_test = 'test'
            data.obs['train_val_test'] = train_val_test
    
    else:
        ###
        # compare reference (model) to new data
        ###
        # check if same number of features
        data_shape = data.shape
        ref_shape = reference.train_set.n_features
        if data_shape[1] != ref_shape:
            print("The new data set is different from the data the model is trained on.")
            print("...trying to match the data structure.")
            # check if same number of modalities
            if isinstance(data, ad.AnnData):
                n_modalities = len(np.unique(data.var['modality']))
            else:
                n_modalities = len(data.mod.keys())
            n_modalities_ref = len(reference.train_set.modalities)
            # compare modalities
            if n_modalities != n_modalities_ref:
                print("The new data set has a different number of modalities than the data the model is trained on.")
            # check which modality the new data is
            if isinstance(data, ad.AnnData):
                modalities = np.unique(data.var['modality'])
            else:
                modalities = data.mod.keys()
            matching_modalities = [np.where(reference.train_set.modalities == mod)[0][0] for mod in modalities if mod in reference.train_set.modalities]
            print("...the new data matches the following modalities of the model: ", matching_modalities)
            
            print("Now ordering the new data features according to the training data...")
            # create space holder for new data
            new_data = md.MuData(dict(zip(reference.train_set.modalities, [None]*len(reference.train_set.modalities))))
            for i, mod in enumerate(reference.train_set.modalities):
                # get the new data for that modality
                new_data.mod[mod] = ad.AnnData(X=np.zeros((temp_data.shape[0], reference.train_set.modality_features[i])), obs=temp_data.obs)
                new_data.mod[mod].uns['usable_features'] = []
                if mod in modalities:
                    if isinstance(data, ad.AnnData):
                        temp_data = data[:,data.var['modality'] == mod]
                    else:
                        temp_data = data.mod[mod]
                    # get the variable index for that modality in train set
                    if i == 0:
                        idx_start = 0
                        idx_end = reference.train_set.modality_switch
                    elif i == len(reference.train_set.modalities)-1:
                        idx_start = reference.train_set.modality_switch
                        idx_end = reference.train_set.n_features
                    else:
                        raise ValueError('Currently only two modalities supported.')
                    var_names = reference.train_set.var_idx[idx_start:idx_end]
                    # get the features of the training data that are also in the new data
                    intersection_original_position = np.where(var_names == temp_data.var.index)[0]
                    # get the new order in which to put the new data
                    intersection_in_oder = [np.where(temp_data.var.index == var_names[i])[0][0] for i, gene in enumerate(var_names) if gene in list(temp_data.var.index)]
                    print(str(len(intersection_in_oder)) + ' out of ' + str(len(var_names)) + ' features found in new data (modality '+str(i)+': '+mod+').')
                    # assign the new data to the new order
                    new_data.mod[mod].X[:,intersection_original_position] = temp_data.X[:,intersection_in_oder]
                    new_data.mod[mod].var.index = var_names
                    # add the indices of usable features as a layer (to know which can be used to compute gradients)
                    new_data.mod[mod].uns['usable_features'] = intersection_original_position
            # make a dictionary with modalities as keys and data as values
            new_data.obs = data.obs
        else:
            raise ValueError('So far only did this for very different data sets.')

        print("...succeeded.")
    
    return data
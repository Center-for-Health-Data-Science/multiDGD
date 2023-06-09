from typing import Union, Optional, Sequence, Tuple, List, Dict, Any
from mudata import MuData
from anndata import AnnData

import anndata as ad
import mudata as md
import numpy as np
from sklearn.model_selection import train_test_split

def setup_data(data: Union[MuData, AnnData], modality_key: str=None, observable_key: str =None, layer: str = None, 
               covariate_keys: List[str] = None, train_fraction: float =0.8, include_test: bool = True) -> Union[MuData, AnnData]:
    '''
    This function will prepare the data for the model. Input formats can be both anndata and mudata objects.

    Parameters
    ----------
    data : anndata or mudata object
    layer : str, optional
        Layer of the data to use. If None, use X. The default is None.
    modality_key: str, optional
        If the object is not a mudata object, this key will be used to define the modalities of the data.
    observable_key: str, optional
        Key of the 'observable' factor that will be used to define the number of GMM components in the prior distribution.
        If None, the model initialization will ask for a definition of the numberof components. The default is None.
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
    Anndata or mudata object

    '''

    n_samples = data.shape[0]

    # Check if data is anndata or mudata
    if isinstance(data, ad.AnnData):
        if modality_key is None:
            print('Warning: No modality key was provided. Assuming that the data is single modality.')
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
    
    return data
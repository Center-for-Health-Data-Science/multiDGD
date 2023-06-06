import torch
import numpy as np
from torch.utils.data import Dataset
import anndata as ad
import mudata as md
from sklearn import preprocessing
import scipy.sparse

class omicsDataset(Dataset):
    '''
    General Dataset class for single cell data.
    Applicable for sinlge modalities and multi-modal data.

    Attributes
    ----------
    data: torch.Tensor
        tensor with raw counts of shape (n_samples, n_features)
    scaling_type: string
        variable defining how to calculate scaling factors
    n_sample: int
        number of samples in dataset
    n_features: int
        number of total features in dataset
    library: torch.Tensor
        tensor of per-sample and -modality scaling factors

    Methods
    ----------
    get_labels(idx=None)
        Return sample-specific values of monitored clustering feature (if available)
    get_correction_labels(corr_id, idx=None)
        Return values of numerically transformed correction features per sample
    '''
    def __init__(self,
            data,
            scaling_type='sum', 
            split='train'
        ):
        '''
        Args: 
            data: mudata or anndata object
            scaling_type: string 
                for different scaling options (currently only supporting 'sum')
            split: string
                for different splits of the data. Can be 'train', 'validation', 'test'. default is 'train'. 
        '''
        # reduce the data to the relevant data split
        if split == 'test': # it can be that one wants to use a completely new data set in "testing"
            if 'test' not in data.obs['train_val_test'].unique():
                self.data = data
            else:
                self.data = data[data.obs['train_val_test'] == split]
        else:
            self.data = data[data.obs['train_val_test'] == split]
        
        # now if there is a lot of cells in this data, we will prefer slower training over large memory use
        self.sparse = False
        if self.data.shape[0] > 1e5:
            self.sparse = True

        # the scaling type determines what makes the scaling factors of each sample
        # it is accessed by the loss function
        self.scaling_type = scaling_type

        # get modality name(s), position(s) at which modalities switch in full tensor, and number of features in each modality
        self.modalities, self.modality_switch, self.modality_features = self._get_modality_names()
        
        # I tested training with mosaic data, but that does not work well
        # I will add a training scheme in the future and then will do proper supprt
        self.mosaic = self._check_if_mosaic()
        self.mosaic_mask = None
        self.mosaic_train_idx = None
        self.modality_mask = None
        """
        if self.mosaic and label == 'train':
            # if the train set is mosaic, we use 10% of the paired data to minimize distances between modalities
            # for this we need to copy those samples, add them with each modality option (paired, GEX, ATAC)
            # and structure them in such a way that we can use the representation distances in the loss
            data, data_triangle, self.modality_mask, self.modality_mask_triangle, self.mosaic_train_idx = self._make_mosaic_train_set(data)
            self.data_triangle = torch.Tensor(data_triangle.X.todense())
        elif self.mosaic and label == 'test':
            self.modality_mask = self._get_mosaic_mask(data)
        """

        # make shape attributes
        self.n_sample = self.data.shape[0]
        self.n_features = self.data.shape[1]

        # get meta data (feature for clustering) and correction factors (if applicable)
        self.meta, self.correction, self.correction_classes = self._init_meta_and_correction()

        self.correction_labels = None
        if self.correction is not None:
            self.correction_labels = self._init_correction_labels_numerical()

        # compute the scaling factors for each sample based on scaling type (later, when desnified)
        self.library = None

    def __len__(self):
        '''Return number of samples in dataset'''
        return(self.data.shape[0])

    def __getitem__(self, idx):
        '''Return data, scaling factors and index per sample (evoked by dataloader)'''
        expression = self.data[idx]
        lib = self.library[idx]
        return expression, lib, idx
    
    def __str__(self):
        return f"""
        omicsDataset:
            Number of samples: {self.n_sample}
            Modalities: {self.modalities}
            Features per modality: {self.modality_features}
            Total number of features: {self.n_features}
            Scaling of values: {self.scaling_type}
        """
    
    def __eq__(self, other):
        '''Check if two instances are the same'''
        is_eq = False
        if (self.data == other.data) and (self.library == other.library):
            is_eq = True
        return is_eq
    
    def get_labels(self, idx=None):
        '''Return sample-specific values of monitored clustering feature (if available)'''
        if self.meta is not None:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            if idx is None:
                return np.asarray(np.array(self.meta))
            else:
                return np.asarray(np.array(self.meta)[idx])
        else:
            print('tyring to access meta labels, but none was given in data initialization.')
            return None
    
    def _get_library(self):
        '''Create tensor of scaling factors of shape (n_samples,n_modalities)'''
        if not self.sparse:
            if self.modality_switch is not None:
                library = torch.cat((torch.sum(self.data[:,:self.modality_switch], dim=-1).unsqueeze(1),torch.sum(self.data[:,self.modality_switch:], dim=-1).unsqueeze(1)),dim=1)
            else:
                library = torch.sum(self.data, dim=-1).unsqueeze(1)
        else:
            if self.modality_switch is not None:
                library = torch.cat((torch.tensor(self.data[:,:self.modality_switch].sum(axis=-1).toarray()), torch.tensor(self.data[:,self.modality_switch:].sum(axis=-1).toarray())))
            else:
                library = torch.tensor(self.data.sum(axis=-1).toarray())
        return library
    
    def _get_modality_names(self):
        '''
        Get the types of modalities of the data.
        In the future this can be important if e.g. 
        including protein abundance for which we need different output distributions.

        This also returns the positions in the data tensor where modalities switch
        and the number of features per modality.
        '''
        if isinstance(self.data, md.MuData):
            modalities = list(self.data.mod.keys())
            return modalities, int(self.data[modalities[0]].shape[1]), [int(self.data[mod].shape[1]) for mod in modalities]
        elif isinstance(self.data, ad.AnnData):
            # let's make the rule that if people want to use a multi-modal anndata object, they have to provide the modalities column name as modalities
            # otherwhise I treat the data as unimodal
            modalities = list(self.data.var['modality'].unique())
            if len(modalities) > 1:
                switch = np.where(self.data.var['modality'] == modalities[1])[0][0]
                # currently only support 2 modalities
                modality_features = [int(np.where(self.data.var['modality'] == modalities[1])[0][0]), int((self.data.shape[1]-np.where(self.data.var['modality'] == modalities[1])[0][0]))]
            else:
                modality_features = [int(self.data.shape[1])]
            return modalities, switch, modality_features
    
    #def _data_to_tensor(self, data):
    def data_to_tensor(self):
        '''
        Make a tensor out of data. In multi-modal cases, modalities are concatenated.
        This only works with mudata and anndata objects.
        '''
        if not self.sparse:
            if isinstance(self.data, md.MuData):
                self.data = torch.cat(tuple([torch.Tensor(self.data[x].X.todense()) for x in self.modalities]), dim=1)
                #return torch.cat(tuple([torch.Tensor(data[x].X.todense()) for x in self.modalities]), dim=1)
            elif isinstance(self.data, ad.AnnData):
                self.data = torch.Tensor(self.data.X.todense())
                #return torch.Tensor(data.X.todense())
            else:
                raise ValueError('unsupported data type provided. please check documentation for further information.')
        self.library = self._get_library()
    
    def _init_meta_and_correction(self):
        '''
        Depending on the user's input, the model may need to disentangle certain
        correction factors (``correction``) in the representation and monitor the clustering performance
        given a specific feature (``meta_label``). 
        These features are given as the data objects observation column names.
        
        This method initializes corresponding attributes accordingly.
        The correction feature can also be a list of column names to accomodate multiple correction factors.
        '''
        # get sample-wise values of the clustering feature
        try:
            meta = self.data.obs['observable'].values
        except:
            meta = None
            print('WARNING: no observable provided in dataset generation. Monitoring clustering will not be possible.')
        
        # get sample-wise values of the correction features and the number of classes per feature
        correction_features = None
        n_correction_classes = None
        covariates = [x for x in self.data.obs.columns if 'covariate_' in x]
        if len(covariates) > 0:
            correction_features = self.data.obs[covariates].values
            if type(correction_features) is np.ndarray: # I don't understand what I did here, need to observe
                if len(correction_features.shape) > 1:
                    correction_features = correction_features.flatten()
                n_correction_classes = len(list(np.unique(correction_features)))
            else:
                n_correction_classes = len(correction_features.unique())
        
        return meta, correction_features, n_correction_classes
    
    def _init_correction_labels_numerical(self):
        '''
        Transforms correction features into numerical variables for supervised training (clustering performance).
        '''
        le = preprocessing.LabelEncoder()
        le.fit(self.correction)
        correction_numerical = torch.tensor(le.transform(self.correction))
        return correction_numerical
    
    #def get_correction_labels(self, corr_id, idx=None):
    def get_correction_labels(self, idx=None):
        '''
        Return values of numerically transformed correction features per sample
        '''
        if idx is None:
            return self.correction_labels.tolist()
        else:
            return self.correction_labels[idx].tolist()
    
    def _check_if_mosaic(self):
        '''Check if data is mosaic data
        that means whether the data has unpaired modalities'''
        mosaic = False
        # this should get support later, but also needs a different learning scheme
        return mosaic
    
    def _get_mosaic_mask(self, d):
        '''Return a list of tensors that indicate which samples belong to which modality'''
        if self.mosaic:
            modality_list = d.obs['modality']
            modality_mask = [torch.zeros((d.shape[0])).bool(), torch.zeros((d.shape[0])).bool()]
            mod_name_1 = [x for x in modality_list.unique() if x in ['rna', 'RNA', 'GEX', 'expression']][0]
            mod_name_2 = [x for x in modality_list.unique() if x in ['atac', 'ATAC', 'accessibility']][0]
            modality_mask[0][modality_list == mod_name_1] = True
            modality_mask[1][modality_list == mod_name_2] = True
        return modality_mask
    
    def _make_mosaic_train_set(self, split=10):
        '''For mosaic train sets, take 10% of the paired data,
        artificially unpair it and keep all three sets (paired, unpaired, unpaired)
        structured so that we can take the first representation segments and compute triangle
        sizes for the loss'''
        idx_paired = np.where(self.data.obs["modality"].values == "paired")[0]
        if len(idx_paired) >= 10: # in case I have a 100% unpaired dataset
            # choose every 10th sample (for reproducibility)
            idx_split = idx_paired[::split]
            n_split = len(idx_split)
            # take selected samples and store them in new data object for each modality
            # then append all the other data to the new data object
            data_1 = self.data.copy()[idx_split,:]
            data_1_rna = data_1.copy()
            data_1_atac = data_1.copy()
            data_1_rna.obs["modality"] = "GEX"
            data_1_atac.obs["modality"] = "ATAC"
            data_1 = data_1.concatenate(data_1_rna)
            data_1 = data_1.concatenate(data_1_atac)
            idx = torch.Tensor(np.arange(n_split)).byte()
            return data_1, self._get_mosaic_mask(self.data), self._get_mosaic_mask(data_1), idx
        else:
            return None, self._get_mosaic_mask(self.data), None, None
    
    def get_mask(self, indices):
        if self.modality_mask is None:
            return None
        else:
            return [x[indices] for x in self.modality_mask]
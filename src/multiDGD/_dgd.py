import torch
import torch.nn as nn
import numpy as np
import pandas as pd
#import wandb
import json
import scipy.sparse

from multiDGD.dataset import omicsDataset
from multiDGD.latent import RepresentationLayer
from multiDGD.latent import GaussianMixture, GaussianMixtureSupervised
from multiDGD.nn import Decoder
from multiDGD.functions import train_dgd, set_random_seed, count_parameters, sc_feature_selection, learn_new_representations
from multiDGD.functions._gene2peak import predict_perturbations

# define device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Central Module of the package'''

class DGD(nn.Module):
    '''
    This is the main class for the Deep Generative Decoder.
    Given a mudata or anndata object, it creates an instance of the DGD
    with all necessary classes from the data.

    Attributes (non-optional ones)
    ----------
    train_set: omicsDataset
        Dataset object derived from only obligatory input `data`.
        it's properties (shape, modality type, observable feature classes) 
        are used to build remaining instances
    param_dict: dict
        dictionary containing hyperparameters for building model instances and training.
        Initialized with default parameters and updated with optional user input.
    decoder: Decoder
        decoder instance initialized based on desired latent dimensionality
        and data features.
    representation: RepresentationLayer
        learnable representation vectors for the training set
    gmm: GMM
        Gaussian mixture model instance for the distribution over latent space.
        If no other information is given (but received a clustering observable),
        number of components is automatically set to the number of classes.

    Methods
    ----------
    train(n_epochs=500, stop_with='loss', stop_after=10, train_minimum=50)
        training of the model instances (decoder, representation, gmm) for a given
        number of epochs with early stopping. 
        Early stopping can be based on the training (or validation, if applicable) loss
        or on the clustering performance on a desired observable (e.g. cell type).
    save()
        saving the model parameters
    load()
        loading trained model parameters to initialized model
    get_representation()
        access training representations
    predict_new()
        learn representations for new data
    differential_expression()
        perform differential expression analysis on selected groupings of data
    '''

    def __init__(
            self,
            data,
            parameter_dictionary=None,
            scaling='sum',
            save_dir='./',
            random_seed = 0,
            model_name='dgd',
            print_outputs=False
        ):
        super().__init__()

        # setting the random seed at the beginning for reproducibility
        set_random_seed(random_seed)

        # setting internal helper attributes
        self._init_internal_attribs(data, model_name, save_dir, scaling, print_outputs) # needs to go? meta_label to data, dev_mode to train
        #self.trained_status = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)

        ###
        # initializing data
        ###
        # this also checks whether the data is in an acceptable format and initialize
        self._init_data(data)
        
        # update parameter dictionary with optional input and info from data
        self.param_dict = self._init_parameter_dictionary(parameter_dictionary)

        ###
        # initialize GMM as latent distribution and representations
        ###
        self._init_gmm()
        self._init_representations()
        
        # initialize supervised mixture models and representations for correction factors
        # these could be batch effect or medical features of interest (disease state etc.)
        self._init_correction_models()
        
        # initialize decoder
        self._init_decoder()
    
    def _init_internal_attribs(self, data, model_name, save_dir, scaling, print_outputs):
        # define what to name the model and where to save it
        self._model_name = model_name
        self._save_dir = save_dir
        # scaling relationship between model output and data
        self._scaling = scaling
        # optional: features to observe and to correct for
        self._clustering_variable = 'observable'
        self._correction_variables = [x for x in data.obs.columns if 'covariate_' in x]
        # developer mode changes from notebook use with progress bar to monitoring jobs with wandb
        #self._developer_mode = developer_mode
        # important for checking whether a model is trained or has loaded learned parameters
        self.trained_status = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self._print_output = print_outputs
    
    def _init_wandb_logging(self, parameter_dictionary): 
        '''start run if in developer mode (otherwise assumes running in notebook)'''
        import wandb
        try:
            wandb.init(
                project=parameter_dictionary['log_wandb'][1], 
                entity=parameter_dictionary['log_wandb'][0], 
                config=parameter_dictionary)
        except:
            raise ValueError('You are trying to run in developer mode, but seem not to have given the parameter dictionary the `log_wandb` statement or you are not logged in to wandb.')
        wandb.run.name = self._model_name
    
    def _init_data(self, data):
        '''Internal function to derive datasets and data-dependent parameters from input data'''
        
        if self._print_output:
            print('#######################')
            print('Initialized data')
            print('#######################')
        
        self.train_set = omicsDataset(data, split='train', scaling_type=self._scaling)
        if 'validation' in data.obs['train_val_test'].values:
            self.val_set = omicsDataset(data, split='validation', scaling_type=self._scaling) # if there is no validation, return None
        if 'test' in data.obs['train_val_test'].values:
            self.test_set = omicsDataset(data, split='test', scaling_type=self._scaling)
        # create the train, val and test indices like in multivi
        self.train_indices = np.where(data.obs['train_val_test'].values == 'train')[0]
        self.validation_indices = np.where(data.obs['train_val_test'].values == 'validation')[0]
        self.test_indices = np.where(data.obs['train_val_test'].values == 'test')[0]
        self.total_cells = data.n_obs

        # save the data split in case people don't keep this info
        self.save_data_splits(data)


    def _init_parameter_dictionary(self, init_dict=None):
        '''initialize the parameter dictionary from default and update with optional input'''
        # init parameter dictionary based on defaults
        # Opening JSON file
        out_dict = default_parameters
        #"""
        if "sd_mean" not in out_dict.keys():
            out_dict['sd_mean'] = 0.2 * round((out_dict['softball_scale'])/(out_dict['n_components']),2)
        # update from optional class input dictionary
        if init_dict is not None:
            for key in init_dict.keys():
                out_dict[key] = init_dict[key]
        # add information gained in data initialization
        out_dict['n_features'] = self.train_set.n_features # number of total features
        out_dict['n_features_per_modality'] = self.train_set.modality_features # list of output features generated depending on number of modalities
        out_dict['modalities'] = self.train_set.modalities
        out_dict['modality_switch'] = int(self.train_set.modality_switch) # updating the modality switch if multi-modal data in AnnData object and switch was not given
        out_dict['scaling'] = self._scaling
        out_dict['clustering_variable(meta)'] = self._clustering_variable
        out_dict['correction_variable'] = self._correction_variables
        
        # automate selection of number of components
        if self.train_set.meta is not None:
            select_n_components = False
            if init_dict is not None:
                if 'n_components' not in list(init_dict.keys()):
                    select_n_components = True
            else:
                select_n_components = True
            if select_n_components:
                out_dict['n_components'] = int(len(list(set(self.train_set.meta))))
                print('selected ',out_dict['n_components'],' number of Gaussian mixture components based on the provided observable.')
        else:
            print("WARNING: Neither observable key nor number of Gaussian components was provided. Creating the embedding based on a single Gaussian, which will most likely result in a lower-quality embedding.")
        
        # overwrite hyperparameters with passed parameter_dictionary (if applicable)
        # it is crucial to update the initial value for the GMM component std if number of component changes from default
        # this will be applied at the end of data initialization
        override_sd_init = True
        if init_dict is not None:
            if 'sd_mean' in list(init_dict.keys()):
                override_sd_init = False
        if override_sd_init:
            out_dict['sd_mean'] = round((2*out_dict['softball_scale'])/(10*out_dict['n_components']),2)
        
        self.latent = out_dict['latent_dimension']
        return out_dict
    
    def _init_gmm(self):
        '''create GMM instance as distribution over latent space'''
        if self._print_output:
            print('#######################')
            print('Initializing model parts')
            print('#######################')
        self.gmm = GaussianMixture(n_mix_comp=self.param_dict['n_components'],
            dim=self.param_dict['latent_dimension'],
            mean_init=(self.param_dict['softball_scale'],self.param_dict['softball_hardness']),
            sd_init=(self.param_dict['sd_mean'],self.param_dict['sd_sd']),
            weight_alpha=self.param_dict['dirichlet_a']).to(device)
        if self._print_output:
            print(self.gmm)
    
    def _init_representations(self):
        '''create representation instances. If validation split is given, creates 2 representations'''
        # initialize representation(s)
        self.representation = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
            n_sample=self.train_set.n_sample,
            value_init=self.param_dict['value_init']).to(device)
        if self._print_output:
            print(self.representation)
        self.validation_rep = None
        if self.val_set is not None:
            self.validation_rep = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
                n_sample=self.val_set.n_sample,
                value_init=self.param_dict['value_init']).to(device)
        self.test_rep = None
    
    def _init_correction_models(self, dim=2):
        '''
        create correction models (additional, disentangled representation + gmm instances)
        the correction input is None or a string as the feature name
        '''
        if self.train_set.correction is not None:
            n_correction_classes = self.train_set.correction_classes
            '''I want to add support for multiple correction variables, but this is not yet implemented'''
            #n_correction_models = len(n_correction_classes)
            self.correction_gmm = GaussianMixtureSupervised(
                    Nclass=n_correction_classes,Ncompperclass=1,dim=dim,
                    mean_init=(self.param_dict['softball_scale_corr'],self.param_dict['softball_hardness_corr']),
                    sd_init=(round((2*self.param_dict['softball_scale_corr'])/(10*n_correction_classes),2),
                    self.param_dict['sd_sd_corr']),alpha=2).to(device)
            self.correction_rep = RepresentationLayer(
                    n_rep=dim,n_sample=self.train_set.n_sample,
                    value_init=self.correction_gmm.supervised_sampling(self.train_set.get_correction_labels(),
                    sample_type='origin')).to(device)
            if self.validation_rep is not None:
                self.correction_val_rep = RepresentationLayer(
                        n_rep=dim,n_sample=self.val_set.n_sample,
                        value_init='zero').to(device)
            print("Covariate model initialized as:")
            print(self.correction_gmm)
        else:
            self.correction_gmm = None
            self.correction_rep = None
            self.correction_val_rep = None
    
    def _init_decoder(self):
        '''create decoder instance'''
        if self.correction_gmm is not None:
            updated_latent_dim = self.param_dict['latent_dimension']+self.correction_gmm.dim
        else:
            updated_latent_dim = self.param_dict['latent_dimension']
        self.decoder = Decoder(in_features=updated_latent_dim,parameter_dictionary=self.param_dict).to(device)
        #if self._developer_mode:
        #    wandb.run.summary["N_parameters"] = count_parameters(self.decoder)
        self.param_dict['n_decoder_parameters'] = count_parameters(self.decoder)
        if self._print_output:
            print(self.decoder)
    
    def _get_train_status(self):
        '''print the training status of the model'''
        print('#######################')
        print('Training status')
        print('#######################')
        print(self.trained_status.item())
    
    def is_trained(self):
        print(self.trained_status.item())
    
    def train(self, n_epochs=500, stop_with='loss', stop_after=10, train_minimum=50, developer_mode=False):
        '''
        train model for n_epochs
        
        options for early stopping are 'loss' and 'clustering' (which requires meta_label in DGD init)
        
        ealry stopping observation interval is stop_after, and a minimum number of epochs to be trained
        can be specified with train_minimum

        developer_mode defines how progress is displayed.
        If false, it is assumed that a jupyter notebook is used. Then, progress will be shown in a progress bar.
        If true, a wandb run will be initialized, logging many training metrics.
        '''

        if developer_mode:
            self._init_wandb_logging(self.param_dict) # go to train
        
        # prepare data loaders
        # first step is to transform the data into tensors
        print('Preparing data loaders')
        self.train_set.data_to_tensor()
        if self.train_set.sparse:
            train_loader = torch.utils.data.DataLoader(
                self.train_set, batch_size=self.param_dict['batch_size'],
                shuffle=True, num_workers=0, collate_fn=collate_sparse_batches
                )
        else:
            train_loader = torch.utils.data.DataLoader(
                self.train_set, batch_size=self.param_dict['batch_size'],
                shuffle=True, num_workers=0
                )
        validation_loader = None
        if self.validation_rep is not None:
            self.val_set.data_to_tensor()
            if self.val_set.sparse:
                validation_loader = torch.utils.data.DataLoader(
                    self.val_set, batch_size=self.param_dict['batch_size'],
                    shuffle=True, num_workers=0, collate_fn=collate_sparse_batches
                    )
            else:
                validation_loader = torch.utils.data.DataLoader(
                self.val_set, batch_size=self.param_dict['batch_size'],
                shuffle=True, num_workers=0
                )
        
        print('Now training')
        self.decoder, self.gmm, self.representation, self.validation_rep, self.correction_gmm, self.correction_rep, self.correction_val_rep, self.history = train_dgd(
            self.decoder, self.gmm, self.representation, 
            self.validation_rep, train_loader, validation_loader,
            self.correction_gmm, self.correction_rep, self.correction_val_rep, n_epochs,
            learning_rates=self.param_dict['learning_rates'],
            adam_betas=self.param_dict['betas'],wd=self.param_dict['weight_decay'],
            stop_method=stop_with,stop_len=stop_after,train_minimum=train_minimum,
            save_dir=self._save_dir,
            developer_mode=developer_mode
            )
        
        self.trained_status[0] = 1

        self._get_train_status()
    
    def save(self, save_dir=None, model_name=None):
        '''save trained parameters'''
        if save_dir is None:
            save_dir = self._save_dir
        if model_name is None:
            model_name = self._model_name
        torch.save(self.state_dict(), save_dir+model_name+'.pt')
        self.save_param_dict_to_file(save_dir+model_name)
    
    def load_parameters(self):
        '''load model'''
        # load state dict
        checkpoint = torch.load(self._save_dir+self._model_name+'.pt',map_location=torch.device('cpu'))
        # based on what people may provide for loading vs. training, there might be discrepancies in dataset numbers and sizes
        # e.g. there might have been a train-val split but now they provide no split list
        # so check sizes of representations
        if self.representation.z.shape != checkpoint['representation.z'].shape:
            print('data(split) is not the same as provided for training. Correcting representation size.')
            self.representation = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
                n_sample=checkpoint['representation.z'].shape[0],
                value_init=self.param_dict['value_init']).to(device)
        if 'validation_rep.z' in list(checkpoint.keys()):
            if self.validation_rep is None:
                self.validation_rep = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
                    n_sample=checkpoint['validation_rep.z'].shape[0],
                    value_init=self.param_dict['value_init']).to(device)
            elif self.validation_rep.z.shape != checkpoint['validation_rep.z'].shape:
                self.validation_rep = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
                    n_sample=checkpoint['validation_rep.z'].shape[0],
                    value_init=self.param_dict['value_init']).to(device)
            if self.correction_gmm is not None:
                #if self.correction_val_rep is not None:
                if (not hasattr(self, 'correction_val_rep')) and (self.validation_rep is not None):
                    self.correction_val_rep = RepresentationLayer(n_rep=self.correction_gmm.dim,
                        n_sample=checkpoint['validation_rep.z'].shape[0],
                        value_init=self.param_dict['value_init']).to(device)
            #else:
            #    print('careful! no validation representation has been initialized')
        if 'test_rep.z' in list(checkpoint.keys()):
            self.test_rep = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
                n_sample=checkpoint['test_rep.z'].shape[0],
                value_init=self.param_dict['value_init']).to(device)
            if self.correction_gmm is not None:
                self.correction_test_rep = RepresentationLayer(
                            n_rep=self.correction_gmm.dim,
                            n_sample=self.test_rep.z.shape[0],
                            value_init='zero').to(device)
                #if 'new_correction_model' in list(checkpoint.keys()):
        
        # dirty hack because I switched naming for one model and dataset
        if '_trained_status' in list(checkpoint.keys()):
            checkpoint['trained_status'] = checkpoint['_trained_status']
            self._trained_status = self.trained_status
        else:
            self._trained_status = None
        
        self.load_state_dict(checkpoint)
        
        print('#######################')
        print('Training status')
        print('#######################')
        print(self.trained_status.item())
    
    def save_param_dict_to_file(self, save_dir):
        #for x in self.param_dict.keys():
        #    if type(self.param_dict[x]) == list:
        #        print((x, [(type(self.param_dict[x][i]), sys.getsizeof(self.param_dict[x][i])) for i in range(len(self.param_dict[x]))]))
        #    else:
        #        print((x, type(self.param_dict[x]), sys.getsizeof(self.param_dict[x])))
        with open(save_dir+'_hyperparameters.json', 'w') as fp:
            json.dump(self.param_dict, fp)
        #with open(save_dir+'_hyperparameters.yml', 'w') as outfile:
        #    yaml.dump(self.param_dict, outfile, default_flow_style=False)
    
    def save_data_splits(self, data):
        # save as csv with the obs names
        df_out = data.obs.copy()
        df_out.to_csv(self._save_dir+'_obs.csv')
    
    @classmethod
    def load(cls,
            data,
            save_dir='./',
            random_seed = 0,
            model_name='dgd'
        ):

        # get saved hyper-parameters
        with open(save_dir+model_name+'_hyperparameters.json', 'r') as fp:
            param_dict = json.load(fp)
        scaling = param_dict['scaling']
        data.obs["train_val_test"] = load_data_splits(save_dir)

        # init model
        model = cls(
                data=data,
                parameter_dictionary=param_dict,
                scaling=scaling,
                save_dir=save_dir,
                random_seed = random_seed,
                model_name=model_name,
                print_outputs=False
            )
        
        model.load_parameters()
        return model
    
    def get_latent_representation(self, data=None):
        '''returning all representations of training data'''
        if data is not None:
            shape = data.shape
            if self.representation.z.shape[0] == shape:
                return self.representation.z.detach().cpu().numpy()
            elif self.validation_rep is not None:
                if self.validation_rep.z.shape[0] == shape:
                    return self.validation_rep.z.detach().cpu().numpy()
            elif self.test_rep is not None:
                if self.test_rep.z.shape[0] == shape:
                    return self.test_rep.z.detach().cpu().numpy()
        else:
            return self.representation.z.detach().cpu().numpy()
    
    def test(self, testdata=None, n_epochs=20, external=False):
        self.predict_new(testdata=testdata, n_epochs=n_epochs, external=False)

    def predict(self, testdata=None, n_epochs=20, external=False):
        self.predict_new(testdata=testdata, n_epochs=n_epochs, external=False)
        
    def init_test_set(self, testdata):
        '''initialize test set'''
        self.test_set = omicsDataset(testdata, split='test', scaling_type=self._scaling)
        ###
        #2do
        ###
        # the omicsDataset needs to create a mask if split=='test' and the data has a mask layer
        # which was prepared beforehand
        self.test_set.data_to_tensor()
    
    def predict_new(self, testdata=None, n_epochs=20, include_correction_error=True, indices_of_new_distribution=None, external=False):
        '''learn the embedding for new datapoints'''

        # prepare test set and loader
        if testdata is not None:
            self.init_test_set(testdata)
        else:
            self.test_set.data_to_tensor()
            if not hasattr(self, "test_set"):
                raise ValueError("No data was provided.")
        if self.test_set.sparse:
            test_loader = torch.utils.data.DataLoader(
                self.test_set, batch_size=8,
                shuffle=True, num_workers=0, collate_fn=collate_sparse_batches
                )
        else:
            test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=8,
            shuffle=True, num_workers=0
            )
        usable_features = None
        if external:
            usable_features = self.test_set.usable_features
        
        # train that representation
        self.test_rep, self.correction_test_rep, new_correction_model = learn_new_representations(
                self.gmm,
                self.decoder,
                test_loader,
                self.test_set.n_sample,
                self.correction_gmm,
                n_epochs=n_epochs,
                include_correction_error=include_correction_error,
                indices_of_new_distribution=indices_of_new_distribution),
                feature_ids=usable_features
        self.param_dict['test_representation'] = True
        # make the new_correction_model an attribute so it is learned
        self.save()
    
    def decoder_forward(self, rep_shape, i=None):
        if self.representation.z.shape[0] == rep_shape:
            if self.correction_gmm is not None:
                if i is not None:
                    z = self.representation(i)
                    z_correction = self.correction_rep(i)
                else:
                    z = self.representation.z
                    z_correction = self.correction_rep.z
                #z_correction = torch.cat(tuple([self.correction_models[corr_id][1](i) for corr_id in range(len(self.correction_models))]), dim=1)
                y = self.decoder(torch.cat((z, z_correction), dim=1))
            else:
                if i is not None:
                    z = self.representation(i)
                else:
                    z = self.representation.z
                y = self.decoder(z)
        elif self.test_rep is not None: # always preferrably use test rep
            if self.test_rep.z.shape[0] == rep_shape:
                if self.correction_gmm is not None:
                    if i is not None:
                        z = self.test_rep(i)
                        z_correction = self.correction_test_rep(i)
                    else:
                        z = self.test_rep.z
                        z_correction = self.correction_test_rep.z
                    #z_correction = torch.cat(tuple([self.test_correction[corr_id](i) for corr_id in range(len(self.correction_models))]), dim=1)
                    y = self.decoder(torch.cat((z, z_correction), dim=1))
                else:
                    if i is not None:
                        z = self.test_rep(i)
                    else:
                        z = self.test_rep.z
                    y = self.decoder(z)
        elif self.validation_rep is not None:
            if self.validation_rep.z.shape[0] == rep_shape:
                if self.correction_gmm is not None:
                    if i is not None:
                        z = self.validation_rep(i)
                        z_correction = self.correction_val_rep(i)
                    else:
                        z = self.validation_rep.z
                        z_correction = self.correction_val_rep.z
                    #z_correction = torch.cat(tuple([self.correction_models[corr_id][2](i) for corr_id in range(len(self.correction_models))]), dim=1)
                    y = self.decoder(torch.cat((z, z_correction), dim=1))
                else:
                    if i is not None:
                        z = self.validation_rep(i)
                    else:
                        z = self.validation_rep.z
                    y = self.decoder(z)
        else:
            raise ValueError('something is wrong with the shape of the dataset you supplied. There is no representation for it.')
        return y
    
    def predict_from_representation(self, rep, correction_rep=None):
        if self.correction_gmm is not None:
            z = rep.z
            z_correction = correction_rep.z
            y = self.decoder(torch.cat((z, z_correction), dim=1))
        else:
            z = rep.z
            y = self.decoder(z)
        return y
    
    def differential_expression(self):
        '''doing differential expression analysis'''
        return
    
    def perturbation_experiment(self):
        '''perform a perturbation of a given feature and examine downstream effects'''
        return
    
    def get_prediction_errors(self, predictions, dataset, reduction='sum'):
        '''returns errors of given predictions
        reduction: `sum`, `sample` or `none` (defines error shape)'''
        if reduction == 'sum':
            error = torch.zeros((1))
        elif reduction == 'sample':
            error = torch.zeros((predictions[0].shape[0]))
        elif reduction == 'gene':
            reduction = 'none'
            error = torch.zeros((predictions[0].shape[0],predictions[0].shape[1]+predictions[1].shape[1]))
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32,
            shuffle=True, num_workers=0
            )
        # x has to be broght into correct shape depending on number of modalities
        for x,lib,i in data_loader:
            #print('mini batch')
            if self.decoder.n_out_groups == 2:
                y = [predictions[0][i,:], predictions[1][i,:]]
                x = [x[:,:self.param_dict['modality_switch']],x[:,self.param_dict['modality_switch']:]]
                library = [lib[:,0].unsqueeze(1), lib[:,1].unsqueeze(1)]
            elif self.decoder.n_out_groups == 1:
                x = [x]
                library = [lib]
            else:
                raise ValueError('number of modalities currently not supported')
            if reduction != 'none':
                temp_error = self.decoder.loss(
                    y,
                    x,
                    scale=library,
                    reduction=reduction
                )
                if reduction == 'sum':
                    error += temp_error
                elif reduction == 'sample':
                    error[i] = temp_error
            else:
                temp_error_1 = self.decoder.loss(
                    y[0],
                    x[0],
                    scale=library[0],
                    reduction=reduction,
                    mod_id=0
                )
                temp_error_2 = self.decoder.loss(
                    y[1],
                    x[1],
                    scale=library[1],
                    reduction=reduction,
                    mod_id=1
                )
                error[i,:] = torch.cat((temp_error_1,temp_error_2),dim=1)
        return error
    
    def get_normalized_expression(self, dataset, indices):
        '''
        given a representation for the dataset is learned, returns the normalized (unscaled) model output
        currently only implemented for multiome
        '''
        # get the dimensions of the dataset
        shape = dataset.X.shape[0]
        return self.decoder_forward(shape, indices)[0]
    
    def get_accessibility_estimates(self, dataset, indices):
        '''
        given a representation for the dataset is learned, returns the normalized (unscaled) model output
        currently only implemented for multiome
        '''
        # get the dimensions of the dataset
        shape = dataset.X.shape[0]
        return self.decoder_forward(shape, indices)[1]
    
    def get_modality_reconstruction(self, dataset, mod_id):
        shape = dataset.X.shape[0]
        predictions = self.decoder_forward(shape, torch.arange(shape))[mod_id]
        return predictions*self.library[mod_id].unsqueeze(0)
    
    def get_reconstruction(self, dataset):
        shape = dataset.X.shape[0]
        predictions = self.decoder_forward(shape, torch.arange(shape))
        return predictions
    
    def view_data_setup(self):
        print("The prepared data consists of the following training set:")
        print(self.train_set)
        if hasattr(self, "val_set"):
            print("There is also a validation set with {} samples".format(self.val_set.n_sample))
        if hasattr(self, "test_set"):
            print("And a test set with {} samples".format(self.test_set.n_sample))
    
    def get_representation(self, split='train'):
        if split == 'train':
            return self.representation.z.detach().cpu().numpy()
        elif split == 'validation':
            return self.validation_rep.z.detach().cpu().numpy()
        elif split == 'test':
            return self.test_rep.z.detach().cpu().numpy()
        elif split == 'all':
            # create a zero numpy array and fill it with all representations according to the data split
            latent = np.zeros((self.total_cells, self.latent))
            latent[self.train_indices,:] = self.representation.z.detach().cpu().numpy()
            latent[self.validation_indices,:] = self.validation_rep.z.detach().cpu().numpy()
            latent[self.test_indices,:] = self.test_rep.z.detach().cpu().numpy()
            return latent
    
    def clustering(self, split='train'):
        if split == 'train':
            return self.gmm.clustering(self.representation.z.detach()).detach().cpu().numpy().astype(int)
        elif split == 'validation':
            return self.gmm.clustering(self.validation_rep.z.detach()).detach().cpu().numpy().astype(int)
        elif split == 'test':
            return self.gmm.clustering(self.test_rep.z.detach()).detach().cpu().numpy().astype(int)
        elif split == 'all':
            # create a zero numpy array and fill it with all representations according to the data split
            latent = np.zeros(self.total_cells)
            latent[self.train_indices] = self.gmm.clustering(self.representation.z.detach()).detach().cpu().numpy()
            latent[self.validation_indices] = self.gmm.clustering(self.validation_rep.z.detach()).detach().cpu().numpy()
            latent[self.test_indices] = self.gmm.clustering(self.test_rep.z.detach()).detach().cpu().numpy()
            return latent.astype(int)
        
    def plot_history(self, export=False):
        '''
        If executed in a jupyter notebook, plots the training history of the model.
        Otherwise provide an export path to save the plot.
        '''
        import matplotlib.pyplot as plt
        import seaborn as sns

        # got error raise ValueError("cannot reindex on an axis with duplicate labels")
        # so reindex the history
        history = self.history.reset_index()

        sns.lineplot(
            data=history,
            #data=self.history,
            x='epoch', y='reconstruction_loss', hue='split'
        )
        if export:
            plt.savefig(self._save_dir+self._model_name+'_traininghistory.png')
        else:
            plt.show()
    
    def plot_latent_space(self):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        palette = ['#58A6A6', '#EFA355', '#421E22']
        # get the train rep
        rep = self.representation.z.detach().cpu().numpy()
        # do a pca
        pca = PCA(n_components=2)
        pca.fit(rep)
        # transform
        rep_pca = pca.transform(rep)
        # get the gmm means
        gmm_means = self.gmm.mean.detach().cpu().numpy()
        # transform
        gmm_means_pca = pca.transform(gmm_means)
        # also do samples from the means
        gmm_samples = self.gmm.sample(int(self.train_set.n_sample/10)).detach().cpu().numpy()
        gmm_samples_pca = pca.transform(gmm_samples)
        # plot
        plt.scatter(rep_pca[:,0], rep_pca[:,1], c=palette[0], s=1, alpha=0.7, label='representation')
        plt.scatter(gmm_samples_pca[:,0], gmm_samples_pca[:,1], c=palette[1], s=1, alpha=0.7, label='GMM samples')
        plt.scatter(gmm_means_pca[:,0], gmm_means_pca[:,1], c=palette[2], s=10, alpha=1, label='GMM means')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Latent space PCA')
        plt.legend()
        plt.show()
    
    def gene2peak(self, gene_name, testset, gene_ref=None):
        if gene_name not in testset.var_names:
            if gene_ref is not None:
                if isinstance(gene_ref, str):
                    gene_ref = testset.var[gene_ref]
                else:
                    raise ValueError("gene_ref must be a string")
            else:
                raise ValueError("Gene was not found in data. It might be spelled wrong or you need to specify the var column name that contains the gene name.")
        else:
            gene_ref = testset.var_names
        gene_id = np.where(gene_ref == gene_name)[0][0]
        predictions_original_gene, predictions_upregulated_gene, indices_of_interest_gene = predict_perturbations(self, testset, gene_id)
        predicted_changes = [(predictions_upregulated_gene[i] - predictions_original_gene[i]) for i in range(len(self.train_set.modalities))]
        return predicted_changes, indices_of_interest_gene


def load_data_splits(save_dir):
    df_in = pd.read_csv(save_dir+'_obs.csv')
    return df_in["train_val_test"].values

default_parameters = {
            'latent_dimension': 20,
            'n_components': 1,
            'n_hidden': 2,
            'n_hidden_modality': 3,
            'n_units': 100,
            'value_init': 'zero', # options are zero or handed values
            'softball_scale': 2,
            'softball_hardness': 5,
            'sd_sd': 1,
            'softball_scale_corr': 2,
            'softball_hardness_corr': 5,
            'sd_sd_corr': 1,
            'dirichlet_a': 1,
            'batch_size': 128,
            'learning_rates': [1e-4, 1e-2, 1e-2],
            'betas': [0.5,0.7],
            'weight_decay': 1e-4,
            'decoder_width': 1,
            'log_wandb': ['username', 'projectname']
        }

###
# functions used for the sparse option
# here the data will have to be transformed to dense format when a batch is called
###

def sparse_coo_to_tensor(mtrx):
    return torch.FloatTensor(mtrx.todense())

def collate_sparse_batches(batch):
    data_batch, library_batch, idx_batch = zip(*batch)
    data_batch = scipy.sparse.vstack(list(data_batch))
    data_batch = sparse_coo_to_tensor(data_batch)
    library_batch = torch.stack(list(library_batch), dim=0)
    idx_batch = list(idx_batch)
    return data_batch, library_batch, idx_batch
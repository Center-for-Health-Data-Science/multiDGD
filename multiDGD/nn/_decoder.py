import torch
import torch.nn as nn
import numpy as np

from multiDGD.nn._outputmodules import OutputModule

class Decoder(nn.Module):
    '''
    Class for the DGD decoder with output modules modelling data 
    according to a modality-specific distribution.

    Attributes
    ----------
    main: torch.nn.modules.container.ModuleList
        the decoder portion shared between modalities (if multiple available)
        starts at representation and gives output to out_modules
    out_modules: torch.nn.modules.container.ModuleList
        modules taking into account a specific distribution for calculating loss
        can include more linear layers

    Methods
    ----------
    forward(z)
    loss(nn_output, target, scale=None, mod_id=None, gene_id=None, reduction='sum')
    log_prob(nn_output, target, scale=None, mod_id=None, gene_id=None, reduction='sum')
    '''

    def __init__(self, in_features: int, parameter_dictionary: dict):
        super(Decoder, self).__init__()

        # initialize main decoder body and output modules
        self.main = nn.ModuleList()
        self.out_modules = nn.ModuleList()

        # get parameters for number of hidden layers and units
        n_hidden = parameter_dictionary['n_hidden']
        n_hidden_modality = parameter_dictionary['n_hidden_modality']
        if n_hidden > 0: n_last_hidden = parameter_dictionary['n_units']
        else: n_last_hidden = in_features
        
        # add hidden layers to main body
        if n_hidden > 0:
            for i in range(n_hidden):
                if i == 0: start = in_features
                else: start = parameter_dictionary['n_units']
                if i == (n_hidden-1): stop = n_last_hidden
                else: stop = parameter_dictionary['n_units']
                self.main.append(nn.Linear(start,stop))
                self.main.append(nn.ReLU(True))
        
        # initialize last layers in output modules (contain output distribtutions)
        n_features = parameter_dictionary['n_features_per_modality']
        modalities = parameter_dictionary['modalities']
        for i, modality in enumerate(modalities):
            modality = modalities[i]
            self.out_modules.append(
                OutputModule(
                    in_features=n_last_hidden,
                    out_features=n_features[i],
                    n_hidden=n_hidden_modality,
                    hidden_features=parameter_dictionary['n_units'],
                    modality=modality,
                    layer_width=parameter_dictionary['decoder_width']
                    )
                )
        self.n_out_groups = len(modalities)

        # add compatibility with shap analysis
        self.shap_compatible = False
    
    def forward(self, z):
        for i in range(len(self.main)):
            z = self.main[i](z)
        if not self.shap_compatible:
            out = [outmod(z) for outmod in self.out_modules]
        else:
            out = torch.cat([outmod.forward_shap(z) for outmod in self.out_modules], dim=-1)
        return out
    
    def loss(self, nn_output, target, scale=None, mod_id=None, gene_id=None, reduction='sum', mask=None):
        return (-self.log_prob(nn_output, target, scale, mod_id, gene_id, reduction, mask))
    
    def log_prob(self, nn_output, target, scale=None, mod_id=None, gene_id=None, reduction='sum', mask=None):
        '''
        calculating the log probability
        
        This function is providied with different options of how the output should be provided.
        All log-probs can be summed into one value (reduction == 'sum'), summed per sample ('sample')
        or not reduced at all (thus giving a log-prob per feature per sample).

        mask has been added to support training of mosaic data, where a modality can be missing
        For now I only added it to the base loss computation where reduction is sum
        '''
        if reduction == 'sum':
            log_prob = 0.
            if mod_id is not None:
                if scale is None:
                    log_prob += self.out_modules[mod_id].log_prob(nn_output,target,gene_id=gene_id,mask=mask).sum()
                else:
                    log_prob += self.out_modules[mod_id].log_prob(nn_output,target,scale,gene_id=gene_id,mask=mask).sum()
            else:
                if scale is None:
                    if mask is None:
                        for i in range(self.n_out_groups):
                            log_prob += self.out_modules[i].log_prob(nn_output[i],target[i]).sum()
                    else:
                        for i in range(self.n_out_groups):
                            log_prob += self.out_modules[i].log_prob(nn_output[i],target[i],mask=mask[i]).sum()
                else:
                    if mask is None:
                        for i in range(self.n_out_groups):
                            log_prob += self.out_modules[i].log_prob(nn_output[i],target[i],scale[i]).sum()
                    else:
                        for i in range(self.n_out_groups):
                            log_prob += self.out_modules[i].log_prob(nn_output[i],target[i],scale[i],mask=mask[i]).sum()
        elif reduction == 'sample':
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if mod_id is not None:
                #log_prob = torch.zeros(nn_output.shape[:-1]).to(dev)
                if scale is None:
                    log_prob = self.out_modules[mod_id].log_prob(nn_output,target,gene_id=gene_id).sum(-1)
                else:
                    log_prob = self.out_modules[mod_id].log_prob(nn_output,target,scale,gene_id=gene_id).sum(-1)
            else:
                #log_prob = torch.zeros(nn_output[0].shape[:-1]).to(dev)
                if scale is None:
                    for i in range(self.n_out_groups):
                        if i == 0:
                            log_prob = self.out_modules[i].log_prob(nn_output[i],target[i]).sum(-1)
                        else:
                            log_prob += self.out_modules[i].log_prob(nn_output[i],target[i]).sum(-1)
                else:
                    for i in range(self.n_out_groups):
                        if i == 0:
                            log_prob = self.out_modules[i].log_prob(nn_output[i],target[i],scale[i]).sum(-1)
                        else:
                            log_prob += self.out_modules[i].log_prob(nn_output[i],target[i],scale[i]).sum(-1)
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if mod_id is not None:
                log_prob = torch.zeros(nn_output.shape).to(dev)
                if scale is None:
                    log_prob += self.out_modules[mod_id].log_prob(nn_output,target,gene_id=gene_id)
                else:
                    log_prob += self.out_modules[mod_id].log_prob(nn_output,target,scale,gene_id=gene_id)
            else:
                if len(self.out_modules) > 1:
                    raise ValueError('This combination does not work. No loss reduction can only be done for single modalities')
                else:
                    log_prob = torch.zeros(nn_output[0].shape[:-1]).to(dev)
                    if scale is None:
                        for i in range(self.n_out_groups):
                            log_prob += self.out_modules[i].log_prob(nn_output[i],target[i])
                    else:
                        for i in range(self.n_out_groups):
                            log_prob += self.out_modules[i].log_prob(nn_output[i],target[i],scale[i])
        return log_prob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from statsmodels.stats.multitest import multipletests

class OutputModule(nn.Module):
    '''
    This is the basis output module class that stands between the decoder and the output data.

    Arguments
    ----------
    in_features: int
        number of features going into this layer
    out_features: int
        number of features that come out of this layer
    n_hidden: int
        number of hidden layers
    hidden_features: int
        number of features in hidden layers
    modality: str
        modality of the data
    layer_width: int
        width of the output layers (as a multiplier for n_units)

    Attributes
    ----------
    fc: torch.nn.modules.container.ModuleList
    n_in: int
        number of hidden units going into this layer
    n_out: int
        number of features that come out of this layer
    distribution: torch.nn.modules.module.Module
        specific class depends on modality argument

    Methods
    ----------
    '''

    def __init__(self,
        in_features: int, 
        out_features: int, 
        n_hidden: int, 
        hidden_features: int,
        modality: str,
        layer_width: int
    ):
        super(OutputModule, self).__init__()

        # init modulelist for (potentially) continued decoder
        self.fc = nn.ModuleList()
        # determine in and out feature sizes
        self.n_in = in_features
        self.n_out = out_features

        if n_hidden > 0:
            for i in range(n_hidden):
                start = layer_width * hidden_features
                stop = layer_width * hidden_features
                if i == (n_hidden-1):
                    start = layer_width * max(hidden_features, round(np.sqrt(out_features))) # was just round
                    stop = out_features
                if i == 0: # this was originally the first condition but made problems with layer depth 1
                    start = in_features
                if i == (n_hidden-2):
                    stop = layer_width * max(hidden_features, round(np.sqrt(out_features))) # was just round
                self.fc.append(nn.Linear(start,stop))
                if i < (n_hidden-1):
                    self.fc.append(nn.ReLU(True))
        else:
            self.fc.append(nn.Linear(in_features,out_features))
        
        if modality in ['rna', 'RNA', 'gex', 'GEX', 'atac', 'ATAC']:
            self.distribution = NB_Layer(out_features)
        else:
            error_message = 'unknown modality provided. can only accept "rna" or "atac" at the moment.\n'
            error_message += 'contact developer if you would like your modality to be included'
            raise ValueError(error_message)

    def forward(self,x):
        '''forward pass through the output module'''
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return self.distribution(x)
    
    def forward_shap(self,x):
        '''placeholder for SHAP compatibility'''
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return x
    
    def loss(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        '''returns loss of the output module'''
        return self.distribution.loss(model_output,target,scaling_factor,gene_id,mask)
    
    def log_prob(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        '''returns log-prob of the output module'''
        return self.distribution.log_prob(model_output,target,scaling_factor,gene_id,mask)

def logNBdensity(k,m,r):
    ''' 
    Negative Binomial NB(k;m,r), where m is the mean and k is "number of failures"
    Here, k is the observed count and m is the predicted mean of the NB. r is the dispersion parameter
    k, and m are tensors of same shape
    r is tensor of shape (1, n_genes)
    Returns the log NB in same shape as k
    '''
    # remember that gamma(n+1)=n!
    eps = 1.e-10 # this is under-/over-flow protection
    x = torch.lgamma(k+r)
    x -= torch.lgamma(r)
    x -= torch.lgamma(k+1)
    inverse_rme = 1/(r+m+eps)
    x = x + k.mul(torch.log(m*inverse_rme+eps))
    x = x + r*torch.log(r*inverse_rme)
    return x

class NB_Layer(nn.Module):
    '''
    This is the Negative Binomial version of the OutputModule distribution layer.

    Arguments
    ----------
    out_features: int
        number of features that come out of this layer
    r_init: int
        initial value for the log-dispersion parameter
    scaling_type: str
        type of scaling to be applied to the output

    Attributes
    ----------
    fc: torch.nn.modules.container.ModuleList
    log_r: torch.nn.parameter.Parameter
        log-dispersion parameter per feature
    dispersion: torch.nn.parameter.Parameter
        dispersion parameter per feature

    Methods
    ----------
    '''
    def __init__(self, out_features, r_init=2, scaling_type='sum'):
        super(NB_Layer, self).__init__()

        # substracting 1 now and adding it to the learned dispersion ensures a minimum value of 1
        self.log_r = torch.nn.Parameter(torch.full(fill_value=math.log(r_init-1), 
            size=(1,out_features)), 
            requires_grad=True)
        self._scaling_type = scaling_type
        if self._scaling_type == 'sum': # could later re-implement more scalings, but sum is arguably the best so far
            self._activation = 'softmax'
    
    def forward(self, x):
        '''forward pass through the NB layer'''
        return F.softmax(x,dim=-1)
    
    @staticmethod
    def rescale(scaling_factor,model_output):
        '''rescales the model output (mean normalized count)'''
        return (scaling_factor*model_output)
    
    def log_prob(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        '''returns the log-prob of the NB layer'''
        # the model output represents the mean normalized count
        # the scaling factor is the used normalization
        if gene_id is not None:
            logprob = logNBdensity(target[:,gene_id],self.rescale(scaling_factor,model_output)[:,gene_id],(torch.exp(self.log_r)+1)[0,gene_id])
        else:
            logprob = logNBdensity(target,self.rescale(scaling_factor,model_output),(torch.exp(self.log_r)+1))
        if mask is not None:
            logprob[mask,:] = 0 # mask must be a boolean 1D tensor
        return logprob
    
    def norm_abs_error(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        '''returns the normalized absolute error of the NB layer'''
        if gene_id is not None:
            error = torch.abs(target[:,gene_id] - self.rescale(scaling_factor,model_output)[:,gene_id])/target[:,gene_id]
        else:
            error = torch.abs(target - self.rescale(scaling_factor,model_output))/target
        if mask is not None:
            error[mask,:] = 0 # mask must be a boolean 1D tensor
        return error
    
    def loss(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        '''returns the loss of the NB layer'''
        neglogprob = - self.log_prob(model_output,target,scaling_factor,gene_id,mask=None)
        return neglogprob
    
    @property
    def dispersion(self):
        '''returns the dispersion parameter'''
        return (torch.exp(self.log_r)+1)
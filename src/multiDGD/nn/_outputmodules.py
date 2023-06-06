import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from statsmodels.stats.multitest import multipletests

class OutputModule(nn.Module):
    '''
    This is the basis output module class that stands between the decoder and the output data.

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
    forward(x)
        input goes through fc modulelist and distribution layer
    loss(model_output,target,scaling_factor,gene_id=None)
        returns loss of distribution for given output
    log_prob(model_output,target,scaling_factor,gene_id=None)
        returns log-prob of distribution for given output
    '''

    def __init__(self,
        in_features: int, 
        out_features: int, 
        n_hidden: int, 
        hidden_features: int,
        modality: list,
        layer_width: int
    ):
        super(OutputModule, self).__init__()

        # init modulelist for (potentially) continued decoder
        self.fc = nn.ModuleList()
        # determine in and out feature sizes
        self.n_in = in_features
        self.n_out = out_features

        if n_hidden > 0:
            #if n_hidden == 1:
            #    start = hidden_features
            #    stop = out_features
            #    self.fc.append(nn.Linear(start,stop))
            #else:
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
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return self.distribution(x)
    
    def forward_shap(self,x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return x
    
    def loss(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        return self.distribution.loss(model_output,target,scaling_factor,gene_id,mask)
    
    def log_prob(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        return self.distribution.log_prob(model_output,target,scaling_factor,gene_id,mask)
    
    ##################################################
    '''This section is for differential expression analysis'''
    ##################################################

    def group_probs(self,target,model_output,scaling_factor,return_style='mean',feature_ids=None):
        ''' computes the mean densities per gene for a group (or single sample)
        this is done rather than calculating the probability of an average mean and expression value
        because it might be applied to groups with a lot of heterogeneiety
        for which the probabilities are expected to be less prone to outliers
        '''
        #print('using general group_probs')
        return torch.exp(self.distribution.group_probs(target,model_output,scaling_factor,return_style,feature_ids))
        
    def p_vals(self,x,mhat,M,mhat_case,x_case=None,M_case=None,feature_ids=None):
        '''
        this function will compute p values for each gene between two groups
        
        the groups can be real case-control groups or perturbed-normal groups,
        for which there is no x_case or M_case

        if there are more than 20k features, the p values are computed in batches
        because we need to compute an n*n matrix of the sum of probabilities smaller
        than the probability of the gene of interest, which is too large for memory
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_cutoff = 20000
        batch_s = 5000
        # get mean probabilities of each gene per group
        probs_control = self.group_probs(x,mhat,M,return_style="mean")
        if x_case is not None:
            # if there are actual case samples, compute mean probabilities wrt them
            probs_case = self.group_probs(x_case,mhat_case,M_case,return_style="mean")
        else:
            # otherwise compute mean probabilities wrt control samples
            # this is used in perturbations
            #probs_case = self.group_probs(mhat_case,x,M,return_style="sample",feature_ids=feature_ids)
            probs_case = self.group_probs(mhat_case*M,mhat,M,return_style="mean")
        print("probs_control",probs_control.shape)
        print("probs_case",probs_case.shape)
        # if assuming independence between observations, p(a,b)=p(a)p(b)
        paired_probs = torch.mul(probs_control,probs_case)
        print("paired_probs",paired_probs.shape)
        # as in DEseq, p values are computed as the sum of paired probs smaller than 
        # that of the gene of interest normalized by the sum of all paired probs
        # for memory reasons, we check if there are more than 20k features
        #if paired_probs.shape[0] <= feature_cutoff:
        if probs_case.shape[0] <= feature_cutoff:
            #if chi_squared.shape[0] <= feature_cutoff:
            # we thus search for the indices of the paired probs smaller than each of the features
            # and sum them up
            #p_vals_unadjusted = torch.where(chi_squared.unsqueeze(1) < chi_squared, chi_squared.unsqueeze(1), 0).sum(0)
            #p_vals_unadjusted = torch.where(probs_case.unsqueeze(1) < probs_case, probs_case.unsqueeze(1), 0).sum(0)
            p_vals_unadjusted = torch.where(paired_probs.unsqueeze(1) < paired_probs, paired_probs.unsqueeze(1), 0).sum(0)
        else:
            # if the number of features is too high,
            # we compute the p values in batches of 1000
            p_vals_unadjusted = torch.zeros(paired_probs.shape[0]).to(device)
            #p_vals_unadjusted = torch.zeros(probs_case.shape[0]).to(device)
            #p_vals_unadjusted = torch.zeros(chi_squared.shape[0]).to(device)
            #for i in range(0,chi_squared.shape[0],batch_s):
            #    p_vals_unadjusted[i:i+batch_s] = torch.where(chi_squared.unsqueeze(1) < chi_squared[i:i+batch_s], chi_squared.unsqueeze(1), 0).sum(0)
            for i in range(0,paired_probs.shape[0],batch_s):
                p_vals_unadjusted[i:i+batch_s] = torch.where(paired_probs.unsqueeze(1) < paired_probs[i:i+batch_s], paired_probs.unsqueeze(1), 0).sum(0)
            #for i in range(0,probs_case.shape[0],batch_s):
            #    p_vals_unadjusted[i:i+batch_s] = torch.where(probs_case.unsqueeze(1) < probs_case[i:i+batch_s], probs_case.unsqueeze(1), 0).sum(0)
        # then we divide by the sum of all paired probs
        p_vals = p_vals_unadjusted / paired_probs.sum()
        #p_vals = p_vals_unadjusted / probs_case.sum()
        #p_vals = p_vals_unadjusted / chi_squared.sum()
        return p_vals#.mean(0)
    
    @staticmethod
    def fdr(p_vals,alpha=0.05):
        '''computes the false discovery rate with adjusted p-values'''
        rejected,padj,_,_ = multipletests(pvals=p_vals,alpha=alpha,method ="fdr_bh")
        return rejected, padj
    
    @staticmethod
    def log2_fold_change(case,control):
        '''log fold change between control and case
        a negative log fold change means underexpression, a positive overexpression
        '''
        eps = 1e-6 # correction for 0 values
        return torch.log2((case+eps)/(control+eps))
    
    def differential_expression_analysis(self,x,mhat,M,mhat_case,x_case=None,M_case=None):
        '''putting all functions above together for easy usage'''

        # first comoute p values
        p_vals = self.p_vals(x,mhat,M,mhat_case,x_case,M_case)
        # next rescale model output to predictions
        predictions_control = self.distribution.rescale(M,mhat).detach().cpu()
        if M_case is not None:
            predictions_case = self.distribution.rescale(M_case,mhat_case).detach().cpu()
        else:
            predictions_case = self.distribution.rescale(M,mhat_case).detach().cpu()
        # now compute log2-fold changes between grouped predictions
        log_fold_changes = self.log2_fold_change(predictions_case,predictions_control)
        # calculate adjusted p values
        rejected, padj = self.fdr(p_vals)

        return p_vals, log_fold_changes, rejected, padj

"""
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
    #x += k*torch.log(m*(r+m+eps)**(-1)+eps)
    #x += r*torch.log(r*(r+m+eps)**(-1))
    inverse_rme = 1/(r+m+eps)
    x += k*torch.log(m*inverse_rme+eps)
    x += r*torch.log(r*inverse_rme)
    return x
"""

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

    Attributes
    ----------
    fc: torch.nn.modules.container.ModuleList
    log_r: torch.nn.parameter.Parameter
        log-dispersion parameter per feature

    Methods
    ----------
    forward(x)
        applies scaling-specific activation
    loss(model_output,target,scaling_factor,gene_id=None)
        returns loss of NB for given output
    log_prob(model_output,target,scaling_factor,gene_id=None)
        returns log-prob of NB for given output
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
        return F.softmax(x,dim=-1)
        #return F.softplus(x)
        #return torch.sigmoid(x)
    
    @staticmethod
    def rescale(scaling_factor,model_output):
        return (scaling_factor*model_output)
    
    def log_prob(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        # the model output represents the mean normalized count
        # the scaling factor is the used normalization
        if gene_id is not None:
            logprob = logNBdensity(target[:,gene_id],self.rescale(scaling_factor,model_output)[:,gene_id],(torch.exp(self.log_r)+1)[0,gene_id])
        else:
            logprob = logNBdensity(target,self.rescale(scaling_factor,model_output),(torch.exp(self.log_r)+1))
        if mask is not None:
            logprob[mask,:] = 0 # mask must be a boolean 1D tensor
        return logprob# - self.norm_abs_error(model_output,target,scaling_factor,gene_id,mask)
    
    def norm_abs_error(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        if gene_id is not None:
            error = torch.abs(target[:,gene_id] - self.rescale(scaling_factor,model_output)[:,gene_id])/target[:,gene_id]
        else:
            error = torch.abs(target - self.rescale(scaling_factor,model_output))/target
        if mask is not None:
            error[mask,:] = 0 # mask must be a boolean 1D tensor
        return error
    
    def loss(self,model_output,target,scaling_factor,gene_id=None,mask=None):
        neglogprob = - self.log_prob(model_output,target,scaling_factor,gene_id,mask=None)
        #norm_abs_error = self.norm_abs_error(model_output,target,scaling_factor,gene_id,mask=None)
        return neglogprob# + norm_abs_error
    
    @property
    def dispersion(self):
        return (torch.exp(self.log_r)+1)
    
    def group_probs(self,target,model_output,scaling_factor,return_style='mean',feature_ids=None):
        """ computes the mean densities per gene for a group (or single sample)
        this is done rather than calculating the probability of an average mean and expression value
        because it might be applied to groups with a lot of heterogeneiety
        for which the probabilities are expected to be less prone to outliers
        """
        #print('using NB group_probs')
        #print('model output shape: ',model_output.shape)
        #print('target shape: ',target.shape)
        #print('scaling factor shape: ',scaling_factor.shape)
        #print('dispersion shape before: ',self.dispersion.shape)
        if len(model_output.shape) < 2:
            #print('A', model_output.shape[0], self.dispersion.shape[1])
            if model_output.shape[0] != self.dispersion.shape[0]:
                if feature_ids is not None:
                    dispersion = self.dispersion[:,feature_ids]#.unsqueeze(1)
                else:
                    raise ValueError('model output has wrong shape or feature ids were not provided')
        elif model_output.shape[1] != self.dispersion.shape[1]:
            #print('B', model_output.shape[1], self.dispersion.shape[1])
            if feature_ids is not None:
                dispersion = self.dispersion[:,feature_ids]#.unsqueeze(1)
            else:
                raise ValueError('model output has wrong shape or feature ids were not provided')
        else:
            dispersion = self.dispersion
        #print('dispersion shape: ',dispersion.shape)
        if return_style == 'mean':
            return torch.exp(logNBdensity(target,self.rescale(scaling_factor,model_output),dispersion)).mean(0)
        else:
            return torch.exp(logNBdensity(target,self.rescale(scaling_factor,model_output),dispersion))
    
    def p_vals(self,x,mhat,M,mhat_case,x_case=None,M_case=None):
        '''
        this function will compute p values for each gene between two groups
        
        the groups can be real case-control groups or perturbed-normal groups,
        for which there is no x_case or M_case

        if there are more than 20k features, the p values are computed in batches
        because we need to compute an n*n matrix of the sum of probabilities smaller
        than the probability of the gene of interest, which is too large for memory
        '''
        print("using NB p_vals")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_cutoff = 20000
        batch_s = 5000
        # get mean probabilities of each gene per group
        probs_control = self.group_probs(mhat,x,M)
        if x_case is not None:
            # if there are actual case samples, compute mean probabilities wrt them
            probs_case = self.group_probs(mhat_case,x_case,M_case)
        else:
            # otherwise compute mean probabilities wrt control samples
            # this is used in perturbations
            probs_case = self.group_probs(mhat_case,x,M)
        # if assuming independence between observations, p(a,b)=p(a)p(b)
        paired_probs = torch.mul(probs_control,probs_case)
        # as in DEseq, p values are computed as the sum of paired probs smaller than 
        # that of the gene of interest normalized by the sum of all paired probs
        # for memory reasons, we check if there are more than 20k features
        if paired_probs.shape[0] <= feature_cutoff:
            # we thus search for the indices of the paired probs smaller than each of the features
            # and sum them up
            p_vals_unadjusted = torch.where(paired_probs.unsqueeze(1) < paired_probs, paired_probs.unsqueeze(1), 0).sum(0)
        else:
            # if the number of features is too high,
            # we compute the p values in batches of 1000
            p_vals_unadjusted = torch.zeros(paired_probs.shape[0]).to(device)
            for i in range(0,paired_probs.shape[0],batch_s):
                p_vals_unadjusted[i:i+batch_s] = torch.where(paired_probs.unsqueeze(1) < paired_probs[i:i+batch_s], paired_probs.unsqueeze(1), 0).sum(0)
        # then we divide by the sum of all paired probs
        p_vals = p_vals_unadjusted / paired_probs.sum()
        return p_vals
    
    @staticmethod
    def fdr(p_vals,alpha=0.05):
        """
        computes the false discovery rate with adjusted p-values
        """
        rejected,padj,_,_ = multipletests(pvals=p_vals,alpha=alpha,method ="fdr_bh")
        return rejected, padj
    
    @staticmethod
    def log2_fold_change(case,control):
        """log fold change between control and case
        a negative log fold change means underexpression, a positive overexpression
        """
        eps = 1e-6 # correction for 0 values
        return torch.log2((case+eps)/(control+eps))
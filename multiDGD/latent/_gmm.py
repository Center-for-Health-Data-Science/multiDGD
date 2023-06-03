import math
import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np

class gaussian():
    '''
    This is a simple Gaussian prior used for initializing mixture model means

    Attributes
    ----------
    dim: int
        dimensionality of the space in which samples live
    mean: float
        value of the intended mean of the Normal distribution
    stddev: float
        value of the intended standard deviation

    Methods
    ----------
    sample(n)
        generates samples from the prior
    log_prob(z)
        returns log probability of a vector
    '''

    def __init__(self, dim: int ,mean: float ,stddev: float):
        self.dim = dim
        self.mean = mean
        self.stddev = stddev
        self._distrib = torch.distributions.normal.Normal(mean,stddev)
    
    def sample(self,n):
        return self._distrib.sample((n, self.dim))
    
    def log_prob(self,x):
        return self._distrib.log_prob(x)

class softball():
    '''
    Approximate mollified uniform prior.
    It can be imagined as an m-dimensional ball.

    The logistic function creates a soft (differentiable) boundary.
    The prior takes a tensor with a batch of z
    vectors (last dim) and returns a tensor of prior log-probabilities.
    The sample function returns n samples from the prior (approximate
    samples uniform from the m-ball). NOTE: APPROXIMATE SAMPLING.

    Attributes
    ----------
    dim: int
        dimensionality of the space in which samples live
    radius: int
        radius of the m-ball
    sharpness: int
        sharpness of the differentiable boundary

    Methods
    ----------
    sample(n)
        generates samples from the prior with APPROXIMATE sampling
    log_prob(z)
        returns log probability of a vector
    '''

    def __init__(self, dim: int, radius: int, sharpness=1):
        self.dim = dim
        self.radius = radius
        self.sharpness = sharpness
        self._norm = math.lgamma(1+dim*0.5)-dim*(math.log(radius)+0.5*math.log(math.pi))
    
    def sample(self,n):
        '''APPROXIMATE sampling'''
        # Return n random samples
        # Approximate: We sample uniformly from n-ball
        with torch.no_grad():
            # Gaussian sample
            sample = torch.randn((n,self.dim))
            # n random directions
            sample.div_(sample.norm(dim=-1,keepdim=True))
            # n random lengths
            local_len = self.radius*torch.pow(torch.rand((n,1)),1./self.dim)
            sample.mul_(local_len.expand(-1,self.dim))
        return sample
    
    def log_prob(self,z):
        # Return log probabilities of elements of tensor (last dim assumed to be z vectors)
        return (self._norm-torch.log(1+torch.exp(self.sharpness*(z.norm(dim=-1)/self.radius-1))))

class GaussianMixture(nn.Module):
    '''
    A mixture of multi-variate Gaussians.

    m_mix_comp is the number of components in the mixture
    dim is the dimension of the space
    covariance_type can be "fixed", "isotropic" or "diagonal"
    the mean_prior is initialized as a softball (mollified uniform) with
        mean_init(<radius>, <hardness>)
    neglogvar_prior is a prior class for the negative log variance of the mixture components
        - neglogvar = log(sigma^2)
        - If it is not specified, we make this prior a Gaussian from sd_init parameters
        - For the sake of interpretability, the sd_init parameters represent the desired mean and (approximately) sd of the standard deviation
        - the difference btw giving a prior beforehand and giving only init values is that with a given prior, the neglogvar will be sampled from it, otherwise they will be initialized the same
    alpha determines the Dirichlet prior on mixture coefficients
    Mixture coefficients are initialized uniformly
    Other parameters are sampled from prior

    Attributes
    ----------
    dim: int
        dimensionality of the space
    n_mix_comp: int
        number of mixture components
    mean: torch.nn.parameter.Parameter
        learnable parameter for the GMM means with shape (n_mix_comp,dim)
    neglogvar: torch.nn.parameter.Parameter
        learnable parameter for the negative log-variance of the components
        shape depends on what covariances we take into account
            diagonal: (n_mix_comp, dim)
            isotropic: (n_mix_comp)
            fixed: 0
    weight: torch.nn.parameter.Parameter
        learnable parameter for the component weights with shape (n_mix_comp)

    Methods
    ----------
    forward(x)
        returning negative log probability density of a set of representations
    log_prob(x)
        computes summed log probability density
    sample(n_sample)
        creates n_sample random samples from the GMM (taking into account mixture weights)
    component_sample(n_sample)
        creates n_sample new samples PER mixture component
    sample_probs(x)
        computes log-probs per sample (not summed and not including priors)
    sample_new_points(n_points, option='random', n_new=1)
        creating new samples either like in component_sample or from component means
    reshape_targets(y, y_type='true')
        reshaping targets (true counts) to be comparable to model outputs 
        from multiple representations per sample
    choose_best_representations(x, losses)
        reducing new representations to 1 per sample (based on lowest loss)
    choose_old_or_new(z_new, loss_new, z_old, loss_old)
        selecting best representation per sample between tow representation tensors (pairwise)
    '''

    def __init__(self,
            n_mix_comp: int,
            dim: int,
            covariance_type='diagonal',
            mean_init=(0.,1.),
            sd_init=(0.5,1.),
            weight_alpha=1
            ):
        super().__init__()
        
        # dimensionality of space and number of components
        self.dim = dim
        self.n_mix_comp = n_mix_comp

        # initialize public parameters
        self._init_means(mean_init)
        self._init_neglogvar(sd_init, covariance_type)
        self._init_weights(weight_alpha)

        # a dimensionality-dependent term needed in PDF
        self._pi_term = - 0.5*self.dim*math.log(2*math.pi)
    
    def _init_means(self, mean_init):
        self._mean_prior = softball(
            self.dim,
            mean_init[0],
            mean_init[1]
        )
        self.mean = nn.Parameter(self._mean_prior.sample(self.n_mix_comp), requires_grad=True)
    
    def _init_neglogvar(self, sd_init, covariance_type):
        # init parameter to learn covariance matrix (as negative log variance to ensure it to be positive definite)
        self._sd_init = sd_init
        self._neglogvar_factor = self.dim*0.5 # dimensionality factor in PDF
        self._neglogvar_dim = 1 # If 'diagonal' the dimension of is dim
        if covariance_type == 'fixed':
            # here there are no gradients needed for training
            # this would mainly be used to assume a standard Gaussian
            self.neglogvar = nn.Parameter(torch.empty(self.n_mix_comp,self._neglogvar_dim),requires_grad=False)
        else:
            if covariance_type == 'diagonal':
                self._neglogvar_factor = 0.5
                self._neglogvar_dim = self.dim
            elif covariance_type != 'isotropic':
                raise ValueError("type must be 'isotropic' (default), 'diagonal', or 'fixed'")
            
            self.neglogvar = nn.Parameter(torch.empty(self.n_mix_comp,self._neglogvar_dim),requires_grad=True)
            with torch.no_grad():
                self.neglogvar.fill_(-2*math.log(sd_init[0]))
            self._neglogvar_prior = gaussian(self._neglogvar_dim,-2*math.log(sd_init[0]),sd_init[1])
    
    def _init_weights(self, alpha):
        '''i.e. Dirichlet prior on mixture'''
        # dirichlet alpha determining the uniformity of the weights
        self._weight_alpha = alpha
        self._dirichlet_constant = math.lgamma(self.n_mix_comp*self._weight_alpha)-self.n_mix_comp*math.lgamma(self._weight_alpha)
        # weights are initialized uniformly so that components start out equi-probable
        self.weight = nn.Parameter(torch.ones(self.n_mix_comp),requires_grad=True)

    def forward(self,x):
        '''
        Forward pass computes the negative log density 
        of the probability of z being drawn from the mixture model
        '''

        # y = logp =  - 0.5*log (2pi) -0.5*beta(x-mean[i])^2 + 0.5*log(beta)
        # sum terms for each component (sum is over last dimension)
        # x is unsqueezed to (n_sample,1,dim), so broadcasting of mean (n_mix_comp,dim) works
        #y = - (x.unsqueeze(-2)-self.mean).square().mul(0.5*torch.exp(self.neglogvar)).sum(-1)
        y = - (x.unsqueeze(-2)-self.mean).square().div(2*self.covariance).sum(-1)
        y = y + self._neglogvar_factor*self.neglogvar.sum(-1)
        y = y + self._pi_term
        # For each component multiply by mixture probs
        y = y + torch.log_softmax(self.weight,dim=0)
        y = torch.logsumexp(y, dim=-1)
        y = y + self._prior_log_prob() # += gives cuda error

        return (-y) # returning negative log probability density
    
    def _prior_log_prob(self):
        ''' Calculate log prob of prior on mean, neglogvar, and mixture coefficients '''
        # Mixture weights
        p = self._dirichlet_constant
        if self._weight_alpha!=1:
            p = p + (self._weight_alpha-1.)*(self.mixture_probs().log().sum())
        # Means
        p = p+self._mean_prior.log_prob(self.mean).sum()
        # neglogvar
        if self._neglogvar_prior is not None:
            p =  p+self._neglogvar_prior.log_prob(self.neglogvar).sum()
        return p
    
    def log_prob(self,x):
        '''return the log density of the probability of z being drawn from the mixture model'''
        return - self.forward(x)
    
    def mixture_probs(self):
        '''transform weights to mixture probabilites'''
        return torch.softmax(self.weight,dim=-1)
    
    @property
    def covariance(self):
        '''transform negative log variance into covariances'''
        return torch.exp(-self.neglogvar)
    
    @property
    def stddev(self):
        return torch.sqrt(self.covariance)

    def _Distribution(self):
        '''create a distribution from mixture model (for sampling)'''
        with torch.no_grad():
            mix = D.Categorical(probs=torch.softmax(self.weight,dim=-1))
            comp = D.Independent(D.Normal(self.mean,torch.exp(-0.5*self.neglogvar)), 1)
            return D.MixtureSameFamily(mix, comp)

    def sample(self,n_sample):
        '''create samples from the GMM distribution'''
        with torch.no_grad():
            gmm = self._Distribution()
            return gmm.sample(torch.tensor([n_sample]))

    def component_sample(self,n_sample):
        '''Returns a sample from each component. Tensor shape (n_sample,n_mix_comp,dim)'''
        with torch.no_grad():
            comp = D.Independent(D.Normal(self.mean,torch.exp(-0.5*self.neglogvar)), 1)
            return comp.sample(torch.tensor([n_sample]))
    
    def sample_probs(self, x):
        '''compute probability densities per sample without prior. returns tensor of shape (n_sample, n_mix_comp)'''
        y = - (x.unsqueeze(-2)-self.mean).square().mul(0.5*torch.exp(self.neglogvar)).sum(-1)
        y = y + self._neglogvar_factor*self.neglogvar.sum(-1)
        y = y + self._pi_term
        y = y + torch.log_softmax(self.weight,dim=0)
        return torch.exp(y)
    
    def __str__(self):
        return f"""
        Gaussian_mix_compture:
            Dimensionality: {self.dim}
            Number of components: {self.n_mix_comp}
        """
    
    ##################################################
    '''This section is for learning new data points'''
    ##################################################

    def sample_new_points(self, resample_type='mean', n_new_samples=1):
        '''
        creates a Tensor with potential new representations. 
        These can be drawn from component samples if resample_type is 'sample' or
        from the mean if 'mean'. For drawn samples, n_new_samples defines the number
        of random samples drawn from each component.
        '''

        if resample_type == 'mean':
            samples = self.mean.clone().cpu().detach()
        else:
            samples = self.component_sample(n_new_samples).view(-1,self.dim).cpu().detach()
        return samples
    
    """
    def sample_new_points(self, n_points, option='random', n_new=1):
        '''
        Generates samples for each new data point
            - n_points defines the number of new data points to learn
            - option defines which of 2 schemes to use
                - random: sample n_new vectors from each component
                    -> n_mix_comp * n_new values per new point
                - mean: take the mean of each component as initial representation values
                    -> n_mix_comp values per new point
        The order of repetition in both options is [a,a,a, b,b,b, c,c,c] on data point ID.
        '''
        self.__new_samples = n_new
        multiplier = self.n_mix_comp
        if option == 'random':
            out = self.component_sample(n_points*n_new)
            multiplier *= n_new

        elif option == 'mean':
            with torch.no_grad():
                out = torch.repeat_interleave(self.mean.clone().cpu().detach().unsqueeze(0), n_points, dim=0)
        else:
            print("Please specify how to initialize new representations correctly \nThe options are 'random' and 'mean'.")
        return out.view(n_points*self.__new_samples*self.n_mix_comp,self.dim)
    
    def reshape_targets(self, y, y_type='true'):
        '''
        Since we have multiple representations for the same new data point,
        we need to reshape the output a little to calculate the losses
        Depending on the y_type, y can be
            - the true targets (y_type: 'true')
            - the model predictions (y_type: 'predicted') (can also be used for rep.z in dataloader loop)
            - the 4-dimensional representation or the loss (y_type: 'reverse')
        '''
        if y_type == 'true':
            if len(y.shape) > 2:
                raise ValueError('Unexpected shape in input to function reshape_targets. Expected 2 dimensions, got '+str(len(y.shape)))
            return y.unsqueeze(1).unsqueeze(1).expand(-1,self.__new_samples,self.n_mix_comp,-1)
        elif y_type == 'predicted':
            if len(y.shape) > 2:
                raise ValueError('Unexpected shape in input to function reshape_targets. Expected 2 dimensions, got '+str(len(y.shape)))
            n_points = int(torch.numel(y) / (self.__new_samples*self.n_mix_comp*y.shape[-1]))
            return y.view(n_points,self.__new_samples,self.n_mix_comp,y.shape[-1])
        elif 'reverse':
            if len(y.shape) < 4:
                # this case is for when the losses are of shape (n_points,self.new_samples,self.n_mix_comp)
                return y.view(y.shape[0]*self.__new_samples*self.n_mix_comp)
            else:
                return y.view(y.shape[0]*self.__new_samples*self.n_mix_comp,y.shape[-1])
        else:
            raise ValueError("The y_type in function reshape_targets was incorrect. Please choose between 'true' and 'predicted'.")
        
    def choose_best_representations(self, x, losses):
        '''
        Selects the representation for each new datapoint that maximizes the objective
            - x are the newly learned representations
            - x and losses have to have the same shape in the first dimension
              make sure that the losses are only summed over the output dimension
        Outputs new representation values
        '''
        n_points = int(torch.numel(losses) / (self.__new_samples*self.n_mix_comp))
        best_sample = torch.argmin(losses.view(-1,self.__new_samples*self.n_mix_comp), dim=1).squeeze(-1)
        best_rep = x.view(n_points,self.__new_samples*self.n_mix_comp,self.dim)[range(n_points),best_sample]
        return best_rep
    
    def choose_old_or_new(self, z_new, loss_new, z_old, loss_old):
        '''
        Allows to retrain representations
        After new representations have been learned, the best new ones are tested against the original representations
        and the best of either are returned.
        This can be applied when hoping to get out of local minima and when samples have been learned in the "wrong"
        components.
        '''
        if (len(z_new.shape) == 2) and (len(z_old.shape) == 2):
            z_conc = torch.cat((z_new.unsqueeze(1),z_old.unsqueeze(1)),dim=1)
        else:
            raise ValueError('Unexpected shape in input to function choose_old_or_new. Expected 2 dimensions for z_new and z_old, got '+str(len(z_new.shape))+' and '+str(len(z_old.shape)))
        
        len_loss_new = len(loss_new.shape)
        for l in range(3-len_loss_new):
            loss_new = loss_new.unsqueeze(1)
        len_loss_old = len(loss_old.shape)
        for l in range(3-len_loss_old):
            loss_old = loss_old.unsqueeze(1)
        losses = torch.cat((loss_new,loss_old),dim=1)

        best_sample = torch.argmin(losses, dim=1).squeeze(-1)
        #print(str(best_sample.sum().item())+' out of '+str(z_new.shape[0])+' samples were resampled.')
        
        best_rep = z_conc[range(z_conc.shape[0]),best_sample]

        return best_rep, round((best_sample.sum().item()/ z_new.shape[0])*100,2)
    """

class GaussianMixtureSupervised(GaussianMixture):
    '''
    Supervised GaussianMixutre class.

    Attributes
    ----------
    Nclass: int
        number of classes to be modeled
    Ncpc: int
        number of components that should model each class

    Methods
    ----------
    forward(x,label)
        computes negative log densities of representations
        if a supervision label is given for a sample, 
        all components not assigned to the class are ignored in computation
    log_prob(x,label=None)
    label_mixture_probs(label)
    supervised_sampling(label,sample_type='random')
    '''

    def __init__(self,
            Nclass: int,
            Ncompperclass: int,
            dim: int, 
            covariance_type="isotropic",
            mean_init=(0.,1.),
            sd_init=(0.5,1.),
            alpha=1
            ):
        super(GaussianMixtureSupervised, self).__init__(Nclass*Ncompperclass, dim, covariance_type, mean_init, sd_init, alpha)
        
        self.Nclass = Nclass # number of classes in the data
        self.Ncpc = Ncompperclass # number of components per class

    def forward(self,x,label=None):

        # return unsupervized loss if there are no labels provided
        if label is None:
            y = super().forward(x)
            return y
        
        # at the moment semi-supervized learning is not fully implemented
        # but this version is the (functional) placeholder
        # labels that are not provided are beforehand replaced with int 999
        if 999 in label:
            # first get normal loss
            idx_unsup = [i for i in range(len(label)) if label[i] == 999]
            y_unsup = super().forward(x[idx_unsup])
            # Otherwise use the component corresponding to the label
            idx_sup = [i for i in range(len(label)) if label[i] != 999]
            # Pick only the Ncpc components belonging class
            y_sup = - (x.unsqueeze(-2).unsqueeze(-2)-self.mean.view(self.Nclass,self.Ncpc,-1)).square().mul(0.5*torch.exp(self.neglogvar.unsqueeze(-2))).sum(-1)
            y_sup = y_sup + self._neglogvar_factor*self.neglogvar.view(self.Nclass,self.Ncpc,-1).sum(-1)
            y_sup = y_sup + self._pi_term
            y_sup = y_sup + torch.log_softmax(self.weight.view(self.Nclass,self.Ncpc),dim=-1)
            y_sup = y_sup.sum(-1)
            y_sup = torch.abs(y_sup[(idx_sup,label[idx_sup])] * self.Nclass) # this is replacement for logsumexp of supervised samples
            # put together y
            y = torch.empty((x.shape[0]), dtype=torch.float32).to(x.device)
            y[idx_unsup] = y_unsup
            y[idx_sup] = y_sup
        else:
            y = - (x.unsqueeze(-2).unsqueeze(-2)-self.mean.view(self.Nclass,self.Ncpc,-1)).square().mul(0.5*torch.exp(self.neglogvar.unsqueeze(-2))).sum(-1)
            y = y + self._neglogvar_factor*self.neglogvar.view(self.Nclass,self.Ncpc,-1).sum(-1)
            y = y + self._pi_term
            y += torch.log_softmax(self.weight.view(self.Nclass,self.Ncpc),dim=-1)
            y = y.sum(-1)
            y = y[(np.arange(y.shape[0]),label)] * self.Nclass # this is replacement for logsumexp of supervised samples
        
        y = y + self._prior_log_prob()
        return - y
    
    def log_prob(self,x,label=None):
        return - self.forward(x,label=label)

    def label_mixture_probs(self,label):
        return torch.softmax(self.weight[label],dim=-1)
    
    def supervised_sampling(self, label, sample_type='random'):
        # get samples for each component
        if sample_type == 'origin':
            # choose the component means
            samples = self.mean.clone().detach().unsqueeze(0).repeat(len(label),1,1)
        else:
            samples = self.component_sample(len(label))
        # then select the correct component
        return samples[range(len(label)),label]
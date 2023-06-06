import torch

class RepresentationLayer(torch.nn.Module):
    '''
    Implements a representation layer, that accumulates pytorch gradients.

    Representations are vectors in n_rep-dimensional real space. By default
    they will be initialized as a tensor of dimension n_sample x n_rep at origin (zero).

    One can also supply a tensor to initialize the representations (values=tensor).
    The representations will then have the same dimension and will assumes that
    the first dimension is n_sample (and the last is n_rep).

    The representations can be updated once per epoch by standard pytorch optimizers.

    Attributes
    ----------
    n_rep: int
        dimensionality of the representation space
    n_sample: int
        number of samples to be modelled (has to match corresponding dataset)
    z: torch.nn.parameter.Parameter
        tensor of learnable representations of shape (n_sample,n_rep)

    Methods
    ----------
    forward(idx=None)
        takes sample index and returns corresponding representation
    '''
    
    def __init__(self,
                n_rep: int,
                n_sample: int,
                value_init = 'zero'
                ):
        super(RepresentationLayer, self).__init__()
        
        self.n_rep=n_rep
        self.n_sample=n_sample

        if value_init == 'zero':
            self._value_init = 'zero'
            self.z = torch.nn.Parameter(torch.zeros(size=(self.n_sample,self.n_rep)), requires_grad=True)
        else:
            self._value_init = 'custom'
            # Initialize representations from a tensor with values
            #assert value_init.shape == (self.n_sample, self.n_rep)
            if isinstance(value_init, torch.Tensor):
                self.z = torch.nn.Parameter(value_init, requires_grad=True)
            else:
                try:
                    self.z = torch.nn.Parameter(torch.Tensor(value_init), requires_grad=True)
                except:
                    raise ValueError('not able to transform representation init values to torch.Tensor')

    def forward(self, idx=None):
        '''
        Forward pass returns indexed representations
        '''
        if idx is None:
            return self.z
        else:
            return self.z[idx]
    
    def __str__(self):
        return f"""
        RepresentationLayer:
            Dimensionality: {self.n_rep}
            Number of samples: {self.n_sample}
            Value initialization: {self._value_init}
        """
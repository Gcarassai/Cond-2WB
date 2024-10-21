from abc import ABC, abstractmethod
import torch.nn.functional as F
from numpy.typing import ArrayLike
import torch
import numpy as np

class encoder(ABC):
    """
    Abstract class for encoding the variable indication probability distribution
    """
    @abstractmethod
    def get_n_dim_out(self):
        """
        Returns the dimension of the output of the encoding
        """
        pass
    @abstractmethod
    def encode(self, s):
        """
        Encodes the input s
        """
        pass

class identity_encoder(encoder):
    """
    Identity encoder
    """
    def __init__(self,s_size):
        self.s_size = s_size
    def get_n_dim_out(self):
        return self.s_size
    def encode(self, s):
        return s

class one_hot_encoder(encoder):
    """
    One hot encoder
    """
    def __init__(self, n_distributions):
        self.n = n_distributions
    def get_n_dim_out(self):
        return self.n
    def encode(self, s):
        return F.one_hot(s.long(), num_classes=self.n).float()
    
class crime_encoder(encoder):
    """
    One hot encoder
    """
    def __init__(self, n_distributions: int, s_vals: ArrayLike):
        """
        n_distributions: number of distributions
        s_vals: all possible values of s, in increasing order
        """
        self.n = n_distributions
        self.s_vals = s_vals
    def get_n_dim_out(self):
        return self.n
    def encode(self, s):
        # map s to integers corresponding to the index s_vals]
        s = s.cpu().numpy()
        s = torch.tensor([np.argmin(np.abs(self.s_vals - s_i)) for s_i in s])
        return F.one_hot(s.long(), num_classes=self.n).float()
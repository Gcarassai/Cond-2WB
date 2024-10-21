from networks import PICNN
from embeddings import encoder
import torch
import torch.nn as nn

class cond_kpot(nn.Module):
    def __init__(self, network:PICNN, embedder:encoder)->None:
        super(cond_kpot, self).__init__()
        """
        Class for conditional Kantorovich potential

        Args:
        network: PICNN
        embedder: encoder
        """
        # check dimensions match
        if network.input_s_dim != embedder.get_n_dim_out(): 
            raise ValueError('Dimensions do not match between emdedder output and network sensitive input')

        self.network = network
        self.embedder = embedder

    def forward(self, x:torch.Tensor, s:torch.Tensor)->torch.Tensor:
        return self.network(x, self.embedder.encode(s))
    
    def push(self, x:torch.Tensor, s:torch.Tensor)->torch.Tensor:
        return self.network.push(x, self.embedder.encode(s))



class cond_2wb():
    def __init__(self, network:PICNN, embedder:encoder, pretrain_mod: str = None)->None:
        self.network = network
        self.embedder = embedder
        if pretrain_mod:
            self.pretrain(pretrain_mod)
        self.f = cond_kpot(network, embedder)
        self.g = cond_kpot(network, embedder)

    def pretrain(self, mod: str = 'parabola')->None:
        if mod == 'parabola':
            self.network.pretrain_parabola()
        if mod == 'network':
            self.network.pretrain()
        elif mod == 'embedder':
            self.embedder.pretrain()
        else:
            raise ValueError('Invalid module name')
        

    # def train(self, optimizer, data_sampler, proposed_sampler, loss_fn, verbose)->None:
    #     """
    #     Train the model
    #     """
    #     pass
    
    # def report():
    #     """
    #     Report and plot stuff (import functions from utils)
    #     """

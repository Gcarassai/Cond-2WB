import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torch.autograd as autograd
from typing import Literal

class PICNN(nn.Module):

    def __init__(self, input_x_dim: int, input_s_dim: int, feature_dim: int, feature_s_dim: int, out_dim: int,
                 num_layers: int, act=F.softplus, act_v=nn.ELU(), nonneg=F.relu, reparam=True) -> None:
        """
        Implementation of the Partially Input Convex Neural Networks (Amos et al., 2017),
        whith three options to impose convexity:
        a) non-negative weights restricted using reparameterization 
        b) non-negative clipping after optimizer step
        c) regularization term in loss

        :param input_x_dim: input data convex dimension
        :param input_s_dim: input data non-convex dimension
        :param feature_dim: intermediate feature dimension
        :param feature_s_dim: intermediate context dimension
        :param out_dim: output dimension
        :param num_layers: number of layers
        :param act: choice of activation for w path
        :param act_v: choice of activation in v path and for v in hadamard product with x
        :param nonneg: activation function for non-negative weights
        :param reparam: handling of non-negative constraints
        """

        super(PICNN, self).__init__()
        self.input_x_dim = input_x_dim
        self.input_s_dim = input_s_dim
        self.feature_dim = feature_dim
        self.feature_s_dim = feature_s_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # forward path for v(y)
        Lv = list()
        Lv.append(nn.Linear(input_s_dim, feature_s_dim, bias=True))
        for k in range(num_layers - 1):
            Lv.append(nn.Linear(feature_s_dim, feature_s_dim, bias=True))
        self.Lv = nn.ModuleList(Lv)

        # forward path for v into w
        Lvw = list()
        Lvw.append(nn.Linear(input_s_dim, feature_dim, bias=False))
        for k in range(num_layers - 1):
            Lvw.append(nn.Linear(feature_s_dim, feature_dim, bias=False))
        Lvw.append(nn.Linear(feature_s_dim, out_dim, bias=False))
        self.Lvw = nn.ModuleList(Lvw)

        # forward path for w product, positive weights
        Lw = list()
        Lw0 = nn.Linear(input_x_dim, feature_dim, bias=True)
        # positive weights
        with torch.no_grad():
            Lw0.weight.data = nonneg(Lw0.weight)
        Lw.append(Lw0)

        for k in range(num_layers - 1):
            Lwk = nn.Linear(feature_dim, feature_dim, bias=True)
            with torch.no_grad():
                Lwk.weight.data = nonneg(Lwk.weight)
            Lw.append(Lwk)

        LwK = nn.Linear(feature_dim, out_dim, bias=True)
        with torch.no_grad():
            LwK.weight.data = nonneg(LwK.weight)
        Lw.append(LwK)
        self.Lw = nn.ModuleList(Lw)

        # context path for v times w
        Lwv = list()
        Lwv.append(nn.Linear(input_s_dim, input_x_dim, bias=True))
        for k in range(num_layers):
            Lwv.append(nn.Linear(feature_s_dim, feature_dim, bias=True))
        self.Lwv = nn.ModuleList(Lwv)

        # context path for v times x
        Lxv = list()
        for k in range(num_layers):
            Lxv.append(nn.Linear(feature_s_dim, input_x_dim, bias=True))
        self.Lxv = nn.ModuleList(Lxv)

        # forward path for x product
        Lx = list()
        for k in range(num_layers - 1):
            Lx.append(nn.Linear(input_x_dim, feature_dim, bias=False))
        Lx.append(nn.Linear(input_x_dim, out_dim, bias=False))
        self.Lx = nn.ModuleList(Lx)

        self.act = act
        self.act_v = act_v
        self.nonneg = nonneg
        self.reparam = reparam

    def forward(self, in_x, in_s):
        # first layer activation
        v = in_s
        w0_prod = torch.mul(in_x, F.relu(self.Lwv[0](v)))    # relu for non-negativity
        if self.reparam is True:
            # reparameterization
            Lw0_pos = self.nonneg(self.Lw[0].weight)
            w = self.act(F.linear(w0_prod, Lw0_pos, self.Lw[0].bias) + self.Lvw[0](v))
        else:
            w = self.act(self.Lw[0](w0_prod) + self.Lvw[0](v))

        # zip the models
        NN_zip = zip(self.Lv[:-1], self.Lvw[1:-1], self.Lw[1:-1],
                     self.Lwv[1:-1], self.Lxv[:-1], self.Lx[:-1])

        # intermediate layers activations
        for lv, lvw, lw, lwv, lxv, lx in NN_zip:
            down = 1 / len(w.t())
            v = self.act_v(lv(v))
            wk_prod = torch.mul(w, F.relu(lwv(v)))
            xk_prod = torch.mul(in_x, lxv(v))
            if self.reparam is True:
                Lwk_pos = self.nonneg(lw.weight)
                w = self.act(F.linear(wk_prod, Lwk_pos, lw.bias) * down + lx(xk_prod) + lvw(v))
            else:
                w = self.act(lw(wk_prod) * down + lx(xk_prod) + lvw(v))

        # last layer activation
        down = 1 / len(w.t())
        vK = self.act_v(self.Lv[-1](v))
        wK_prod = torch.mul(w, F.relu(self.Lwv[-1](vK)))
        xK_prod = torch.mul(in_x, self.Lxv[-1](vK))
        if self.reparam is True:
            LwK_pos = self.nonneg(self.Lw[-1].weight)
            w = F.linear(wK_prod, LwK_pos, self.Lw[-1].bias) * down + self.Lx[-1](xK_prod) + self.Lvw[-1](vK)
        else:
            w = self.Lw[-1](wK_prod) * down + self.Lx[-1](xK_prod) + self.Lvw[-1](vK)
        return w
    
    def convexify(self) -> None:
        """
        Perform non-negative clipping of Lw weights after optimizer step
        """
        if self.reparam:
            warnings.warn("Reparametrization is set to True, therfore PICNN is always convex", 
                          category=DeprecationWarning)
            return 
        
        for lw in self.Lw:
            with torch.no_grad():
                lw.weight.data = self.nonneg(lw.weight)
        return

    def regularize_conv_loss(self) -> torch.Tensor:
        """
        Regularization term to relax convexity in x path: 
        R_conv = \sum_{l=1}^{L} ||ReLU(-W_l)||_F
        """
        reg = 0.
        if not self.reparam:
            for lw in self.Lw:
                reg += torch.linalg.matrix_norm(torch.relu(-lw.weight),ord='fro')
        else:
            warnings.warn("Reparametrization is set to True, therfore PICNN is convex", 
                          category=DeprecationWarning)
        
        return reg
    
    def push(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Calculate transport map as gradient of Kantorovich potential
        """
        assert x.requires_grad
        out = self.forward(x,s)
        grad = autograd.grad(
            outputs=out, 
            inputs=x,
            create_graph=True, 
            retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones_like(out).float()
        )[0]
        return grad
    
    def initialize_weights(self, mod: Literal['gaussian'], device: str) -> None:
        """
        default is uniform initialization
        Initialize weights !! SEE BUNNE and implement !!!
        """
        if mod == 'gaussian':
            for lw in self.Lw:
                lw.weight.data = nn.init.xavier_normal_(lw.weight.data)
            for lv in self.Lv:
                lv.weight.data = nn.init.xavier_normal_(lv.weight.data)
            for lvw in self.Lvw:
                lvw.weight.data = nn.init.xavier_normal_(lvw.weight.data)
            for lwv in self.Lwv:
                lwv.weight.data = nn.init.xavier_normal_(lwv.weight.data)
            for lxv in self.Lxv:
                lxv.weight.data = nn.init.xavier_normal_(lxv.weight.data)
            for lx in self.Lx:
                lx.weight.data = nn.init.xavier_normal_(lx.weight.data)
        return
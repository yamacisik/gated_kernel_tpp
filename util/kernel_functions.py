import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class rational_quadratic_kernel(nn.Module):

    def __init__(self,
                 num_types, d_type, sigma=1, norm=1, t_max=200, alpha=1.0):
        super().__init__()

        self.d_type = d_type
        self.num_types = num_types
        self.norm = norm
        self.sigma = sigma
        self.param_loss = 0
        self.scores = None
        self.t_max = 1
        """
        If the model is 1-D we still need the type embedding as an input and train a linear layer followed by softplus
        to make sure the length_scale parameter is positive. Adding a parameter followed by a sigmoid creates a non leaf
        tensor and there for doesn't work.
        """

        if num_types == 1:
            self.lengthscale = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus())
            # self.alpha = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Sigmoid())
            self.register_buffer('alpha',torch.tensor([alpha]),persistent = False)

        else:
            self.lengthscale = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus())
            # self.alpha = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Sigmoid())
            self.alpha = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False),
                                       nn.Sigmoid()) if alpha == None else torch.tensor(alpha)

    def forward(self, x, type_emb):

        d = (torch.abs(x[0] - x[1]) / self.t_max) ** self.norm

        if self.num_types == 1:
            space = type_emb[0]
        else:
            space = torch.cat(type_emb[0], type_emb[1], -1)

        lengthscale = self.lengthscale(space).squeeze()
        # alpha = self.alpha(space).squeeze()
        alpha = self.alpha

        self.scores = (self.sigma ** 2) * (1 + (d ** 2) / (alpha * lengthscale ** 2)) ** (-alpha)

        return self.scores

    def params(self, type_emb):

        params = []
        for space in type_emb[1:]:
            lengthscale = self.lengthscale(space).item()
            # alpha = self.alpha(space).item()
            alpha = self.alpha.item()

            params.append({'length_scale': lengthscale, 'alpha': alpha, 'sigma': self.sigma, 'Norm-P': self.norm})

        return params

class fixed_kernel(nn.Module):

    def __init__(self, sigma=1, norm=1, t_max=200,length_scale = 1.0,alpha = 0.1):
        super().__init__()


        self.norm = norm
        self.sigma = sigma
        self.length_scale = length_scale
        self.alpha = alpha
        self.scores = None
        self.t_max = t_max

    def forward(self, x):

        d = (torch.abs(x[0] - x[1]) / self.t_max) ** self.norm

        # self.scores = (self.sigma ** 2) * torch.exp(-(d ** 2) / self.length_scale ** 2)
        self.scores = (self.sigma ** 2) * (1 + (d ** 2) / (self.alpha * self.length_scale ** 2)) ** (-self.alpha)

        return self.scores


class squared_exponential_kernel(nn.Module):

    def __init__(self,
                 num_types, d_type, sigma=1, norm=1, t_max=200, alpha=2.0,beta = 1.0):
        super().__init__()

        self.d_type = d_type
        self.num_types = num_types
        self.norm = norm
        self.sigma = sigma
        self.param_loss = 0
        self.scores = None
        self.t_max = 1
        """
        If the model is 1-D we still need the type embedding as an input and train a linear layer followed by softplus
        to make sure the length_scale parameter is positive. Adding a parameter followed by a sigmoid creates a non leaf
        tensor and there for doesn't work.
        """

        if num_types == 1:
            if self.t_max == 1:
                self.lengthscale = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus(beta=beta))
            else:
                self.lengthscale = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Sigmoid())
        else:
            if self.t_max == 1:
                self.lengthscale = nn.Sequential(nn.Linear(d_type*2, 1, bias=False), nn.Softplus(beta=beta))
            else:
                self.lengthscale = nn.Sequential(nn.Linear(d_type*2, 1, bias=False), nn.Sigmoid())

    def forward(self, x, type_emb):

        d = (torch.abs(x[0] - x[1]) / self.t_max) ** self.norm

        if self.num_types == 1:
            space = type_emb[0]
        else:
            space = torch.cat(type_emb[0], type_emb[1], -1)

        self.scores = (self.sigma ** 2) * torch.exp(-(d ** 2) / self.lengthscale(space).squeeze(-1) ** 2)

        return self.scores

    def params(self, type_emb):

        params = []
        for space in type_emb[1:]:
            lengthscale = self.lengthscale(space).item()
            params.append({'length_scale': lengthscale, 'sigma': self.sigma, 'Norm-P': self.norm})
        return params


class SigmoidGate(nn.Module):

    def __init__(self, num_types, d_type, norm=1, t_max=200,s = 0.1):
        super().__init__()

        self.d_model = d_type
        self.num_types = num_types
        self.t_max = t_max
        self.norm = 1
        self.scores = None
        if num_types == 1:
            self.l = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus())

            self.register_buffer('s',torch.tensor([s]),persistent = False)
            # self.s = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus())

        else:
            self.l = nn.Sequential(nn.Linear(d_type*2, 1, bias=False), nn.Softplus())
            self.s = nn.Sequential(nn.Linear(d_type*2, 1, bias=False), nn.Softplus())

    def forward(self, x, type_emb):

        d = (torch.abs(x[0] - x[1]) / self.t_max) ** self.norm

        if self.num_types == 1:
            space = type_emb[0]
        else:
            space = torch.cat(type_emb[0], type_emb[1], -1)


        l = self.l(space).squeeze()
        # s = self.s(space).squeeze()

        s = self.s
        self.scores = 0.5 + 0.5 * torch.tanh((d - l) / s)

        return self.scores

    def params(self, type_emb):

        params = []
        for space in type_emb[1:]:
            l = self.l(space).item()
            # s = self.s(space).item()
            s = self.s.item()
            params.append({'Gate_Param L': l, 'Gate_Param S': s, 'Norm-P': self.norm})
        return params





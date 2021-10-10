import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class rational_quadratic_kernel(nn.Module):
#
#     def __init__(self,
#                  num_types, d_type, sigma=1, p=1, alpha=1,lengthscale = 1.0):
#         super().__init__()
#
#         self.d_type = d_type
#         self.num_types = num_types
#         self.norm = p
#         # self.sigma = sigma
#         self.param_loss = 0
#         self.scores = None
#         """
#         If the model is 1-D we still need the type embedding as an input and train a linear layer followed by softplus
#         to make sure the length_scale parameter is positive. Adding a parameter followed by a sigmoid creates a non leaf
#         tensor and there for doesn't work.
#         """
#
#         if num_types == 1:
#             # self.lengthscale = nn.Sequential(nn.Linear(d_type, d_type, bias=False),nn.ReLU(),nn.Linear(d_type, 1, bias=False), nn.Sigmoid())
#             # self.lengthscale = nn.Sequential(nn.Linear(d_type, 1, bias=True), nn.Softplus())
#
#             self.lengthscale = torch.nn.Parameter(torch.randn(1))
#             # self.alpha = torch.nn.Parameter(torch.randn(1))
#             # self.sigma = torch.nn.Parameter(torch.randn(1))
#
#             # self.lengthscale.requires_grad = True
#
#
#             # self.alpha = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Sigmoid())
#             # self.register_buffer('alpha',torch.tensor([alpha]),persistent = False)
#             # self.register_buffer('lengthscale', torch.tensor([lengthscale]), persistent=False)
#             # self.register_buffer('sigma', torch.tensor([sigma]), persistent=False)
#
#         else:
#             self.lengthscale = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus())
#             # self.alpha = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Sigmoid())
#             self.alpha = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False),
#                                        nn.Sigmoid()) if alpha == None else torch.tensor(alpha)
#
#     def forward(self, x, type_emb):
#
#         if len(x[0].size()) > 3:
#             d = (torch.abs(x[0] - x[1]) ** self.norm).sum(-1) ** (1 / self.norm)
#         else:
#             d = torch.abs(x[0] - x[1])**self.norm
#         if self.num_types == 1:
#             space = type_emb[0]
#         else:
#             space = torch.cat(type_emb[0], type_emb[1], -1)
#
#         # lengthscale = self.lengthscale(space).squeeze()
#         # alpha = self.alpha(space).squeeze()
#         # alpha = self.alpha
#         lengthscale = F.softplus(self.lengthscale,beta = 1)
#         # sigma = self.sigma
#
#
#         lengthscale = 0.8
#         alpha = 1
#         sigma = 1
#
#         self.scores = (sigma ** 2) * (1 + (d ** 2) / (alpha * lengthscale ** 2)) ** (-alpha)
#
#         return self.scores
#
#     def params(self, type_emb):
#
#         params = []
#         for space in type_emb[1:]:
#             lengthscale = self.lengthscale(space).item()
#             # alpha = self.alpha(space).item()
#             alpha = self.alpha.item()
#
#             params.append({'length_scale': lengthscale, 'alpha': alpha, 'sigma': self.sigma, 'Norm-P': self.norm})
#
#         return params

class multiplication_kernel(nn.Module):

    def __init__(self,
                 num_types, d_type, sigma=1, alpha=1,lengthscale = 1.0):
        super().__init__()

        self.d_type = d_type
        self.num_types = num_types
        self.scores = None
        """
        If the model is 1-D we still need the type embedding as an input and train a linear layer followed by softplus
        to make sure the length_scale parameter is positive. Adding a parameter followed by a sigmoid creates a non leaf
        tensor and there for doesn't work.
        """

        if num_types == 1:
            self.kernel_function = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Softplus())

        else:
            self.lengthscale = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus(beta = 5))
            self.alpha = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False),
                                       nn.Sigmoid()) if alpha == None else torch.tensor(alpha)

    def forward(self, time_diff, type_emb):
        pass

        d = time_diff

        # if self.num_types == 1:




class squared_exponential_kernel(nn.Module):

    def __init__(self,
                 num_types, d_type, sigma=1, norm=1,alpha=2.0,beta = 1.0):
        super().__init__()

        self.d_type = d_type
        self.num_types = num_types
        self.norm = norm
        self.sigma = sigma
        self.param_loss = 0
        self.scores = None

        """
        If the model is 1-D we still need the type embedding as an input and train a linear layer followed by softplus
        to make sure the length_scale parameter is positive. Adding a parameter followed by a sigmoid creates a non leaf
        tensor and there for doesn't work.
        """

        if num_types == 1:
            self.lengthscale = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus(beta=beta))

        else:

            self.lengthscale = nn.Sequential(nn.Linear(d_type*2, 1, bias=False), nn.Softplus(beta=beta))


    def forward(self, x, type_emb):
        if len(x[0].size())>3:
            d = (torch.abs(x[0] - x[1])**self.norm).sum(-1)**(1/self.norm)
        else:
            d = torch.abs(x[0] - x[1])

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

    def __init__(self, num_types, d_type, norm=1,s = 1.0,l = 1.0):
        super().__init__()

        self.d_model = d_type
        self.num_types = num_types

        self.norm = 1
        self.scores = None
        if num_types == 1:
            # self.l = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus())
            # self.register_buffer('s',torch.tensor([s]),persistent = False)
            # self.register_buffer('l', torch.tensor([l]), persistent=False)
            # self.s = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus())

            self.s = torch.nn.Parameter(torch.randn(1))
            self.l = torch.nn.Parameter(torch.randn(1))
            self.b = torch.nn.Parameter(torch.randn(1))


        else:
            self.l = nn.Sequential(nn.Linear(d_type*2, 1, bias=False), nn.Softplus())
            self.s = nn.Sequential(nn.Linear(d_type*2, 1, bias=False), nn.Softplus())

    def forward(self, x, type_emb):

        d = torch.abs(x[0] - x[1]) + 1e-6

        if self.num_types == 1:
            space = type_emb[0]
        else:
            space = torch.cat(type_emb[0], type_emb[1], -1)


        # l = self.l(space).squeeze()
        # s = self.s(space).squeeze()
        #
        # s = F.softplus(self.s,beta = 1)
        # l = F.softplus(self.l,beta = 1)
        b = F.softplus(self.b,beta =10)+1

        s = 0.5 +F.sigmoid(self.s)/2
        l = 1.4
        # b = 1.0
        scale = 2.5

        self.scores = (b + torch.tanh((d - l) / s))/scale

        return self.scores

    def params(self, type_emb):

        params = []
        for space in type_emb[1:]:
            # l = self.l(space).item()
            # s = self.s(space).item()
            s = self.s.item()
            l = self.l.item()
            params.append({'Gate_Param L': l, 'Gate_Param S': s, 'Norm-P': self.norm})
        return params










class space_time_kernel(nn.Module):

    def __init__(self, sigma=1, norm=1,length_scale = 1.0,alpha = 0.1):
        super().__init__()


        self.norm = norm
        self.sigma = sigma
        self.length_scale = length_scale
        self.alpha = alpha
        self.scores = None

    def forward(self, x):

        if len(x[0].size()) > 3:
            d = (torch.abs(x[0] - x[1]) ** self.norm).sum(-1) ** (1 / self.norm)
        else:
            d = torch.abs(x[0] - x[1])

        # self.scores = (self.sigma ** 2) * torch.exp(-(d ** 2) / self.length_scale ** 2)
        self.scores = (self.sigma ** 2) * (1 + (d ** 2) / (self.alpha * self.length_scale ** 2)) ** (-self.alpha)
        return self.scores


class rational_quadratic_kernel(nn.Module):

    def __init__(self,
                 num_types=1, d_type=1, sigma=1, p=1, alpha=1, lengthscale=1.0, betas=[5, 1, 1]):
        super().__init__()

        self.d_type = d_type
        self.num_types = num_types
        self.norm = p
        # self.sigma = sigma
        self.param_loss = 0
        self.scores = None
        self.alpha = alpha
        self.betas = betas
        """
        If the model is 1-D we still need the type embedding as an input and train a linear layer followed by softplus
        to make sure the length_scale parameter is positive. Adding a parameter followed by a sigmoid creates a non leaf
        tensor and there for doesn't work.
        """

        if num_types == 1:

            self.lengthscale = torch.nn.Parameter(torch.randn(1))
            self.base_intensity = torch.nn.Parameter(torch.randn(1))
            self.sigma = torch.nn.Parameter(torch.randn(1))

        else:
            self.lengthscale = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus())
            self.sigma = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus())
            self.base_intensity = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus())

    def forward(self, time_diff, combined_embeddings=None):

        d = time_diff

        if self.num_types == 1:
            lengthscale = F.softplus(self.lengthscale)
            sigma = F.softplus(self.sigma)
            base_intensity = F.softplus(self.base_intensity)

        else:
            lengthscale = self.lengthscale(combined_embeddings).squeeze(-1)
            sigma = self.sigma(combined_embeddings).squeeze(-1)
            base_intensity = self.base_intensity(combined_embeddings[:, :, :, self.d_type:]).squeeze(-1)

        self.scores = (sigma ** 2) * (1 + (d ** 2) / (self.alpha * lengthscale ** 2)) ** (-self.alpha)

        return self.scores

    def regularizer_loss(self):

        return self.lengthscale + self.sigma


class magic_kernel(nn.Module):

    def __init__(self,
                 num_types=1, d_type=1, sigma=1, p=1, alpha=1, lengthscale=1.0, betas=[1, 1, 1]):
        super().__init__()

        self.d_type = d_type
        self.num_types = num_types
        self.norm = p
        self.param_loss = 0
        self.scores = None
        self.sigma = sigma
        self.alpha = alpha

        self.betas = betas

        """
        If the model is 1-D we still need the type embedding as an input and train a linear layer followed by softplus
        to make sure the length_scale parameter is positive. Adding a parameter followed by a sigmoid creates a non leaf
        tensor and there for doesn't work.
        """

        if num_types == 1:

            self.lengthscale = torch.nn.Parameter(torch.randn(1))
            self.base_intensity = torch.nn.Parameter(torch.randn(1))
            self.sigma = torch.nn.Parameter(torch.randn(1))
            self.alpha = torch.nn.Parameter(torch.randn(1))

        else:
            self.lengthscale = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus(self.betas[0]))
            self.alpha = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus(self.betas[1]))
            self.sigma = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Sigmoid())
            self.base_intensity = nn.Sequential(nn.Linear(d_type, 1, bias=False), nn.Softplus(self.betas[2]))

    def forward(self, time_diff, combined_embeddings=None,non_event_intensity = False):

        d = time_diff

        if self.num_types == 1:
            lengthscale = F.softplus(self.lengthscale,self.betas[0])
            sigma = torch.sigmoid(self.sigma)
            # sigma = 1
            alpha = F.softplus(self.alpha,self.betas[1])
            base_intensity = F.softplus(self.base_intensity,self.betas[2])

        else:
            if not non_event_intensity:
                lengthscale = self.lengthscale(combined_embeddings).squeeze(-1)
                sigma = self.sigma(combined_embeddings).squeeze(-1)
                alpha = self.alpha(combined_embeddings).squeeze(-1)

            else:
                lengthscale = self.lengthscale(combined_embeddings)
                sigma = self.sigma(combined_embeddings)
                alpha = self.alpha(combined_embeddings)

            base_intensity = self.base_intensity(combined_embeddings[:, :, :, self.d_type:]).squeeze(-1)

        # (sigma ** 2) * (1 + (d ** 2) / (self.alpha * lengthscale ** 2)) ** (-self.alpha)
        #
        # (sigma ** 2) * (1 + (d ** 2) / (self.alpha * lengthscale ** 2)) ** (-self.alpha) *((1 + torch.exp(-d)) ** -alpha)
        self.scores = sigma * torch.exp(-d / lengthscale) * ((1 + torch.exp(-d)) ** -alpha)
        # self.scores = (sigma ** 2) * (1 + (d ** 2) / (1 * lengthscale ** 2)) ** (-1) *((1 + torch.exp(-d)) ** -alpha)

        return self.scores

    def regularizer_loss(self):

        return self.lengthscale + self.sigma


def get_pairwise_times(event_time):
    xt_bar = event_time.unsqueeze(1). \
        expand(event_time.size(0), event_time.size(1), event_time.size(1))
    xt = xt_bar.transpose(1, 2)
    return (xt_bar, xt)


def get_pairwise_type_embeddings(embeddings):
    xd_bar = embeddings.unsqueeze(1).expand(embeddings.size(
        0), embeddings.size(1), embeddings.size(1), embeddings.size(-1))
    xd = xd_bar.transpose(1, 2)

    return (xd_bar, xd)

def get_sample_intensities(kernel,event_time, arrival_time, event_type, device='cpu', embeddings=None):

    # event_time, arrival_time, event_type, _ = map(lambda x: x.to(device), batch)

    xt_bar, xt = get_pairwise_times(event_time)
    t_diff = torch.abs(xt_bar - xt)
    n_batch = t_diff.size()[0]
    length_batch = t_diff.size()[1]

    if kernel.num_types == 1:
        scores = kernel(t_diff)
        scores_0 = kernel(torch.zeros(n_batch, length_batch).to(device))  # We need this to get actual intensities
        base_intensity = F.softplus(kernel.base_intensity, beta=kernel.betas[-1])

    else:

        pair_wise_embeddings = get_pairwise_type_embeddings(embeddings)
        combined_embeddings = torch.cat([pair_wise_embeddings[0], pair_wise_embeddings[1]],
                                        dim=-1)  # Last Part is the current event
        scores = kernel(t_diff, combined_embeddings)
        scores_0 = kernel(torch.zeros(n_batch, length_batch, length_batch).to(device),
                          combined_embeddings)  # We need this to get actual intensities
        scores_0 = torch.diagonal(scores_0, dim1=1, dim2=-1)
        base_intensity = kernel.base_intensity(embeddings).squeeze(-1)

    subsequent_mask = get_subsequent_mask(event_type)
    sample_intensities = scores.masked_fill_(subsequent_mask == 0, value=0).sum(-1)

    sample_intensities = scores.sum(-1) - scores_0
    seq_length_mask = (event_type != 0) * 1

    return (sample_intensities + base_intensity) * seq_length_mask


def get_non_event_intensities(kernel,  event_time, arrival_time, event_type,
                              type_embeddings=None, device='cpu',mc_sample_size = 5):

    # event_time, arrival_time, event_type, _ = map(lambda x: x.to(device), batch)

    sample_arrival_time = arrival_time[:, 1:]
    sample_event_time = event_time[:, :-1]
    t_last = sample_event_time.max(-1)[0]
    n_batch = sample_arrival_time.size(0)
    n_t = sample_arrival_time.size(1)

    mc_values = torch.rand((n_batch, n_t, mc_sample_size)).to(device) * \
                sample_arrival_time.unsqueeze(-1) + sample_event_time.unsqueeze(-1)

    samples = sample_event_time.unsqueeze(-1).expand((n_batch, n_t, mc_sample_size))
    samples = samples.unsqueeze(1).expand(samples.size(
        0), samples.size(1), samples.size(1), samples.size(-1))

    mc_values_bar = mc_values.unsqueeze(1).expand(mc_values.size(
        0), mc_values.size(1), mc_values.size(1), mc_values.size(-1))

    mc_values_bar = mc_values_bar.transpose(1, 2)
    d = torch.abs((mc_values_bar - samples))

    if kernel.num_types == 1:

        non_event_intensities = kernel(d)
        trigger_integral = non_event_intensities.mean(-1)
        subsequent_mask = get_subsequent_mask(event_type[:, 1:])
        integral = trigger_integral * subsequent_mask
        integral = integral.sum(-1)
        seq_length_mask = (event_type[:, 1:] != 0) * 1
        integral = integral * seq_length_mask
        base_intensity = F.softplus(kernel.base_intensity, beta=1)
        sequence_integral =integral.sum(-1)+base_intensity*t_last
    else:
        sequence_integral = 0
        for i in range(kernel.num_types):
            sample_event_types = event_type[:, 1:]
            sample_embeddings = type_embeddings(sample_event_types)
            xd_bar = sample_embeddings.unsqueeze(1).expand(sample_embeddings.size(
                0), sample_embeddings.size(1), sample_embeddings.size(1), sample_embeddings.size(-1))

            current_embedding = type_embeddings(torch.tensor([i]).to(device))
            current_embedding = current_embedding[:, None, None:].expand(xd_bar.size()).transpose(1, 2)
            combined_embeddings = torch.cat([xd_bar, current_embedding], dim=-1)

            non_event_intensities = kernel(d, combined_embeddings, non_event_intensity=True)
            trigger_integral = non_event_intensities.mean(-1)
            subsequent_mask = get_subsequent_mask(sample_event_types)
            integral = trigger_integral * subsequent_mask
            integral = integral.sum(-1)
            seq_length_mask = (event_type[:, 1:] != 0) * 1
            integral = integral * seq_length_mask
            base_intensity = kernel.base_intensity(type_embeddings(torch.tensor([i]).to(device)))
            sequence_integral += integral.sum(-1) + base_intensity * t_last

    return sequence_integral

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    subsequent_mask = (subsequent_mask - 1) ** 2
    return subsequent_mask

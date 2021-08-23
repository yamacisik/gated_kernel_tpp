import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class gated_TPP(nn.Module):

    def __init__(self,
                 num_types, d_model, t_max=200, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.encoder = Encoder(num_types, d_model, t_max=t_max)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.decoder = Decoder(num_types, d_model, dropout)

    def forward(self, event_type, event_time):
        scores, embeddings, _ = self.encoder(event_type, event_time)
        hidden = torch.matmul(scores, embeddings)
        hidden = self.norm(hidden)

        return self.decoder(hidden)

    def calculate_loss(self, batch_arrival_times, sampled_arrival_times, batch_types):
        arrival_times = batch_arrival_times[:, 1:]
        sampled_times = sampled_arrival_times[:, :-1]

        l1_loss = torch.abs(arrival_times - sampled_times)

        batch_lengths = (batch_types != 0).sum(-1).to('cpu')  ## Non events are 0

        batch_loss = nn.utils.rnn.pack_padded_sequence(l1_loss.T, batch_lengths - 1, enforce_sorted=False)[0]

        time_loss = batch_loss.sum()

        ## Add Param Regularizer 1-D
        types = self.encoder.type_emb.weight * math.sqrt(self.d_model)
        param_size = torch.abs(self.encoder.kernel.length_scale(types)[-1][-1]).sum()
        return time_loss


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self,
                 num_types, d_model, t_max=200):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.type_emb = nn.Embedding(num_types + 1, d_model, padding_idx=0)
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i) / d_model) for i in range(d_model)])
        self.sigmoid = SigmoidGate(num_types, d_model, norm=1)
        self.kernel = squared_exponential_kernel(num_types, d_model, norm=1)
        self.t_max = t_max

    def forward(self, event_type, event_time):
        """ Encode event sequences via kernel functions """

        # Temporal Encoding
        temp_enc = event_time.unsqueeze(-1) / self.position_vec.to(event_time.device)
        temp_enc[:, :, 0::2] = torch.sin(temp_enc[:, :, 0::2])
        temp_enc[:, :, 1::2] = torch.cos(temp_enc[:, :, 1::2])

        ## Type Encoding
        type_embedding = self.type_emb(event_type) * math.sqrt(
            self.d_model)  ## Scale the embedding with the hidden vector size
        embedding = temp_enc
        # embedding = torch.cat([temp_enc, type_embedding], dim=-1)

        ## Future Masking
        subsequent_mask = get_subsequent_mask(event_type)

        ## Time Scores
        normalized_event_time = event_time
        xt_bar = normalized_event_time.unsqueeze(1). \
            expand(normalized_event_time.size(0), normalized_event_time.size(1), normalized_event_time.size(1))
        xt = xt_bar.transpose(1, 2)

        ## Space  Input
        xd_bar = type_embedding.unsqueeze(1).expand(type_embedding.size(
            0), type_embedding.size(1), type_embedding.size(1), type_embedding.size(-1))
        xd = xd_bar.transpose(1, 2)

        scores = self.kernel((xt, xt_bar), (xd, xd_bar)) * (self.sigmoid((xt, xt_bar), (xd, xd_bar)))

        # scores = self.kernel((xt, xt_bar), (xd, xd_bar))

        scores = scores.masked_fill_(subsequent_mask == 0, value=-1e10)
        # scores = scores.masked_fill(scores == 0, value=-1000)
        self.scores = scores
        scores = F.softmax(scores, dim=-1)

        self.t_diff = (xt - xt_bar)

        return scores, embedding, self.t_diff


class Decoder(nn.Module):
    """ A non parametric decoder. """

    def __init__(self,
                 num_types, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.GAN = GenerativeAdversarialNetwork(num_types, d_model, dropout)

    def forward(self, hidden):
        return self.GAN(hidden)


class GenerativeAdversarialNetwork(nn.Module):

    def __init__(self,
                 num_types, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types

        self.generator = Generator(num_types, d_model, dropout)

    def forward(self, hidden):
        return self.generator(hidden)


class Generator(nn.Module):

    def __init__(self,
                 num_types, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types

        self.input_weights = nn.Linear(d_model, d_model, bias=False)
        self.noise_weights = nn.Linear(d_model, d_model, bias=False)
        self.event_time_calculator = nn.Linear(d_model, 1, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        b_n, s_n, h_n = hidden.size()
        noise = torch.rand((b_n, s_n, h_n), device=hidden.device)
        hidden_sample = torch.relu(self.noise_weights(noise) + self.input_weights(hidden))
        hidden_sample = self.dropout(hidden_sample)

        return nn.functional.softplus(self.event_time_calculator(hidden_sample)).squeeze(-1)


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    subsequent_mask = (subsequent_mask - 1) ** 2
    return subsequent_mask


class SigmoidGate(nn.Module):

    def __init__(self,
                 num_types, d_model, t_max=200, norm=1):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.t_max = t_max
        self.norm = 1
        self.scores = None
        if num_types == 1:
            # self.params = nn.Parameter(torch.tensor([0.1, 0.1]), requires_grad=True)
            self.params = nn.Sequential(nn.Linear(d_model, 2, bias=True), nn.Sigmoid())

        else:
            self.params = nn.Sequential(nn.Linear(d_model * 2, 2, bias=True), nn.Sigmoid())

    def forward(self, x, type_emb):

        d = (torch.abs(x[0] - x[1]) / self.t_max) ** self.norm

        if self.num_types == 1:
            space = type_emb[0]
            params = self.params(space)
            l = params[:, :, :, :1].squeeze()
            s = params[:, :, :, 1:].squeeze()
        else:
            l, s = self.params(type_emb[0], type_emb[1])

        self.scores = 1 + 0.5 * torch.tanh((d - l) / s)
        return self.scores

    def get_params(self):

        l, s = torch.sigmoid(self.params)[0], torch.sigmoid(self.params)[1]
        return l, s


class squared_exponential_kernel(nn.Module):

    def __init__(self,
                 num_types, d_model, sigma=1, norm=1):
        super().__init__()

        self.d_model = d_model
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
            self.length_scale = nn.Sequential(clipper(d_model, 1, lower_limit=None, upper_limit=10),
                                              nn.Softplus(beta = 0.2))
        else:
            self.length_scale = nn.Sequential(clipper(d_model * 2, lower_limit=None, upper_limit=10), nn.Softplus())

    def forward(self, x, type_emb):

        d = torch.abs(x[0] - x[1]) ** self.norm
        # # length_scale = nn.functional.softplus(self.length_scale)
        # length_scale = self.length_scale

        if self.num_types == 1:
            space = type_emb[0]

        else:
            space = torch.cat(type_emb[0], type_emb[1], -1)

        self.scores = (self.sigma ** 2) * torch.exp(-(d ** 2) / self.length_scale(space).squeeze(-1) ** 2)
        return self.scores

    def get_params(self, type_embeddings):

        if self.num_types == 1:
            # length_scale = nn.functional.softplus(self.length_scale)
            length_scale = self.length_scale

        else:
            pass

        return {'length_scale': length_scale, 'sigma': self.sigma, 'Norm-P': self.norm}


class clipper(nn.Module):

    def __init__(self, input_size, param_size=1, lower_limit=1.0, upper_limit=10):
        super().__init__()

        self.param = nn.Linear(input_size, param_size, bias=True)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def forward(self, input):
        # return self.param(input).clamp(min=None, max = 20)

        return self.param(input)

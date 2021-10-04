import torch
import torch.nn as nn
import torch.nn.functional as F
from sahp import one_hot_embedding
import math
import sys

sys.path.append('../util')
sys.path.append('util')

import kernel_functions


class gated_tpp(nn.Module):

    def __init__(self,
                 num_types, d_model, d_type=1, dropout=0.1, length_scale=1.0,
                 kernel_type='squared_exponential', s=1.0, l=1.0, p=1.0,
                 alpha=1.0, softmax=False, embed_time=False, timetovec=False,sigma = 1.0):
        super().__init__()

        self.d_model = d_model
        self.d_type = d_type
        self.num_types = num_types
        self.encoder = Encoder(num_types, d_model, d_type, length_scale=length_scale,
                               kernel_type=kernel_type, alpha=alpha, test_softmax=softmax, embed_time=embed_time,
                               timetovec=timetovec, s=s, l=l, p=p,sigma= sigma)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.decoder = Decoder(num_types, d_model, dropout)

    def forward(self, event_type, event_time, arrival_times):
        scores, embeddings, _ = self.encoder(event_type, event_time, arrival_times)
        hidden = torch.matmul(scores, embeddings)
        # hidden = scores.sum(-1).unsqueeze(-1)
        # hidden = torch.matmul(scores, event_time)

        hidden = self.norm(hidden)

        return self.decoder(hidden, embeddings),hidden

    def calculate_loss(self, batch_arrival_times, sampled_arrival_times, batch_types, times, reg_param=1.0,hidden = None ):
        arrival_times = batch_arrival_times[:, 1:]
        sampled_times = sampled_arrival_times[:, :-1]
        ## Check the loss

        # loss = torch.abs(arrival_times - sampled_times)
        loss = (arrival_times - sampled_times)**2
        batch_lengths = (batch_types != 0).sum(-1).to('cpu')  ## Non events are 0
        batch_loss = nn.utils.rnn.pack_padded_sequence(loss.T, batch_lengths - 1, enforce_sorted=False)[0]
        time_loss = batch_loss.sum()

        seq_onehot_types = batch_types -1
        seq_onehot_types[seq_onehot_types < 0] = self.num_types
        seq_onehot_types = one_hot_embedding(seq_onehot_types, self.num_types + 1)

        self.decoder.GAN._calculate_loss(batch_arrival_times, seq_onehot_types, n_mc_samples = 20)



        return time_loss

    def train_epoch(self, dataloader, optimizer, params):

        epoch_loss = 0
        events = 0
        for batch in dataloader:
            optimizer.zero_grad()

            event_time, arrival_time, event_type, _ = map(lambda x: x.to(params.device), batch)
            predicted_times,hidden = self(event_type, event_time, arrival_time)

            batch_loss = self.calculate_loss(arrival_time, predicted_times, event_type, event_time,
                                             reg_param=params.reg_param,hidden = hidden)
            epoch_loss += batch_loss
            events += ((event_type != 0).sum(-1) - 1).sum()

            batch_loss.backward()
            optimizer.step()
        return epoch_loss, events

    def validate_epoch(self, dataloader, device='cpu', reg_param=0):

        epoch_loss = 0
        events = 0
        with torch.no_grad():
            last_errors = []
            all_errors = []
            for batch in dataloader:
                event_time, arrival_time, event_type, _ = map(lambda x: x.to(device), batch)
                predicted_times,hidden = self(event_type, event_time, arrival_time)
                batch_loss = self.calculate_loss(arrival_time, predicted_times, event_type, event_time,
                                                 reg_param=reg_param,hidden = hidden)
                epoch_loss += batch_loss
                events += ((event_type != 0).sum(-1) - 1).sum()

                last_event_index = event_type.sum(-1) - 2
                errors = predicted_times[:, :-1] - arrival_time[:, 1:]
                seq_index = 0

                for idx in last_event_index:
                    last_errors.append(errors[seq_index][idx].unsqueeze(-1))
                    all_errors.append(errors[seq_index][:idx + 1])
            last_errors = torch.cat(last_errors)
            last_RMSE = (last_errors ** 2).mean().sqrt()
            all_errors = torch.cat(all_errors)
            all_RMSE = (all_errors ** 2).mean().sqrt()

        return epoch_loss, events, all_RMSE, last_RMSE


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self,
                 num_types, d_model, d_type=1, length_scale=1.0, kernel_type='squared_exponential',
                 alpha=1.0, s=1.0, l=1.0, p=1,
                 test_softmax=False, embed_time=False, timetovec=False,sigma = 1.0):
        super().__init__()

        self.d_model = d_model
        self.d_type = d_type
        self.num_types = num_types
        if timetovec:
            self.embedding = TimetoVec(d_model)
        else:
            self.embedding = BiasedPositionalEmbedding(d_model, max_len=4096)
        # self.embedding = nn.Embedding(num_types + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(num_types + 1, d_type, padding_idx=0)

        # self.w_k = torch.tensor([math.pow(10000.0, 2.0 * (i) / d_model) for i in range(d_model)])

        self.w_k = torch.tensor([math.pow(10000.0, 2.0 * ((i // 2) + 1) / d_model) for i in range(d_model)])
        self.sigmoid = kernel_functions.SigmoidGate(num_types, d_type, norm=1, l=l, s=s)
        # self.kernel = getattr(kernel_functions, kernel_type +'_kernel')(num_types, d_type, norm=1)
        self.kernel = kernel_functions.rational_quadratic_kernel(num_types, d_type, alpha=alpha,
                                                                 lengthscale=length_scale, p=p,sigma=sigma)
        self.softmax = test_softmax
        self.embed_time = embed_time

    def forward(self, event_type, event_time, arrival_times):
        """ Encode event sequences via kernel functions """

        # Temporal Encoding

        temp_enc = self.embedding(event_type, event_time)

        # temp_enc = self.embedding(event_type, torch.cat([torch.zeros(event_time.size()[0], 1).to(event_time.device), event_time], dim=-1))
        # temp_enc =  (temp_enc[:, 1:, :] - temp_enc[:, :-1, :])
        # temp_enc = self.embedding(event_type) * math.sqrt(self.d_model)

        ## Type Encoding
        type_embedding = self.type_emb(event_type) * math.sqrt(self.d_model)
        ## Scale the embedding with the hidden vector size
        embedding = temp_enc
        # embedding = torch.cat([temp_enc, type_embedding], dim=-1)

        ## Future Masking
        subsequent_mask = get_subsequent_mask(event_type)

        ## Time Scores
        if self.embed_time:
            xt_bar = embedding.unsqueeze(1).expand(embedding.size(
                0), embedding.size(1), embedding.size(1), embedding.size(-1))
            xt = xt_bar.transpose(1, 2)
        else:
            normalized_event_time = event_time
            xt_bar = normalized_event_time.unsqueeze(1). \
                expand(normalized_event_time.size(0), normalized_event_time.size(1), normalized_event_time.size(1))
            xt = xt_bar.transpose(1, 2)

        ## Space  Input
        xd_bar = type_embedding.unsqueeze(1).expand(type_embedding.size(
            0), type_embedding.size(1), type_embedding.size(1), type_embedding.size(-1))
        xd = xd_bar.transpose(1, 2)

        scores = self.kernel((xt, xt_bar), (xd, xd_bar)) * (self.sigmoid((xt, xt_bar), (xd, xd_bar)))
        # scores = self.kernel((xt, xt_bar), (xd - xd_bar))
        if self.softmax:
            scores = scores.masked_fill_(subsequent_mask == 0, value=-1e5)
            scores = F.softmax(scores, dim=-1)

        else:
            scores = scores.masked_fill_(subsequent_mask == 0, value=0)  ### BUGGGG ?????
        self.scores = scores
        self.t_diff = (xt - xt_bar)

        return scores, embedding, self.t_diff


class Decoder(nn.Module):
    """ A non parametric decoder. """

    def __init__(self,
                 num_types, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        # self.GAN = GenerativeAdversarialNetwork(num_types, d_model, dropout)

        self.GAN = Intensity_Function(num_types, d_model)


    def forward(self, hidden, temp_encoding=None):

        return self.GAN(hidden)


class GenerativeAdversarialNetwork(nn.Module):

    def __init__(self,
                 num_types, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types

        # self.input_weights = nn.Linear(d_model, d_model, bias=False)
        # self.noise_weights = nn.Linear(d_model, d_model, bias=False)

        self.mean = None
        self.std = None
        self.input_weights = nn.Linear(d_model, d_model, bias=False)
        self.noise_weights = nn.Linear(d_model, d_model, bias=False)

        self.event_time_calculator = nn.Linear(d_model, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        b_n, s_n, h_n = hidden.size()
        sample = 50

        noise = torch.rand((b_n, s_n,sample, h_n), device=hidden.device)
        # hidden_sample = torch.relu(self.noise_weights(noise) + self.input_weights(hidden))
        hidden_sample = torch.relu(self.input_weights(hidden))
        # hidden_sample = self.dropout(hidden_sample)

        noise_sampled = self.noise_weights(noise)
        hidden_samples = torch.relu(noise_sampled + self.input_weights(hidden)[:, :, None, :])
        mean = nn.functional.softplus(self.event_time_calculator(hidden_samples)).squeeze(-1).mean(-1)
        std = nn.functional.softplus(self.event_time_calculator(hidden_samples)).squeeze(-1).std(-1)

        return mean


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


class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

        self.Wt = nn.Linear(1, d_model // 2, bias=False)

    def forward(self, x, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        aa = len(x.size())
        if aa > 1:
            length = x.size(1)
        else:
            length = x.size(0)

        # pe = torch.zeros(length, len(self.Wt.weight)).float()
        arc = (self.position[:length] * self.div_term).unsqueeze(0)
        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


class TimetoVec(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.Wt = nn.Linear(1, d_model, bias=True)
        self.d_model = d_model

    def forward(self, eventype, time):
        t2v = self.Wt(time.unsqueeze(-1)) / math.sqrt(self.d_model)
        t2v[:, :, 1:] = torch.sin(t2v[:, :, 1:].clone())

        return t2v

        return t2v

class Intensity_Function(nn.Module):

    def __init__(self,
                 num_types, d_model):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.input_size = num_types + 1
        self.gelu  = GELU()

        self.start_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            self.gelu
        )


        self.converge_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            self.gelu
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True)
            , nn.Softplus(beta=10.0)
        )

        self.intensity_layer = nn.Sequential(
            nn.Linear(self.d_model, self.num_types, bias=True)
            , nn.Softplus(beta=1.)
        )

    def forward(self,embed_info):

        self.start_point = self.start_layer(embed_info)
        self.converge_point = self.converge_layer(embed_info)
        self.omega = self.decay_layer(embed_info)



    def state_decay(self, converge_point, start_point, omega, duration_t):
        # * element-wise product
        cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))
        return cell_t


    def _compute_loss(self,dt_seq, seq_onehot_types, n_mc_samples=20):

        cell_t = self.state_decay(self.converge_point, self.start_point, self.omega, dt_seq[:, :, None])
        n_batch = dt_seq.size(0)
        n_times = dt_seq.size(1) - 1
        device = dt_seq.device
        # Get the intensity process
        intens_at_evs = self.intensity_layer(cell_t)
        # print(intens_at_evs.shape)
        # intens_at_evs = nn.utils.rnn.pad_sequence(
        #     intens_at_evs, padding_value=1.0, batch_first=True)  # pad with 0 to get rid of the non-events, log1=0

        log_intensities = intens_at_evs.log()  # log intensities
        log_intensities = log_intensities * seq_onehot_types.sum(dim=-1).unsqueeze(-1)

        seq_mask = seq_onehot_types[0,:,:-1]!=0
        log_sum = (log_intensities * seq_mask).sum(dim=(2, 1))  # shape batch

        taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)  # self.process_dim replaced 1
        taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)

        cell_tau = self.state_decay(
            self.converge_point[:, :, :, None],
            self.start_point[:, :, :, None],
            self.omega[:, :, :, None],
            taus)
        cell_tau = cell_tau.transpose(2, 3)
        intens_at_samples = self.intensity_layer(cell_tau).transpose(2, 3)
        # intens_at_samples = nn.utils.rnn.pad_sequence(
        #     intens_at_samples, padding_value=0.0, batch_first=True)

        intens_at_samples = intens_at_samples * seq_onehot_types.sum(dim=-1).unsqueeze(-1).unsqueeze(-1)

        total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC
        partial_integrals = dt_seq * total_intens_samples.mean(dim=2)

        integral_ = partial_integrals.sum(dim=1)

        res = torch.sum(- log_sum + integral_)
        return res



class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
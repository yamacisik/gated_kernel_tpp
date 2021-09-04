import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

sys.path.append('../util')
sys.path.append('util')

import kernel_functions


class gated_tpp(nn.Module):

    def __init__(self,
                 num_types, d_model, d_type=1, t_max=200, dropout=0.1, length_scale=1.0,
                 kernel_type='squared_exponential',
                 alpha=1.0, softmax=False):
        super().__init__()

        self.d_model = d_model
        self.d_type = d_type
        self.num_types = num_types
        self.encoder = Encoder(num_types, d_model, d_type, t_max=t_max, length_scale=length_scale,
                               kernel_type=kernel_type, alpha=alpha, test_softmax=softmax)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.decoder = Decoder(num_types, d_model, dropout)

    def forward(self, event_type, event_time, arrival_times):
        scores, embeddings, _ = self.encoder(event_type, event_time)
        hidden = torch.matmul(scores, embeddings)
        hidden = self.norm(hidden)

        return self.decoder(hidden, arrival_times)

    def calculate_loss(self, batch_arrival_times, sampled_arrival_times, batch_types, reg_param=1.0):
        arrival_times = batch_arrival_times[:, 1:]
        sampled_times = sampled_arrival_times[:, :-1]
        ## Check the loss

        # loss = torch.abs(arrival_times - sampled_times)
        loss = (arrival_times - sampled_times)**2
        batch_lengths = (batch_types != 0).sum(-1).to('cpu')  ## Non events are 0

        batch_loss = nn.utils.rnn.pack_padded_sequence(loss.T, batch_lengths - 1, enforce_sorted=False)[0]

        time_loss = batch_loss.sum()

        ## Add Param Regularizer 1-D
        type_emb = self.encoder.type_emb.weight * math.sqrt(self.d_model)
        param_size = 0
        # for space in type_emb[1:]:
        #     param_size += torch.abs(self.encoder.kernel.alpha[0](space))
        # param_size = -param_size.sum()
        loss = time_loss + reg_param * param_size

        return time_loss

    def train_epoch(self, dataloader, optimizer, params):

        epoch_loss = 0
        events = 0
        for batch in dataloader:
            optimizer.zero_grad()

            event_time, arrival_time, event_type, _ = map(lambda x: x.to(params.device), batch)
            predicted_times = self(event_type, event_time, arrival_time)

            batch_loss = self.calculate_loss(arrival_time, predicted_times, event_type, reg_param=params.reg_param)
            epoch_loss += batch_loss
            events += ((event_type != 0).sum(-1) - 1).sum()

            batch_loss.backward()
            optimizer.step()
        return epoch_loss, events

    def validate_epoch(self, dataloader, device ='cpu',reg_param = 0):

        epoch_loss = 0
        events = 0
        with torch.no_grad():
            last_errors = []
            all_errors = []
            for batch in dataloader:
                event_time, arrival_time, event_type, _ = map(lambda x: x.to(device), batch)
                predicted_times = self(event_type, event_time, arrival_time)
                batch_loss = self.calculate_loss(arrival_time, predicted_times, event_type, reg_param=reg_param)
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
                 num_types, d_model, d_type=1, t_max=200, length_scale=1.0, kernel_type='squared_exponential',
                 alpha=1.0,
                 test_softmax=False):
        super().__init__()

        self.d_model = d_model
        self.d_type = d_type
        self.num_types = num_types

        self.type_emb = nn.Embedding(num_types + 1, d_type, padding_idx=0)
        # self.w_k = torch.tensor([math.pow(10000.0, 2.0 * (i) / d_model) for i in range(d_model)])

        self.w_k = torch.tensor([math.pow(10000.0, 2.0 * ((i//2)+1) / d_model) for i in range(d_model)])
        # self.sigmoid = kernel_functions.SigmoidGate(num_types, d_type, norm=1, t_max=t_max)
        # self.kernel = getattr(kernel_functions, kernel_type +'_kernel')(num_types, d_type, norm=1, t_max=t_max,alpha= alpha)
        self.kernel = kernel_functions.fixed_kernel(t_max=t_max, length_scale=length_scale,alpha = alpha)
        self.t_max = t_max
        self.softmax = test_softmax

    def forward(self, event_type, event_time):
        """ Encode event sequences via kernel functions """

        # Temporal Encoding
        temp_enc = event_time.unsqueeze(-1) / self.w_k.to(event_time.device)
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

        # scores = self.kernel((xt, xt_bar), (xd, xd_bar)) * (self.sigmoid((xt, xt_bar), (xd, xd_bar)))
        # scores = self.kernel((xt, xt_bar), (xd, xd_bar))
        scores = self.kernel((xt, xt_bar))
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
        self.GAN = GenerativeAdversarialNetwork(num_types, d_model, dropout)

    def forward(self, hidden, arrival_time=None):
        # hidden = torch.cat([arrival_time.unsqueeze(-1), hidden], -1)
        return self.GAN(hidden)


class GenerativeAdversarialNetwork(nn.Module):

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
        # hidden_sample = torch.relu(self.noise_weights(noise) + self.input_weights(hidden))
        hidden_sample = torch.relu(self.input_weights(hidden))
        # hidden_sample = self.dropout(hidden_sample)

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

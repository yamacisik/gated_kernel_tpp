import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from sklearn.metrics import f1_score
sys.path.append('../util')
sys.path.append('util')

import kernel_functions


class gated_tpp(nn.Module):

    def __init__(self,
                 num_types, d_model, d_type=1, dropout=0.1, length_scale=1.0,
                 kernel_type='squared_exponential', s=1.0, l=1.0, p=1.0,regulizing_param=5,betas = [0.4,0.3,1.0],
                 alpha=1.0, softmax=False, embed_time=False, timetovec=False,sigma = 1.0):
        super().__init__()

        self.d_model = d_model
        self.d_type = d_type
        self.num_types = num_types
        self.encoder = Encoder(num_types, d_model, d_type, length_scale=length_scale,
                               kernel_type=kernel_type, alpha=alpha, test_softmax=softmax, embed_time=embed_time,
                               timetovec=timetovec, s=s, l=l, p=p,sigma= sigma,regulizing_param=regulizing_param,betas = betas)
        self.norm = nn.LayerNorm(d_model+d_type, eps=1e-6)
        self.decoder = Decoder(num_types, d_model+d_type, dropout)

    def forward(self, event_type, event_time, arrival_times):
        scores, embeddings, _ = self.encoder(event_type, event_time, arrival_times)
        hidden = torch.matmul(scores, embeddings)
        # hidden = scores.sum(-1).unsqueeze(-1)
        # hidden = torch.matmul(scores, event_time)

        hidden = self.norm(hidden)

        return self.decoder(hidden, embeddings)

    def calculate_loss(self, batch_arrival_times, sampled_arrival_times, batch_types,batch_probs,event_time,regularize = False):
        arrival_times = batch_arrival_times[:, 1:]
        sampled_times = sampled_arrival_times[:, :-1]
        ## Check the loss

        loss = torch.abs(arrival_times - sampled_times)
        # loss = (arrival_times - sampled_times)
        seq_length_mask = (batch_types[:,1:] != 0)*1
        batch_loss = loss*seq_length_mask
        time_loss = batch_loss.sum()

        non_event_mask_prob = torch.ones((batch_probs.size(0),batch_probs.size(1),1)).to(batch_arrival_times.device)
        probs = torch.cat([non_event_mask_prob,batch_probs],dim = -1)
        one_hot_encodings = one_hot_embedding(batch_types[:, 1:],self.num_types+1)

        # print(one_hot_encodings*torch.log(probs[:,:-1,:]))
        cross_entropy_loss =-(one_hot_encodings*torch.log(probs[:,:-1,:])).sum(-1)
        cross_entropy_loss = cross_entropy_loss * seq_length_mask
        mark_loss = cross_entropy_loss.sum()
        param_loss = self.encoder.kernel.param_loss

        nll_loss = 0
        if regularize:
        ## NLL Loss:
            device = batch_arrival_times.device
            embeddings = self.encoder.type_emb(batch_types)
            sample_intensities = kernel_functions.get_sample_intensities(self.encoder.kernel,event_time, batch_arrival_times, batch_types,
                                                                         device=device, embeddings=embeddings)
            sample_intensities[sample_intensities == 0] = 1
            sample_intensities = sample_intensities.log().sum(-1)
            non_event_intensities = kernel_functions.get_non_event_intensities(self.encoder.kernel,  event_time, batch_arrival_times,
                                                                               batch_types,
                                  type_embeddings=self.encoder.type_emb, device=device,mc_sample_size = 5)
            nll_loss = -(sample_intensities-non_event_intensities).sum()
        return time_loss+mark_loss,nll_loss,param_loss

    def train_epoch(self, dataloader, optimizer, params):

        epoch_loss = 0
        events = 0
        for batch in dataloader:
            optimizer.zero_grad()

            event_time, arrival_time, event_type, _ = map(lambda x: x.to(params.device), batch)
            predicted_times,probs = self(event_type, event_time, arrival_time)

            batch_loss,nll_loss,param_loss= self.calculate_loss(arrival_time, predicted_times, event_type,probs,event_time,regularize = params.regularize)
            batch_loss = batch_loss+nll_loss+param_loss

            epoch_loss += batch_loss.item()
            events += ((event_type != 0).sum(-1) - 1).sum()
            # nll_loss.backward()
            # optimizer.zero_grad()
            # self.encoder.kernel.lengthscale[0].weight = self.encoder.kernel.lengthscale[0].weight.clamp(-0.5,0.5)
            # self.encoder.kernel.alpha[0].weight = self.encoder.kernel.lengthscale[0].weight.clamp(-0.5, 0.5)
            batch_loss.backward()

            optimizer.step()
        return epoch_loss, events

    def validate_epoch(self, dataloader, device='cpu', reg_param=0,regularize = False):

        epoch_loss = 0
        events = 0
        with torch.no_grad():
            last_errors = []
            all_errors = []
            all_predicted_type = []
            all_actual_type=  []
            accuracy = 0
            for batch in dataloader:
                event_time, arrival_time, event_type, _ = map(lambda x: x.to(device), batch)
                predicted_times,probs = self(event_type, event_time, arrival_time)
                # predicted_times = torch.ones(arrival_time.size()).to(arrival_time.device)

                batch_loss,nll_loss,param_loss = self.calculate_loss(arrival_time, predicted_times, event_type, probs,event_time,regularize = regularize)


                batch_loss = batch_loss + nll_loss+param_loss
                epoch_loss += batch_loss
                events += ((event_type != 0).sum(-1) - 1).sum()

                last_event_index = (event_type != 0).sum(-1) - 2
                errors = predicted_times[:, :-1] - arrival_time[:, 1:]
                seq_index = 0

                predicted_events = torch.argmax(probs,dim = -1)+1 ## Events go from 1 to N in the dataset
                type_prediction_hits = (predicted_events[:, :-1] ==event_type[:, 1:])*1

                ## Clean Up TO DO
                actual_type = event_type[:, 1:]
                predicted_type =predicted_events[:, :-1]
                for idx in last_event_index:
                    last_errors.append(errors[seq_index][idx].unsqueeze(-1))
                    all_errors.append(errors[seq_index][:idx + 1])
                    all_predicted_type.append(predicted_type[seq_index][idx].item())
                    all_actual_type.append(actual_type[seq_index][idx].item())
                    accuracy+=type_prediction_hits[seq_index][idx].item()

            last_errors = torch.cat(last_errors)
            last_RMSE = (last_errors ** 2).mean().sqrt()
            all_errors = torch.cat(all_errors)
            all_RMSE = (all_errors ** 2).mean().sqrt()
            last_event_accuracy = accuracy/len(dataloader.dataset.event_type)

            f_score = f1_score(all_actual_type, all_predicted_type, average='micro')

            print(f'Micro F-1:{f_score}')
        return epoch_loss, events, all_RMSE, last_RMSE,last_event_accuracy


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self,
                 num_types, d_model, d_type=1, length_scale=1.0, kernel_type='squared_exponential',
                 alpha=1.0, s=1.0, l=1.0, p=1,
                 test_softmax=False, embed_time=False, timetovec=False,sigma = 1.0,
                 regulizing_param=5,betas = [0.4,0.3,1.0]):
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
        self.type_emb_prediction = nn.Embedding(num_types + 1, d_type, padding_idx=0)

        # self.w_k = torch.tensor([math.pow(10000.0, 2.0 * (i) / d_model) for i in range(d_model)])

        self.w_k = torch.tensor([math.pow(10000.0, 2.0 * ((i // 2) + 1) / d_model) for i in range(d_model)])
        self.sigmoid = kernel_functions.rational_quadratic_kernel(num_types, d_type)
        # self.kernel = getattr(kernel_functions, kernel_type +'_kernel')(num_types, d_type, norm=1)
        self.kernel = kernel_functions.magic_kernel(num_types, d_type,regulizing_param = regulizing_param,betas = betas)
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
        type_embedding = self.type_emb(event_type)
        xd_bar, xd = get_pairwise_type_embeddings(type_embedding)
        combined_embeddings = torch.cat([xd_bar, xd], dim=-1)


        xt_bar, xt = get_pairwise_times(event_time)
        t_diff = torch.abs(xt_bar - xt)
        ## Scale the embedding with the hidden vector size
        if self.num_types ==1:
            hidden_vector = temp_enc
        else:
            hidden_vector = torch.cat([temp_enc, type_embedding],dim =-1)

        ## Future Masking
        subsequent_mask = get_subsequent_mask(event_type)

        # scores = self.kernel((xt, xt_bar), (xd, xd_bar)) * (self.sigmoid((xt, xt_bar), (xd, xd_bar)))
        scores = self.kernel(t_diff,combined_embeddings)
        if self.softmax:
            scores = scores.masked_fill_(subsequent_mask == 0, value=-1e7)
            scores = F.softmax(scores, dim=-1)

        else:
            scores = scores.masked_fill_(subsequent_mask == 0, value=0)  ### BUGGGG ?????

        self.scores = scores

        return scores, hidden_vector, t_diff


class Decoder(nn.Module):
    """ A non parametric decoder. """

    def __init__(self,
                 num_types, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.GAN = GenerativeAdversarialNetwork(num_types, d_model, dropout)

        # self.GAN = Intensity_Function(num_types, d_model)


    def forward(self, hidden, temp_encoding=None):

        return self.GAN(hidden)


class GenerativeAdversarialNetwork(nn.Module):

    def __init__(self,
                 num_types, d_model, dropout=0.1,layers=1,sample_size = 50):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.samples = sample_size
        self.layers = layers
        # self.input_weights = nn.Linear(d_model, d_model, bias=False)
        # self.noise_weights = nn.Linear(d_model, d_model, bias=False)

        self.mean = None
        self.std = None
        self.input_weights = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for i in range(layers)])
        self.noise_weights = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for i in range(layers)])

        self.event_time_calculator = nn.Linear(d_model, 1, bias=False)
        self.event_type_predictor = nn.Sequential(nn.Linear(d_model, num_types, bias=False))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        b_n, s_n, h_n = hidden.size()
        sample = self.samples

        mark_probs = F.softmax(self.event_type_predictor(hidden), -1)

        for i in range(self.layers):
            noise = torch.rand((b_n, s_n, sample, h_n), device=hidden.device)
            noise_sampled = self.noise_weights[i](noise)
            hidden = torch.relu(noise_sampled + self.input_weights[i](hidden)[:, :, None, :])

        # noise = torch.rand((b_n, s_n,sample, h_n), device=hidden.device)
        # # hidden_sample = torch.relu(self.noise_weights(noise) + self.input_weights(hidden))
        # hidden_sample = torch.relu(self.input_weights(hidden))
        # hidden_sample = self.dropout(hidden_sample)

        # noise_sampled = self.noise_weights(noise)
        # hidden_samples = torch.relu(noise_sampled + self.input_weights(hidden)[:, :, None, :])
        mean = nn.functional.softplus(self.event_time_calculator(hidden)).squeeze(-1).mean(-1)
        std = nn.functional.softplus(self.event_time_calculator(hidden)).squeeze(-1).std(-1)





        return mean,mark_probs


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
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=0)
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
        n_times = dt_seq.size(1)
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


def one_hot_embedding(labels,num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.
    Args:
        labels: class labels, sized [N,].
        num_classes: number of classes.
    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    device = labels.device
    y = torch.eye(num_classes).to(device)
    return y[labels]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
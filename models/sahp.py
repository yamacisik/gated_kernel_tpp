'''
self-attentive Hawkes process
'''
import argparse
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math, copy
from torch import Tensor
from matplotlib import pyplot as plt
import torch.optim as optim
import random
import tqdm
from torch import autograd
import time

class SAHP(nn.Module):
    "Generic N layer attentive Hawkes with masking"

    def __init__(self, nLayers, d_model, atten_heads, dropout, process_dim, device, max_sequence_length):
        super(SAHP, self).__init__()
        self.nLayers = nLayers
        self.process_dim = process_dim
        self.input_size = process_dim + 1
        self.query_size = d_model // atten_heads
        self.device = device
        self.gelu = GELU()

        self.d_model = d_model
        self.type_emb = TypeEmbedding(self.input_size, d_model, padding_idx=self.process_dim)
        self.position_emb = BiasedPositionalEmbedding(d_model=d_model, max_len=max_sequence_length)

        self.attention = MultiHeadedAttention(h=atten_heads, d_model=self.d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_model * 4, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=self.d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=self.d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

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
            nn.Linear(self.d_model, self.process_dim, bias=True)
            , nn.Softplus(beta=1.)
        )

    def state_decay(self, converge_point, start_point, omega, duration_t):
        # * element-wise product
        cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))
        return cell_t

    def forward(self, seq_dt, seq_types, src_mask):
        type_embedding = self.type_emb(seq_types) * math.sqrt(self.d_model)  #
        position_embedding = self.position_emb(seq_types, seq_dt)

        x = type_embedding + position_embedding
        for i in range(self.nLayers):
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=src_mask))
            x = self.dropout(self.output_sublayer(x, self.feed_forward))

        embed_info = x

        self.start_point = self.start_layer(embed_info)
        self.converge_point = self.converge_layer(embed_info)
        self.omega = self.decay_layer(embed_info)

    def compute_loss(self, seq_times, seq_onehot_types, n_mc_samples=20):
        """
        Compute the negative log-likelihood as a loss function.

        Args:
            seq_times: event occurrence timestamps
            seq_onehot_types: types of events in the sequence, one hot encoded
            batch_sizes: batch sizes for each event sequence tensor, by length
            tmax: temporal horizon

        Returns:
            log-likelihood of the event times under the learned parameters

        Shape:
            one-element tensor
        """

        dt_seq = seq_times[:, 1:] - seq_times[:, :-1]
        cell_t = self.state_decay(self.converge_point, self.start_point, self.omega, dt_seq[:, :, None])

        n_batch = seq_times.size(0)
        n_times = seq_times.size(1) - 1
        device = dt_seq.device
        # Get the intensity process
        intens_at_evs = self.intensity_layer(cell_t)
        # print(intens_at_evs.shape)
        # intens_at_evs = nn.utils.rnn.pad_sequence(
        #     intens_at_evs, padding_value=1.0, batch_first=True)  # pad with 0 to get rid of the non-events, log1=0

        log_intensities = intens_at_evs.log()  # log intensities
        log_intensities = log_intensities * seq_onehot_types[:, 1:, :].sum(dim=-1).unsqueeze(-1)

        seq_mask = seq_onehot_types[:, 1:]
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

        intens_at_samples = intens_at_samples * seq_onehot_types[:, 1:, :].sum(dim=-1).unsqueeze(-1).unsqueeze(-1)

        total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC
        partial_integrals = dt_seq * total_intens_samples.mean(dim=2)

        integral_ = partial_integrals.sum(dim=1)

        res = torch.sum(- log_sum + integral_)
        return res

    def read_predict(self, seq_times, seq_types, seq_lengths, pad, device,
                     hmax=40, n_samples=1000, plot=False, print_info=False):
        """
        Read an event sequence and predict the next event time and type.

        Args:
            seq_times: # start from 0
            seq_types:
            seq_lengths:
            hmax:
            plot:
            print_info:

        Returns:

        """

        length = seq_lengths.item()  # exclude the first added event

        ## remove the first added event
        dt_seq = seq_times[1:] - seq_times[:-1]
        last_t = seq_times[length - 1]
        next_t = seq_times[length]

        dt_seq_valid = dt_seq[:length]  # exclude the last timestamp
        dt_seq_used = dt_seq_valid[:length - 1]  # exclude the last timestamp
        next_dt = dt_seq_valid[length - 1]

        seq_types_valid = seq_types[1:length + 1]  # include the first added event
        last_type = seq_types[length - 1]
        next_type = seq_types[length]
        if next_type == self.process_dim:
            print('Error: wrong next event type')
        seq_types_used = seq_types_valid[:-1]
        seq_types_valid_masked = MaskBatch(seq_types_used[None, :], pad, device)
        seq_types_used_mask = seq_types_valid_masked.src_mask

        with torch.no_grad():
            self.forward(dt_seq_used, seq_types_used, seq_types_used_mask)

            if self.omega.shape[1] == 0:  # only one element
                estimate_dt, next_dt, error_dt, next_type, estimate_type = 0, 0, 0, 0, 0
                return estimate_dt, next_dt, error_dt, next_type, estimate_type

            elif self.omega.shape[1] == 1:  # only one element
                converge_point = torch.squeeze(self.converge_point)[None, :]
                start_point = torch.squeeze(self.start_point)[None, :]
                omega = torch.squeeze(self.omega)[None, :]
            else:
                converge_point = torch.squeeze(self.converge_point)[-1, :]
                start_point = torch.squeeze(self.start_point)[-1, :]
                omega = torch.squeeze(self.omega)[-1, :]

            dt_vals = torch.linspace(0, hmax, n_samples + 1).to(device)
            h_t_vals = self.state_decay(converge_point,
                                        start_point,
                                        omega,
                                        dt_vals[:, None])
            if print_info:
                print("last event: time {:.3f} type {:.3f}"
                      .format(last_t.item(), last_type.item()))
                print("next event: time {:.3f} type {:.3f}, in {:.3f}"
                      .format(next_t.item(), next_type.item(), next_dt.item()))

            return predict_from_hidden(self, h_t_vals, dt_vals, next_dt, next_type,
                                       plot, hmax, n_samples, print_info)

def predict_from_hidden(model, h_t_vals, dt_vals, next_dt, next_type, plot, hmax: float = 40.,
                        n_samples=1000, print_info: bool = False):
    model.eval()
    timestep = hmax / n_samples

    intens_t_vals: Tensor = model.intensity_layer(h_t_vals)
    intens_t_vals_sum = intens_t_vals.sum(dim=1)
    integral_ = torch.cumsum(timestep * intens_t_vals_sum, dim=0)
    # density for the time-until-next-event law
    density = intens_t_vals_sum * torch.exp(-integral_)
    # Check density
    if print_info:
        print("sum of density:", (timestep * density).sum())
    t_pit = dt_vals * density  # integrand for the time estimator
    ratio = intens_t_vals / intens_t_vals_sum[:, None]
    prob_type = ratio * density[:, None]  # integrand for the types
    # trapeze method
    estimate_dt = (timestep * 0.5 * (t_pit[1:] + t_pit[:-1])).sum()
    estimate_type_prob = (timestep * 0.5 * (prob_type[1:] + prob_type[:-1])).sum(dim=0)
    if print_info:
        print("type probabilities:", estimate_type_prob)
    estimate_type = torch.argmax(estimate_type_prob)
    # next_dt += 1e-5
    # error_dt = ((estimate_dt - next_dt)/next_dt)** 2#, normalization, np.abs,
    error_dt = ((estimate_dt - next_dt))** 2#, normalization, np.abs,

    if plot:
        process_dim = model.process_dim
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        ax0.plot(dt_vals.numpy(), density.numpy(),
                 linestyle='-', linewidth=.8)
        ax0.set_title("Probability density $p_i(u)$\nof the next increment")
        ax0.set_xlabel("Time $u$")
        ax0.set_ylabel('density $p_i(u)$')
        ylims = ax0.get_ylim()
        ax0.vlines(estimate_dt.item(), *ylims,
                   linestyle='--', linewidth=.7, color='red',
                   label=r'estimate $\hat{t}_i - t_{i-1}$')
        ax0.vlines(next_dt.item(), *ylims,
                   linestyle='--', linewidth=.7, color='green',
                   label=r'true $t_i - t_{i-1}$')
        ax0.set_ylim(ylims)
        ax0.legend()
        ax1.plot(dt_vals.numpy(), intens_t_vals_sum.numpy(),
                 linestyle='-', linewidth=.7, label=r'total intensity $\bar\lambda$')
        for k in range(process_dim):
            ax1.plot(dt_vals.numpy(), intens_t_vals[:, k].numpy(),
                     label='type {}'.format(k),
                     linestyle='--', linewidth=.7)
        ax1.set_title("Intensities")
        ax1.set_xlabel("Time $t$")
        ax1.legend()
        # definite integral of the density
        return (estimate_dt, next_dt, error_dt, next_type, estimate_type), fig
    return estimate_dt, next_dt, error_dt, next_type, estimate_type



def make_model(nLayers=4, d_model=16, atten_heads=8, dropout=0.1, process_dim=2,
               device='cpu', pe='concat', max_sequence_length=4096):
    "helper: construct a models form hyper parameters"

    model = SAHP(nLayers, d_model, atten_heads, dropout=dropout, process_dim=process_dim, device=device,
                 max_sequence_length=max_sequence_length)

    # initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1, size, size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa


class MaskBatch():
    "object for holding a batch of data with mask during training"

    def __init__(self, src, pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt, pad, device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask


def l1_loss(model):
    ## l1 loss
    l1 = 0
    for p in model.parameters():
        l1 = l1 + p.abs().sum()
    return l1


def eval_sahp(batch_size, loop_range, seq_lengths, seq_times, seq_types, model, device, lambda_l1=0):
    model.eval()
    epoch_loss = 0
    for i_batch in loop_range:
        batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
            get_batch(batch_size, i_batch, model, seq_lengths, seq_times, seq_types, rnn=False)
        batch_seq_types = batch_seq_types[:, 1:]

        masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim,
                                     device=device)  # exclude the first added event
        model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask)
        nll = model.compute_loss(batch_seq_times, batch_onehot)

        loss = nll
        epoch_loss += loss.detach()
    event_num = torch.sum(seq_lengths).float()
    model.train()
    return event_num, epoch_loss

def get_batch(batch_size, i_batch, model, seq_lengths, seq_times, seq_types, rnn = True):
    start_pos = i_batch
    end_pos = i_batch + batch_size
    batch_seq_lengths = seq_lengths[start_pos:end_pos]
    max_seq_length = batch_seq_lengths[0]
    batch_seq_times = seq_times[start_pos:end_pos, :max_seq_length + 1]
    batch_seq_types = seq_types[start_pos:end_pos, :max_seq_length + 1]
    # Inter-event time intervals
    batch_dt = batch_seq_times[:, 1:] - batch_seq_times[:, :-1]

    batch_onehot = one_hot_embedding(batch_seq_types, model.input_size)
    batch_onehot = batch_onehot[:, :, :model.process_dim]# [1,0], [0,1], [0,0]

    if rnn:
        # Pack the sequences for rnn
        packed_dt = nn.utils.rnn.pack_padded_sequence(batch_dt, batch_seq_lengths, batch_first=True)
        packed_types = nn.utils.rnn.pack_padded_sequence(batch_seq_types, batch_seq_lengths, batch_first=True)
        max_pack_batch_size = packed_dt.batch_sizes[0]
    else:
        # self-attention
        packed_dt,packed_types,max_pack_batch_size = None, None,0
    return batch_onehot, batch_seq_times, batch_dt, batch_seq_types, \
           max_pack_batch_size, packed_dt, packed_types, batch_seq_lengths

def one_hot_embedding(labels: Tensor, num_classes: int) -> torch.Tensor:
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

def get_attentions_sahp(batch_size, loop_range, seq_lengths, seq_times, seq_types, model, device, lambda_l1=0):
    model.eval()
    epoch_loss = 0
    for i_batch in loop_range:
        batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
            get_batch(batch_size, i_batch, model, seq_lengths, seq_times, seq_types, rnn=False)
        batch_seq_types = batch_seq_types[:, 1:]

        masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim,
                                     device=device)  # exclude the first added event
        model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask)
        nll = model.compute_loss(batch_seq_times, batch_onehot)

        loss = nll
        epoch_loss += loss.detach()
    event_num = torch.sum(seq_lengths).float()
    model.train()
    return event_num, epoch_loss

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, initial_lr, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.initial_lr = initial_lr

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.initial_lr + self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def train_eval_sahp(params):
    args, process_dim, device, tmax, \
    train_seq_times, train_seq_types, train_seq_lengths, \
    dev_seq_times, dev_seq_types, dev_seq_lengths, \
    test_seq_times, test_seq_types, test_seq_lengths, \
    batch_size, epoch_num, use_cuda, test_seq_intensities = params

    ## sequence length
    train_seq_lengths, reorder_indices_train = train_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    train_seq_times = train_seq_times[reorder_indices_train]
    train_seq_types = train_seq_types[reorder_indices_train]
    #
    dev_seq_lengths, reorder_indices_dev = dev_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    dev_seq_times = dev_seq_times[reorder_indices_dev]
    dev_seq_types = dev_seq_types[reorder_indices_dev]

    test_seq_lengths, reorder_indices_test = test_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    test_seq_times = test_seq_times[reorder_indices_test]
    test_seq_types = test_seq_types[reorder_indices_test]
    # test_seq_intensities = test_seq_intensities[reorder_indices_test]

    max_sequence_length = max(train_seq_lengths[0], dev_seq_lengths[0], test_seq_lengths[0])
    print('max_sequence_length: {}'.format(max_sequence_length))

    d_model = args.d_model
    atten_heads = args.atten_heads
    dropout = args.dropout

    model = make_model(nLayers=args.nLayers, d_model=d_model, atten_heads=atten_heads,
                       dropout=dropout, process_dim=process_dim, device=device, pe=args.pe,
                       max_sequence_length=max_sequence_length + 1).to(device)

    print("the number of trainable parameters: " + str(count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.lambda_l2)
    model_opt = NoamOpt(args.d_model, 1, 100, initial_lr=args.lr, optimizer=optimizer)

    ## Size of the traing dataset
    train_size = train_seq_times.size(0)
    dev_size = dev_seq_times.size(0)
    test_size = test_seq_times.size(0)
    tr_loop_range = list(range(0, train_size, batch_size))
    de_loop_range = list(range(0, dev_size, batch_size))
    test_loop_range = list(range(0, test_size, batch_size))

    last_dev_loss = 0.0
    early_step = 0

    random_seeds = list(range(0, 1000))
    random.shuffle(random_seeds)

    model.train()
    for epoch in range(epoch_num):
        epoch_start_time = time.time()
        epoch_train_loss = 0.0
        ## training
        random.Random(random_seeds[epoch]).shuffle(tr_loop_range)
        for i_batch in tr_loop_range:
            model_opt.optimizer.zero_grad()

            batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
                get_batch(batch_size, i_batch, model, train_seq_lengths, train_seq_times, train_seq_types,
                               rnn=False)

            batch_seq_types = batch_seq_types[:, 1:]

            masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim,
                                         device=device)  # exclude the first added even
            model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask)
            nll = model.compute_loss(batch_seq_times, batch_onehot)

            loss = nll

            loss.backward()
            model_opt.optimizer.step()

            epoch_train_loss += loss.detach()


        train_event_num = torch.sum(train_seq_lengths).float()
        train_loss =  epoch_train_loss / train_event_num

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        ## dev
        dev_event_num, epoch_dev_loss = eval_sahp(batch_size, de_loop_range, dev_seq_lengths, dev_seq_times,
                                                  dev_seq_types, model, device, args.lambda_l2)

        dev_loss = epoch_dev_loss / dev_event_num

        ## test
        test_event_num, epoch_test_loss = eval_sahp(batch_size, test_loop_range, test_seq_lengths, test_seq_times,
                                                    test_seq_types, model, device, args.lambda_l2)

        test_loss = epoch_test_loss / test_event_num
        # print(f'Epoch:{epoch}, Train Loss:{train_loss:.6f}, Valid Loss:{valid_loss:.6f}, Test Loss:{test_loss:.6f}')
        print(f'Epoch:{epoch}--- Train NLL:{train_loss:.6f}, Valid NLL: {dev_loss:.6f}, Test NLL: {test_loss:.6f},Time: {epoch_time:.4f}')

        ## early stopping
        gap = epoch_dev_loss / dev_event_num - last_dev_loss
        if abs(gap) < args.early_stop_threshold:
            early_step += 1
        last_dev_loss = epoch_dev_loss / dev_event_num

        if early_step >= 3:
            print('Early Stopping')
            # prediction
            # avg_rmse, types_predict_score = \
            #     prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_size, tmax)
            break

    # prediction

    train_rmse, train_accuracy, results = \
        prediction_evaluation(device, model, train_seq_lengths, train_seq_times, train_seq_types, train_size, tmax)
    valid_rmse, valid_accuracy, results = \
        prediction_evaluation(device, model, dev_seq_lengths, dev_seq_times, dev_seq_types, dev_size, tmax)
    test_rmse, test_accuracy, results = \
        prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_size, tmax)


    test_event_num, epoch_test_loss = eval_sahp(batch_size, test_loop_range, test_seq_lengths, test_seq_times,
                                                test_seq_types, model, device, args.lambda_l2)
    final_test_loss = epoch_test_loss / test_event_num
    return model, (train_rmse,valid_rmse,test_rmse), (train_accuracy,valid_accuracy,test_accuracy), final_test_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_size, tmax):
    model.eval()
    test_data = (test_seq_times, test_seq_types, test_seq_lengths)
    incr_estimates, incr_reals, types_real, types_estimates = \
        predict_test(model, *test_data, pad=model.process_dim, device=device,
                                hmax=tmax, use_jupyter=False, rnn=False)
    if device != 'cpu':
        incr_reals = [incr_real.item() for incr_real in incr_reals]
        incr_reals = np.asarray(incr_reals)

        incr_estimates = [incr_estimate.item() for incr_estimate in incr_estimates]
        incr_estimates = np.asarray(incr_estimates)

        types_real = [types_rl.item() for types_rl in types_real]
        types_estimates = [types_esti.item() for types_esti in types_estimates]

    # incr_reals += 1e-5
    # incr_errors = ((incr_estimates - incr_reals) / incr_reals) ** 2  # , normalization, np.abs,
    incr_errors = ((incr_estimates - incr_reals)) ** 2  # , normalization, np.abs,

    avg_rmse = np.sqrt(np.mean(incr_errors), dtype=np.float64)
    print("rmse", avg_rmse)
    mse_var = np.var(incr_errors, dtype=np.float64)

    delta_meth_stderr = 1 / test_size * mse_var / (4 * avg_rmse)

    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    types_predict_score = f1_score(types_real, types_estimates, average='micro')  # preferable in class imbalance
    print("Type prediction score:", types_predict_score)
    # print("Confusion matrix:\n", confusion_matrix(types_real, types_estimates))
    model.train()
    return avg_rmse, types_predict_score, (types_real, types_estimates, incr_reals, incr_estimates)

def process_loaded_sequences(loaded_hawkes_data: dict, process_dim: int):
    """
    Preprocess synthetic Hawkes data by padding the sequences.
    Args:
        loaded_hawkes_data:
        process_dim:
        tmax:
    Returns:
        sequence event times, event types and overall lengths (dim0: batch size)
    """
    # Tensor of sequence lengths (with additional BOS event)
    seq_lengths = torch.Tensor(loaded_hawkes_data['lengths']).int()

    event_times_list = loaded_hawkes_data['timestamps']
    event_types_list = loaded_hawkes_data['types']
    # event_intensities_list = loaded_hawkes_data['intensities'] if 'intensities' in loaded_hawkes_data.keys() else []

    event_times_list = [torch.from_numpy(e) for e in event_times_list]
    event_types_list = [torch.from_numpy(e) for e in event_types_list]

    # if len(event_intensities_list[0])>1:
    #     all_intensities = []
    #     for i in range(len(event_intensities_list[0])):
    #         current_node_intensity = [torch.from_numpy(e[i]) for e in event_intensities_list]
    #         all_intensities.append(current_node_intensity)
    #     event_intensities_list = all_intensities
    # else:
    #     event_intensities_list = [torch.from_numpy(e) for e in event_intensities_list]

    tmax = 0
    for tsr in event_times_list:
        if torch.max(tsr) > tmax:
            tmax = torch.max(tsr)

    #  Build a data tensor by padding
    seq_times = nn.utils.rnn.pad_sequence(event_times_list, batch_first=True, padding_value=tmax).float()
    seq_times = torch.cat((torch.zeros_like(seq_times[:, :1]), seq_times), dim=1) # add 0 to the sequence beginning

    # seq_intensities= nn.utils.rnn.pad_sequence(event_intensities_list, batch_first=True, padding_value=tmax).float()
    # seq_intensities = torch.cat((torch.zeros_like(seq_intensities[:, :1]), seq_intensities), dim=1) # add 0 to the sequence beginning


    seq_types = nn.utils.rnn.pad_sequence(event_types_list, batch_first=True, padding_value=process_dim)
    seq_types = torch.cat(
        (process_dim*torch.ones_like(seq_types[:, :1]), seq_types), dim=1).long()# convert from floattensor to longtensor


    return seq_times, seq_types, seq_lengths, tmax,None

def get_intensities_from_sahp(model, test_data, batch_size=32):
    device = model.device

    test_seq_times, test_seq_types, test_seq_lengths, test_seq_intensities = test_data

    test_seq_lengths, reorder_indices_test = test_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    test_seq_times = test_seq_times[reorder_indices_test]
    test_seq_types = test_seq_types[reorder_indices_test]
    test_seq_intensities = test_seq_intensities[reorder_indices_test]

    test_size = test_seq_times.size(0)
    test_loop_range = list(range(0, test_size, batch_size))

    model.eval()

    all_intensities = []
    all_predicted_intensities = []

    for i_batch in test_loop_range:

        batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
            get_batch(batch_size, i_batch, model, test_seq_lengths, test_seq_times, test_seq_types, rnn=False)
        batch_seq_types = batch_seq_types[:, 1:]
        batch_intensities = test_seq_intensities[i_batch:i_batch + batch_size][:, 1:]

        masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim,
                                     device=device)  # exclude the first added event
        model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask)
        # nll = model.compute_loss(batch_seq_times, batch_onehot)

        type_embedding = model.type_emb(masked_seq_types.src) * math.sqrt(model.d_model)  #
        position_embedding = model.position_emb(masked_seq_types.src, batch_dt)

        x = type_embedding + position_embedding

        dt_seq = batch_seq_times[:, 1:] - batch_seq_times[:, :-1]
        cell_t = model.state_decay(model.converge_point, model.start_point, model.omega, dt_seq[:, :, None])

        n_batch = test_seq_times.size(0)
        n_times = test_seq_times.size(1) - 1
        device = dt_seq.device
        # Get the intensity process
        intens_at_evs = model.intensity_layer(cell_t)
        intens_at_evs = intens_at_evs.squeeze(-1)

        index = 0
        for i in batch_seq_lengths:
            predicted_intensities = intens_at_evs[index, :i]
            actual_intensities = batch_intensities[index, :i]
            all_intensities.append(actual_intensities.cpu().detach().numpy())
            all_predicted_intensities.append(predicted_intensities.cpu().detach().numpy())

            index += 1

    all_predicted_intensities = np.concatenate(all_predicted_intensities)
    all_intensities = np.concatenate(all_intensities)

    return all_predicted_intensities, all_intensities

def predict_test(model, seq_times, seq_types, seq_lengths, pad, device='cpu',
                 hmax: float = 40., use_jupyter: bool = False, rnn: bool = True):
    """Run predictions on testing dataset

    Args:
        seq_lengths:
        seq_types:
        seq_times:
        model:
        hmax:
        use_jupyter:

    Returns:

    """
    incr_estimates = []
    incr_real = []
    incr_errors = []
    types_real = []
    types_estimates = []
    test_size = seq_times.shape[0]
    if use_jupyter:
        index_range_ = tqdm.tnrange(test_size)
    else:
        index_range_ = tqdm.trange(test_size)
    for index_ in index_range_:
        _seq_data = (seq_times[index_],
                     seq_types[index_],
                     seq_lengths[index_])
        if rnn:
            est, real_dt, err, real_type, est_type = model.read_predict(*_seq_data, hmax)
        else:
            est, real_dt, err, real_type, est_type = model.read_predict(*_seq_data, pad, device, hmax)

        if err != err: # is nan
            continue
        incr_estimates.append(est)
        incr_real.append(real_dt)
        incr_errors.append(err)
        types_real.append(real_type)
        types_estimates.append(est_type)

    incr_real = np.asarray(incr_real)
    incr_estimates = np.asarray(incr_estimates)
    types_real = np.asarray(types_real)
    types_estimates = np.asarray(types_estimates)

    return incr_estimates,incr_real, types_real, types_estimates


class TypeEmbedding(nn.Embedding):
    def __init__(self, type_size, embed_size, padding_idx):
        super().__init__(type_size, embed_size, padding_idx=padding_idx)# padding_idx not 0


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

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


class TimetoVec(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.Wt = nn.Linear(1, d_model , bias=True)

    def forward(self, time):

        t2v = self.Wt(time)
        t2v[:,:,1:] = torch.sin(t2v[:,:,1:])

        return t2v

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):

        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))

        scores = torch.exp(torch.matmul(query, key.transpose(-2, -1))) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in models size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=True)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # the same mask applies to all heads
            # unsqueeze Returns a new tensor with a dimension of size one
            # inserted at the specified position.
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation.forward(self.w_1(x))))



class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))



if __name__ == "__main__":

    DEFAULT_BATCH_SIZE = 32
    DEFAULT_HIDDEN_SIZE = 32
    DEFAULT_LEARN_RATE = 1e-4

    parser = argparse.ArgumentParser(description="Train the models.")
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs.')
    parser.add_argument('-b', '--batch', type=int,
                        dest='batch_size', default=DEFAULT_BATCH_SIZE,
                        help='batch size. (default: {})'.format(DEFAULT_BATCH_SIZE))
    parser.add_argument('-lr', '--lr', default=DEFAULT_LEARN_RATE, type=float,
                        help="set the optimizer learning rate. (default {})".format(DEFAULT_LEARN_RATE))
    parser.add_argument('--hidden', type=int,
                        dest='hidden_size', default=DEFAULT_HIDDEN_SIZE,
                        help='number of hidden units. (default: {})'.format(DEFAULT_HIDDEN_SIZE))
    parser.add_argument('-d_model', '--d-model', type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument('--atten-heads', type=int, default=8)
    parser.add_argument('--pe', type=str, default='add', help='concat, add')
    parser.add_argument('--nLayers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='override the size of the training dataset.')
    parser.add_argument('--lambda-l2', type=float, default=3e-4,
                        help='regularization loss.')
    parser.add_argument('--dev-ratio', type=float, default=0.1,
                        help='override the size of the dev dataset.')
    parser.add_argument('--early-stop-threshold', type=float, default=1e-4,
                        help='early_stop_threshold')
    parser.add_argument('--log-dir', type=str,
                        dest='log_dir', default='logs',
                        help="training logs target directory.")
    parser.add_argument('--save_model', default=True,
                        help="do not save the models state dict and loss history.")
    parser.add_argument('--bias', default=False,
                        help="use bias on the activation (intensity) layer.")
    parser.add_argument('--samples', default=10,
                        help="number of samples in the integral.")
    parser.add_argument('-m', '--model', default='sahp',
                        type=str, choices=['sahp'],
                        help='choose which models to train.')
    parser.add_argument('-t', '--task', type=str, default='synthetic',
                        help='task type')
    parser.add_argument('-st', '--synth_task', type=int, default=2,
                        help='task type')
    parser.add_argument('-seed', '--seed', type=int, default=42,
                        help='seed')
    args = parser.parse_args()
    print(args)

    SYNTH_DATA_FILES = ['../../data/simulated/power_hawkes/power_hawkes.pkl',
                        '../../data/simulated/sin_hawkes/sinusodial_1d_hawkes.pkl',
                        '../../data/simulated/2_d_hawkes/hawkes_2d.pkl', ]
    TYPE_SIZE_DICT = {'retweet': 3, 'bookorder': 8, 'meme': 5000, 'mimic': 75, 'stackOverflow': 22,
                      'synthetic': 2}
    REAL_WORLD_TASKS = list(TYPE_SIZE_DICT.keys())[:5]
    SYNTHETIC_TASKS = list(TYPE_SIZE_DICT.keys())[5:]

    start_time = time.time()
    print(SYNTH_DATA_FILES)


    if torch.cuda.is_available():
        use_cuda = True
        device = 'cuda'
        # torch.set_default_tensor_type(cuda_tensor)
        torch.cuda.manual_seed(seed=args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # torch.set_default_tensor_type(cpu_tensor)
        device = 'cpu'
        use_cuda = False

    cuda_num = 'cuda:{}'.format(args.cuda)
    print("Training on device {}".format(device))

    process_dim = TYPE_SIZE_DICT[args.task]

    print("Loading {}-dimensional process.".format(process_dim), end=' \n')

    if args.task in SYNTHETIC_TASKS:

        # print("Available files:")
        #
        # for i, s in enumerate(SYNTH_DATA_FILES):
        #     print("{:<8}{:<8}".format(i, s))


        chosen_file_index = args.synth_task
        chosen_file = SYNTH_DATA_FILES[chosen_file_index]
        print('chosen file:%s' + str(chosen_file))

        with open(chosen_file, 'rb') as f:
            loaded_hawkes_data = pickle.load(f)
        chosen_file_name = str.split(chosen_file,"\\")[-1]
        chosen_file_name = str.split(chosen_file_name, "/")[-1]
        print(chosen_file_name)

        tmax = loaded_hawkes_data['tmax']

        process_dim = loaded_hawkes_data['process_dim'] if 'process_dim' in loaded_hawkes_data.keys() else process_dim
        seq_times, seq_types, seq_lengths, _,seq_intensities = process_loaded_sequences(loaded_hawkes_data, process_dim)

        seq_times = seq_times.to(device)
        seq_types = seq_types.to(device)
        seq_lengths = seq_lengths.to(device)
        # seq_intensities = seq_intensities.to(device)

        total_sample_size = seq_times.size(0)
        print("Total sample size: {}".format(total_sample_size))

        train_ratio = args.train_ratio
        train_size = int(train_ratio * total_sample_size)
        dev_ratio = args.dev_ratio
        dev_size = int(dev_ratio * total_sample_size)
        print("Train sample size: {:}/{:}".format(train_size, total_sample_size))
        print("Dev sample size: {:}/{:}".format(dev_size, total_sample_size))


        # Define training data
        train_times_tensor = seq_times[:train_size]
        train_seq_types = seq_types[:train_size]
        train_seq_lengths = seq_lengths[:train_size]
        print("No. of event tokens in training subset:", train_seq_lengths.sum())

        # Define development data
        dev_times_tensor = seq_times[train_size:train_size + dev_size]  # train_size+dev_size
        dev_seq_types = seq_types[train_size:train_size + dev_size]
        dev_seq_lengths = seq_lengths[train_size:train_size + dev_size]
        print("No. of event tokens in development subset:", dev_seq_lengths.sum())

        test_times_tensor = seq_times[-dev_size:]
        test_seq_types = seq_types[-dev_size:]
        test_seq_lengths = seq_lengths[-dev_size:]
        # test_seq_intensities = seq_intensities[-dev_size:]


        print("No. of event tokens in test subset:", test_seq_lengths.sum())


    else:
        pass


    learning_rate = args.lr
    # Training parameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # with autograd.detect_anomaly():
    params = args, process_dim, device, tmax, \
             train_times_tensor, train_seq_types, train_seq_lengths, \
             dev_times_tensor, dev_seq_types, dev_seq_lengths, \
             test_times_tensor, test_seq_types, test_seq_lengths, \
             BATCH_SIZE, EPOCHS, use_cuda, None
    model, rmse, micro_f1, test_loss = train_eval_sahp(params)

    data,_ = chosen_file_name.split('.')
    model_name = data + '_debug_model'

    torch.save(model.state_dict(), '../trained_models/sahp/' + model_name + '.pt')
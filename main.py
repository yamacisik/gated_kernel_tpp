import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import csv
import math

model_name = str(datetime.now())[:19].replace(':', '_').replace(' ', '_').replace('-', '_')

DATASET_PATHS = {'sin_hawkes': '../data/simulated/sin_hawkes/', 'power_hawkes': '../data/simulated/power_hawkes/',
                 '2_d_hawkes': '../data/simulated/2_d_hawkes/', 'mimic2': '../data/mimic/',
                 'stackOverflow': '../data/stackOverflow/', 'retweet': '../data/retweet/'}

DATASET_EVENT_TYPES = {'sin_hawkes': 1, 'power_hawkes': 1, '2_d_hawkes': 2, 'mimic2': 75, 'stackOverflow': 22,
                       'retweet': 3}

from dataset import get_dataloader
from tqdm import tqdm
import random
from models.gated_tpp import gated_TPP
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('-data', type=str, default='power_hawkes')
parser.add_argument('-epoch', type=int, default=2)
parser.add_argument('-batch_size', type=int, default=40)
parser.add_argument('-d_model', type=int, default=32)
parser.add_argument('-d_type', type=int, default=8)
parser.add_argument('-beta', type=float, default=1.0)
parser.add_argument('-param_reg', type=float, default=1.0)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-l2', type=float, default=0.0001)
parser.add_argument('-seed', type=int, default=42)
parser.add_argument('-save', type=bool, default=True)
parser.add_argument('-normalized', type=bool, default=False)

params = parser.parse_args()

# ------Reproducibility-------

torch.manual_seed(params.seed)
random.seed(params.seed)
np.random.seed(params.seed)

# use_cuda = torch.cuda.is_available()
if torch.cuda.is_available():
    device = 'cuda'
    # torch.set_default_tensor_type(cuda_tensor)
    torch.cuda.manual_seed(seed=params.seed)
    torch.cuda.manual_seed_all(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    # torch.set_default_tensor_type(cpu_tensor)
    device = 'cpu'

params.device = torch.device(device)

print('[Info] parameters: {}'.format(params))

data_path = DATASET_PATHS[params.data]
num_types = DATASET_EVENT_TYPES[params.data]

with open(data_path + 'train.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin-1')
    train_data = data['train']

with open(data_path + 'dev.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin-1')
    dev_data = data['dev']

with open(data_path + 'test.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin-1')
    test_data = data['test']

trainloader = get_dataloader(train_data, params.batch_size, shuffle=True)
testloader = get_dataloader(test_data, 1, shuffle=False)  # 1 makes it easy to calculate RMSE
valloader = get_dataloader(dev_data, 1, shuffle=False)

t_max = max([seq[-1]['time_since_start'] for data in [train_data, dev_data, test_data] for seq in data])
if not params.normalized:
    t_max = 1

model = gated_TPP(num_types, params.d_model, params.d_type, t_max=t_max, dropout=params.dropout, beta=params.beta)  #
optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                       params.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=params.l2)

model = model.to(device)
# torch.nn.init.xavier_uniform_(model.encoder.kernel.length_scale[0].param.weight)

torch.nn.init.xavier_uniform_(model.encoder.kernel.lengthscale[0].weight)
# torch.nn.init.xavier_uniform_(model.encoder.kernel.alpha[0].weight)
# torch.nn.init.xavier_uniform_(model.encoder.sigmoid.params[0].weight)


for epoch in range(params.epoch):
    train_epoch_loss = 0
    train_events = 0
    for batch in trainloader:
        optimizer.zero_grad()

        event_time, arrival_time, event_type, _ = map(lambda x: x.to(params.device), batch)

        predicted_times = model(event_type, event_time)

        batch_loss = model.calculate_loss(arrival_time, predicted_times, event_type, reg_param=params.param_reg)
        train_epoch_loss += batch_loss
        train_events += ((event_type != 0).sum(-1) - 1).sum()

        batch_loss.backward()
        optimizer.step()

    val_epoch_loss = 0
    val_events = 0
    test_epoch_loss = 0
    test_events = 0
    with torch.no_grad():
        val_last_errors = []
        val_all_errors = []
        for batch in valloader:
            event_time, arrival_time, event_type, _ = map(lambda x: x.to(params.device), batch)
            predicted_times = model(event_type, event_time)
            batch_loss = model.calculate_loss(arrival_time, predicted_times, event_type)
            val_epoch_loss += batch_loss
            val_events += ((event_type != 0).sum(-1) - 1).sum()

            last_event_index = event_type.sum(-1) - 2
            errors = predicted_times[:, :-1] - arrival_time[:, 1:]
            seq_index = 0
            for idx in last_event_index:
                val_last_errors.append(errors[seq_index][idx].unsqueeze(-1))
                val_all_errors.append(errors[seq_index][:idx + 1])
        val_last_errors = torch.cat(val_last_errors)
        val_RMSE = (val_last_errors ** 2).mean().sqrt()
        val_all_errors = torch.cat(val_all_errors)
        val_all_RMSE = (val_all_errors ** 2).mean().sqrt()

        test_last_errors = []
        test_all_errors = []
        for batch in testloader:

            event_time, arrival_time, event_type, _ = map(lambda x: x.to(params.device), batch)
            predicted_times = model(event_type, event_time)
            batch_loss = model.calculate_loss(arrival_time, predicted_times, event_type)
            test_epoch_loss += batch_loss
            test_events += ((event_type != 0).sum(-1) - 1).sum()

            last_event_index = event_type.sum(-1) - 2
            errors = predicted_times[:, :-1] - arrival_time[:, 1:]
            seq_index = 0
            for idx in last_event_index:
                test_last_errors.append(errors[seq_index][idx].unsqueeze(-1))
                test_all_errors.append(errors[seq_index][:idx + 1])

        test_last_errors = torch.cat(test_last_errors)
        test_RMSE = (test_last_errors ** 2).mean().sqrt()
        test_all_errors = torch.cat(test_all_errors)
        test_all_RMSE = (test_all_errors ** 2).mean().sqrt()

    train_loss = train_epoch_loss / train_events
    valid_loss = val_epoch_loss / val_events
    test_loss = test_epoch_loss / test_events

    print(f'Epoch:{epoch}, Train Loss:{train_loss:.4f}, Valid Loss:{valid_loss:.4f}, Test Loss:{test_loss:.4f}')

    type_emb = model.encoder.type_emb.weight * math.sqrt(model.d_model)
    lengthscale = model.encoder.kernel.lengthscale(type_emb)[-1]
    # alpha  =model.encoder.kernel.alpha(type_emb)[-1]
    # print(f'Alpha: {alpha.item():.4f}, Length Scale: {lengthscale.item():.4f}')
    print(f'Length Scale: {lengthscale.item():.4f}')
    alpha = 'NAN'

print(f' Valid Last Event RMSE:{val_RMSE:.4f}, Test Last Event RMSE:{test_RMSE:.4f}')
print(f' Valid All Event RMSE:{val_all_RMSE:.4f}, Test All Event RMSE:{test_all_RMSE:.4f}')

# type_emb=model.encoder.type_emb.weight* math.sqrt(model.d_model)
# lengthscale  =model.encoder.kernel.lengthscale(type_emb)[-1]

# alpha  =model.encoder.kernel.alpha(type_emb)[-1]

# s,l = model.encoder.sigmoid.params(model.encoder.type_emb.weight)[-1]
# alpha,lengthscale = model.encoder.kernel.params(model.encoder.type_emb.weight)[-1]

# print(f'Length Scale: {length_scale.item():.4f}, Gate Param s: {s:.4f}, Gate Param l: {l:.4f}')
# print(f'Alpha: {alpha.item():.4f}, Length Scale: {lengthscale.item():.4f}, Gate Param s: {s:.4f}, Gate Param l: {l:.4f}')

# print(f'Alpha: {alpha.item():.4f}, Length Scale: {lengthscale.item():.4f}')
# print(f'Length Scale: {lengthscale.item():.4f}')

results_to_record = [str(params.data), str(params.epoch), str(params.batch_size), str(params.d_model),
                     str(params.d_type),str(params.lr), str(params.beta), str(valid_loss.item()), str(test_loss.item()),
                     str(val_all_RMSE.item()), str(test_all_RMSE.item()), str(lengthscale.item()), str(alpha),
                     str(params.normalized),str(params.param_reg)]

with open(r'results.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(results_to_record)

model_name = params.data + 'debug_model'
if params.save:
    torch.save(model.state_dict(), 'trained_models/' + model_name + '.pt')

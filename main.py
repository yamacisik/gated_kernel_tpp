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

KERNEL_TYPES = {1: 'squared_exponential', 2: 'rational_quadratic'}
MODELS = {1: 'gated_TPP', 2: 'LogNormMix'}

from dataset import get_dataloader
from tqdm import tqdm
import random
from models.gated_tpp import gated_tpp
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('-data', type=str, default='power_hawkes')
parser.add_argument('-model', type=int, default=1)
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-d_model', type=int, default=22)
parser.add_argument('-d_type', type=int, default=8)
parser.add_argument('-alpha', type=float, default=1.0)
parser.add_argument('-length_scale', type=float, default=1.0)
parser.add_argument('-reg_param', type=float, default=0.0)
parser.add_argument('-kernel_type', type=int, default=1)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-l2', type=float, default=0.0001)
parser.add_argument('-seed', type=int, default=42)
parser.add_argument('-save', type=bool, default=True)
parser.add_argument('-normalized', type=int, default=0)
parser.add_argument('-softmax', type=int, default=0)

params = parser.parse_args()

params.normalized = True if params.normalized == 1 else False
params.softmax = True if params.softmax == 1 else False
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
    'Arrival Times are not normalized...'
    t_max = 1

print(t_max)
model = gated_tpp(num_types, params.d_model, params.d_type, t_max=t_max, dropout=params.dropout,
                  length_scale=params.length_scale,
                  kernel_type=KERNEL_TYPES[params.kernel_type], alpha=params.alpha, softmax=params.softmax)
#
optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                       params.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=params.l2)

model = model.to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# with open('log.txt', 'w') as f:
#     f.write('Epoch, Loss, RMSE_ALL, RMSE_LAST, Lengthscale\n')

for epoch in range(params.epoch):
    train_epoch_loss, train_events = model.train_epoch(trainloader, optimizer, params)
    valid_epoch_loss, valid_events, val_RMSE, val_all_RMSE = model.validate_epoch(valloader, device = params.device,reg_param=params.reg_param)
    test_epoch_loss, test_events, test_RMSE, test_all_RMSE = model.validate_epoch(testloader, device = params.device,reg_param=params.reg_param)

    train_loss = train_epoch_loss / train_events
    valid_loss = valid_epoch_loss / valid_events
    test_loss = test_epoch_loss / test_events

    # type_emb = model.encoder.type_emb.weight * math.sqrt(model.d_model)
    # length_scale = model.encoder.kernel.params(type_emb)[0]['length_scale']
    # with open('log.txt', 'a') as f:
    #     f.write('{epoch}, {loss: 8.5f}, {rmse_all: 8.5f}, {rmse_last: 8.5f}, {length_scale: 8.5f}\n'
    #             .format(epoch=epoch, loss=valid_loss, rmse_all=val_RMSE, rmse_last=val_RMSE,length_scale =length_scale ))

    print(f'Epoch:{epoch}, Train Loss:{train_loss:.4f}, Valid Loss:{valid_loss:.4f}, Test Loss:{test_loss:.4f}')
    # print(model.encoder.kernel.params(type_emb))

print(f' Valid Last Event RMSE:{val_RMSE:.4f}, Test Last Event RMSE:{test_RMSE:.4f}')
print(f' Valid All Event RMSE:{val_all_RMSE:.4f}, Test All Event RMSE:{test_all_RMSE:.4f}')

if params.kernel_type == 1:
    alpha = 'NAN'
else:
    alpha = params.alpha


# type_emb = model.encoder.type_emb.weight * math.sqrt(model.d_model)
# length_scale = model.encoder.kernel.params(type_emb)[0]['length_scale']
length_scale = params.length_scale
results_to_record = [str(params.data), str(params.epoch), str(params.batch_size), str(params.d_model),
                     str(params.d_type), str(params.lr), str(train_loss.item()), str(valid_loss.item()),
                     str(test_loss.item()),
                     str(val_all_RMSE.item()), str(test_all_RMSE.item()), str(length_scale), str(alpha),
                     str(params.normalized), str(params.reg_param), str(params.softmax)]

with open(r'results.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(results_to_record)

model_name = params.data + 'debug_model'
if params.save:
    torch.save(model.state_dict(), 'trained_models/' + model_name + '.pt')

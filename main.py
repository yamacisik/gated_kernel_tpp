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
                'poisson': '../data/simulated/poisson/','exp_hawkes':'../data/simulated/exp_hawkes/','sin_hawkes_2':'../data/simulated/sin_hawkes_2/',
                 '2_d_hawkes': '../data/simulated/2_d_hawkes/', 'mimic2': '../data/mimic/',
                 'stackOverflow': '../data/stackOverflow/', 'retweet': '../data/retweet/'}

DATASET_EVENT_TYPES = {'sin_hawkes': 1, 'power_hawkes': 1,'poisson':1, '2_d_hawkes': 2, 'mimic2': 75, 'stackOverflow': 22,
                       'retweet': 3,'exp_hawkes':1,'sin_hawkes_2':1}

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
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-d_model', type=int, default=32)
parser.add_argument('-d_type', type=int, default=8)
parser.add_argument('-alpha', type=float, default=1.0)
parser.add_argument('-length_scale', type=float, default=1.0)
parser.add_argument('-l', type=float, default=1.0)
parser.add_argument('-s', type=float, default=1.0)
parser.add_argument('-reg_param', type=float, default=0.0)
parser.add_argument('-kernel_type', type=int, default=1)
parser.add_argument('-p_norm', type=float, default=1)
parser.add_argument('-sigma', type=float, default=1)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-l2', type=float, default=0.0000)
parser.add_argument('-seed', type=int, default=42)
parser.add_argument('-save', type=bool, default=True)
parser.add_argument('-normalize', type=int, default=0)
parser.add_argument('-embed_time', type=int, default=0)
parser.add_argument('-timetovec', type=int, default=0)
parser.add_argument('-softmax', type=int, default=0)

params = parser.parse_args()

params.timetovec = True if params.timetovec == 1 else False
params.normalize = True if params.normalize == 1 else False
params.embed_time = True if params.embed_time == 1 else False
params.softmax = True if params.softmax == 1 else False

time_stamp =datetime.now()
date = str(time_stamp.date()).replace('-','')
time = str(time_stamp.time())[:8].replace(':','')
modelname = date +'_' +time

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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
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

t_max = max([seq[-1]['time_since_start'] for data in [train_data, dev_data, test_data] for seq in data])
if not params.normalize:
    'Arrival Times are not normalized...'
    t_max = 1

trainloader = get_dataloader(train_data, params.batch_size, shuffle=True,t_max = t_max)
testloader = get_dataloader(test_data, 1, shuffle=False,t_max = t_max)  # 1 makes it easy to calculate RMSE
valloader = get_dataloader(dev_data, 1, shuffle=False,t_max = t_max)



print(t_max)
model = gated_tpp(num_types, params.d_model, params.d_type,dropout=params.dropout,
                  length_scale=params.length_scale,
                  kernel_type=KERNEL_TYPES[params.kernel_type], alpha=params.alpha, softmax=params.softmax,
                  embed_time=params.embed_time,timetovec=params.timetovec,l = params.l,s = params.s,
                  p = params.p_norm,sigma = params.sigma)





for p in model.encoder.embedding.parameters():
    p.requires_grad = False

# for p in model.encoder.type_emb.parameters():
#         p.requires_grad = False

optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                       params.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=params.l2)

model = model.to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

if params.timetovec:
    stated_dict = torch.load('trained_embeddings/timetovec' +str(params.d_model)+  '.pt')
    model.encoder.embedding.load_state_dict(stated_dict)


log_name = 'training_logs/'+modelname+'_log.txt'
with open(log_name, 'w') as f:
    f.write('Epoch, train_loss, validation_loss,s,length_scale\n')

train_losses = []
validation_losses = []
parameters = []
for epoch in range(params.epoch):
    train_epoch_loss, train_events = model.train_epoch(trainloader, optimizer, params)
    valid_epoch_loss, valid_events, val_RMSE, val_all_RMSE = model.validate_epoch(valloader, device = params.device,reg_param=params.reg_param)
    test_epoch_loss, test_events, test_RMSE, test_all_RMSE = model.validate_epoch(testloader, device = params.device,reg_param=params.reg_param)

    train_loss = train_epoch_loss / train_events
    valid_loss = valid_epoch_loss / valid_events
    test_loss = test_epoch_loss / test_events

    train_losses.append(train_loss)
    validation_losses.append(valid_loss)
    # type_emb = model.encoder.type_emb.weight * math.sqrt(model.d_model)
    # length_scale = model.encoder.kernel.params(type_emb)[0]['length_scale']
    length_scale = F.softplus(model.encoder.kernel.lengthscale).item()
    alpha = 1
    # sigma = F.softplus(model.encoder.kernel.sigma).item()
    sigma = 1
    l =  F.softplus(model.encoder.sigmoid.l).item()
    s = 1
    b = F.softplus(model.encoder.sigmoid.b).item() +1
    # if params.kernel_type == 1:
    #     alpha = 'NAN'
    # else:
    #     alpha = model.encoder.kernel.params(type_emb)[0]['alpha']
        # alpha = params.alpha
    with open(log_name, 'a') as f:
        f.write('{epoch}, {train_loss: 10.8f}, {validation_loss: 10.8f}, {l: 10.8f},{length_scale: 10.8f}\n'
                .format(epoch=epoch, train_loss=train_loss, validation_loss=valid_loss, l=l, length_scale=length_scale))

    print(f'Epoch:{epoch}, Train Loss:{train_loss:.6f}, Valid Loss:{valid_loss:.6f}, Test Loss:{test_loss:.6f}')
    # print(f' Valid Last Event RMSE:{val_RMSE:.4f}, Test Last Event RMSE:{test_RMSE:.4f}')
    # print(f' Valid All Event RMSE:{val_all_RMSE:.4f}, Test All Event RMSE:{test_all_RMSE:.4f}')
    print(f'b:{b}')

    # print(f'Lengthscale:{length_scale},alpha:{alpha},sigma:{sigma}')
    # print(f'l:{l},s:{s}')

    parameters.append([l,length_scale])

min_val_loss_index = np.argmin(validation_losses)
print(f' Min Val Loss:{min(validation_losses)},Best Params: l:{parameters[min_val_loss_index][0]}, lengthscale:{parameters[min_val_loss_index][1]}')
# print(f' Valid Last Event RMSE:{val_RMSE:.4f}, Test Last Event RMSE:{test_RMSE:.4f}')
# print(f' Valid All Event RMSE:{val_all_RMSE:.4f}, Test All Event RMSE:{test_all_RMSE:.4f}')

# if params.kernel_type == 1:
#     alpha = 'NAN'
# else:
#     alpha = params.alpha


# length_scale = params.length_scale
results_to_record = [str(params.data), str(params.epoch), str(params.batch_size), str(params.d_model),
                     str(params.d_type), str(params.lr), str(train_loss.item()), str(valid_loss.item()),
                     str(test_loss.item()),str(val_all_RMSE.item()),
                     str(test_all_RMSE.item()), str(val_RMSE.item()),
                     str(test_RMSE.item()),
                     str(length_scale), str(alpha),str(l),str(s),str(params.softmax),
                     str(params.timetovec),str(params.p_norm)]

# with open(r'results.csv', 'a', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(results_to_record)

with open(r'param_tuning_results.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(results_to_record)

model_name = params.data + 'debug_model'
if params.save:
    torch.save(model.state_dict(), 'trained_models/' + modelname + '.pt')


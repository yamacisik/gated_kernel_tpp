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
import secrets

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
from models.gated_tpp import gated_tpp,count_parameters,get_pairwise_type_embeddings
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
parser.add_argument('-reg_param', type=float, default=5.0)
parser.add_argument('-beta_1', type=float, default=0.4)
parser.add_argument('-beta_2', type=float, default=0.3)
parser.add_argument('-beta_3', type=float, default=1.0)



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
parser.add_argument('-regularize', type=int, default=0)

params = parser.parse_args()

params.timetovec = True if params.timetovec == 1 else False
params.normalize = True if params.normalize == 1 else False
params.embed_time = True if params.embed_time == 1 else False
params.softmax = True if params.softmax == 1 else False
params.regularize = True if params.regularize == 1 else False



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
valid_events = 0
test_events = 0
train_events = 0
for seq in valloader.dataset.event_type:
    valid_events += len(seq)
for seq in trainloader.dataset.event_type:
    train_events += len(seq)
for seq in testloader.dataset.event_type:
    test_events  += len(seq)
print(t_max)
betas = [params.beta_1,params.beta_2,params.beta_3]
model = gated_tpp(num_types, params.d_model, params.d_type,dropout=params.dropout,
                  length_scale=params.length_scale,
                  kernel_type=KERNEL_TYPES[params.kernel_type], alpha=params.alpha, softmax=params.softmax,
                  embed_time=params.embed_time,timetovec=params.timetovec,l = params.l,s = params.s,
                  p = params.p_norm,sigma = params.sigma,regulizing_param= params.reg_param,betas = betas)

optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                       params.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=params.l2)

model = model.to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

if params.timetovec:
    stated_dict = torch.load('trained_embeddings/timetovec' +str(params.d_model)+  '.pt')
    model.encoder.embedding.load_state_dict(stated_dict)
    for p in model.encoder.embedding.parameters():
        p.requires_grad = False




train_losses = []
validation_losses = []
parameters = []

print("the number of trainable parameters: " + str(count_parameters(model)))
start_time = time.time()
for epoch in range(params.epoch):
    train_epoch_loss, _ = model.train_epoch(trainloader, optimizer, params)
    # valid_epoch_loss, _, val_RMSE, val_all_RMSE,val_accuracy = model.validate_epoch(valloader, device = params.device,regularize=params.regularize)
    # test_epoch_loss, _, test_RMSE, test_all_RMSE,test_accuracy = model.validate_epoch(testloader, device = params.device,regularize=params.regularize)

    train_loss = train_epoch_loss / train_events
    # valid_loss = valid_epoch_loss / valid_events
    # test_loss = test_epoch_loss / test_events


    total_time = time.time() - start_time
    average_time = total_time/(epoch+1)

    print(f'Epoch:{epoch}, Train Loss:{train_loss:.6f}, Valid Loss:{train_loss:.6f}, Test Loss:{train_loss:.6f},Time per Epoch :{average_time:.6f}')
    # print(f'Epoch:{epoch}, Train Loss:{train_loss:.6f}, Valid Loss:{train_loss:.6f}, Test Loss:{train_loss:.6f},Time per Epoch :{average_time:.6f}')


valid_epoch_loss, _, val_RMSE, val_all_RMSE,val_accuracy = model.validate_epoch(valloader, device = params.device,regularize=params.regularize)
test_epoch_loss, _, test_RMSE, test_all_RMSE,test_accuracy = model.validate_epoch(testloader, device = params.device,regularize=params.regularize)
valid_loss = valid_epoch_loss / valid_events
test_loss = test_epoch_loss / test_events
print(f' Valid Last Event RMSE:{val_RMSE:.4f}, Test Last Event RMSE:{test_RMSE:.4f},')
print(f' Valid Event Accuracy:{val_accuracy}, Test Event Accuracy:{test_accuracy} \n')


num_types = DATASET_EVENT_TYPES[params.data]
kernel = model.encoder.kernel
if num_types>1:
    events = torch.tensor([range(1,num_types+1)]).to(params.device)
    embeddings = model.encoder.type_emb(events)
    print(embeddings.shape)
    xd_bar, xd = get_pairwise_type_embeddings(embeddings)
    combined_embeddings = torch.cat([xd_bar, xd], dim=-1)

    lengthscales = kernel.lengthscale(combined_embeddings).cpu().detach().numpy().flatten().tolist()
    alphas = kernel.alpha(combined_embeddings).cpu().detach().numpy().flatten().tolist()
    sigmas = kernel.sigma(combined_embeddings).cpu().detach().numpy().flatten().tolist()
    base_intensities = []
    for i in range(1,num_types+1):
        embedding = model.encoder.type_emb(torch.tensor(1).to(device))
        base_intensities.append(kernel.base_intensity(embedding).item())


else:
    lengthscales= [F.softplus(kernel.lengthscale,beta = kernel.betas[0]).item() for i in range(num_types**2)]
    alphas = [F.softplus(kernel.alpha,beta = kernel.betas[1]).item() for i in range(num_types**2)]
    base_intensities = [F.softplus(kernel.base_intensity, beta=kernel.betas[2]).item() for i in range(num_types)]
    sigmas = [torch.sigmoid(kernel.sigma).item() for i in range(num_types**2)]

model_name =secrets.token_hex(5)

params_to_record = lengthscales +sigmas +alphas+base_intensities +kernel.betas

params_to_record = [str(model_name)] +params_to_record
results_to_record = [str(params.data), str(params.epoch), str(params.batch_size), str(params.d_model),
                     str(params.d_type), str(params.lr), str(train_loss), str(valid_loss),
                     str(test_loss),str(val_all_RMSE.item()),
                     str(test_all_RMSE.item()), str(val_RMSE.item()),
                     str(test_RMSE.item()),str(val_accuracy),str(test_accuracy),str(params.softmax),
                     str(params.timetovec),str(model_name)]

with open(r'results.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(results_to_record)

with open(r'learned_params.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(params_to_record)

# model_name = params.data + 'debug_model'
if params.save:
    torch.save(model.state_dict(), 'trained_models/' + model_name + '.pt')


#!/bin/bash


data=2_d_hawkes

python main.py -data mimic -d_model 8 -kernel_type 2  -epoch 1000 -lr 0.001  -batch 32
python main.py -data mimic -d_model 10 -kernel_type 2  -epoch 1000 -lr 0.001  -batch 32
python main.py -data mimic -d_model 12 -kernel_type 2  -epoch 1000 -lr 0.001  -batch 32
python main.py -data mimic -d_model 14 -kernel_type 2  -epoch 1000 -lr 0.001  -batch 32
python main.py -data mimic -d_model 16 -kernel_type 2  -epoch 1000 -lr 0.001  -batch 32





#python main.py -data $data -d_model 12 -kernel_type 2  -epoch 100 -lr 0.001  -batch 32
#python main.py -data stackOverflow -d_model 18 -kernel_type 2  -epoch 250 -lr 0.003  -batch 32
#python main.py -data retweet -d_model 16 -kernel_type 2  -epoch 250 -lr 0.0003  -batch 32



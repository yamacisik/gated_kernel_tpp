#!/bin/bash


data=2_d_hawkes

python main.py -data mimic -d_model 16 -kernel_type 2  -epoch 1000 -lr 0.001  -batch 32
python main.py -data $data -d_model 24 -kernel_type 2  -epoch 100 -lr 0.001  -batch 32
python main.py -data stackoverflow -d_model 16 -kernel_type 2  -epoch 100 -lr 0.001  -batch 32
python main.py -data retweet -d_model 16 -kernel_type 2  -epoch 100 -lr 0.0001  -batch 32



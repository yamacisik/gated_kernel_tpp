#!/bin/bash


data=2_d_hawkes


python main.py -data $data -d_model 16 -epoch 100 -lr 0.0001  -batch 16
python main.py -data $data -d_model 16 -epoch 50 -lr 0.0001  -batch 16

python main.py -data $data -d_model 32 -epoch 100 -lr 0.0001  -batch 16
python main.py -data $data -d_model 32 -epoch 50 -lr 0.0001  -batch 16


#!/bin/bash


data=2_d_hawkes


python main.py -data $data -d_model 16 -epoch 100 -lr 0.0001  -batch 16 -b3 1
python main.py -data $data -d_model 16 -epoch 100 -lr 0.0001  -batch 16 -b3 5
python main.py -data $data -d_model 16 -epoch 100 -lr 0.0001  -batch 16 -b3 10

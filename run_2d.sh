#!/bin/bash


data=2_d_hawkes

python main.py -data $data -d_model 16 -kernel_type 2  -epoch 15 -lr 0.001  -batch 15



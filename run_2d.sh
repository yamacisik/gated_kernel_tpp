#!/bin/bash



python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.0005 -batch 25  -timetovec 0 -softmax 1 -regularize 1
python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.0005 -batch 25  -timetovec 0 -softmax 1


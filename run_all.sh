#!/bin/bash





#python main.py -data mimic -d_model 16 -d_type 16 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 32  -timetovec 0 -softmax 0 -regularize 0

python main.py -data stackOverflow -d_model 16 -d_type 16 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 0 -softmax 0


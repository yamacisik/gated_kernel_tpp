#!/bin/bash



python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 5  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1

python main.py -data sin_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data sin_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data sin_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data sin_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1






#!/bin/bash



python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 0 -softmax 1

python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 1 -softmax 1


python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 1 -softmax 1

python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 1 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 0 -softmax 1
#
python main.py -data 2_d_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 1 -softmax 1

python main.py -data 2_d_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 1 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr .0001 -batch 32  -timetovec 0 -softmax 1

python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 0 -softmax 1

python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 1 -softmax 1


python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 1 -softmax 1

python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 0 -softmax 1
#

python main.py -data 2_d_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 1 -softmax 1
python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 1 -softmax 1

python main.py -data 2_d_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 0 -softmax 1
python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr .0001 -batch 32  -timetovec 0 -softmax 1
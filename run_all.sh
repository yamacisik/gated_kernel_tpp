#!/bin/bash



python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 25 -lr 0.001 -batch 30  -timetovec 0 -softmax 0
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 25 -lr .001 -batch 30  -timetovec 0 -softmax 0
python main.py -data sin_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 25 -lr .001 -batch 30  -timetovec 0 -softmax 0
#
python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 25 -lr 0.001 -batch 30  -timetovec 1 -softmax 0
python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 25 -lr .001 -batch 30  -timetovec 1 -softmax 0
python main.py -data sin_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 25 -lr .001 -batch 30  -timetovec 1 -softmax 0
##
#
#python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data sin_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#
#python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr .0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data sin_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 100 -lr .0001 -batch 20  -timetovec 0 -softmax 1
##
#python main.py -data 2_d_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#
#python main.py -data 2_d_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr 0.0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr .0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 100 -lr .0001 -batch 20  -timetovec 0 -softmax 1
#
#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data sin_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 0 -softmax 1
#
#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data power_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data sin_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#
#
#python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data sin_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#
#python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data power_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data sin_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 0 -softmax 1
##
#
#python main.py -data 2_d_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 1 -softmax 1
#
#python main.py -data 2_d_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr 0.0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data power_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 0 -softmax 1
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 250 -lr .0001 -batch 20  -timetovec 0 -softmax 1
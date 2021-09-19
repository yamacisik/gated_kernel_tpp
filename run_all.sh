#!/bin/bash

#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  5000 -l 0.0 -s 0.0
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.2 -l 1.25 -s 0.55
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.1 -l 0.99 -s 0.433
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.01 -l 0.0 -s 0.0
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.00001 -l 0.0 -s 0.0
#
#
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  5000 -l 0.0 -s 0.0 -timetovec 1
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.2 -l 1.25 -s 0.55 -timetovec 1
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.1 -l 0.99 -s 0.433 -timetovec 1
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.01 -l 0.0 -s 0.0 -timetovec 1
#python main.py -data sin_hawkes -d_model 128 -d_type 128 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.00001 -l 0.0 -s 0.0 -timetovec 1



python main.py -data power_hawkes -d_model 128 -d_type 256 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  5
python main.py -data power_hawkes -d_model 128 -d_type 256 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale 0.01

python main.py -data power_hawkes -d_model 128 -d_type 256 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  5 -timetovec 1
python main.py -data power_hawkes -d_model 128 -d_type 256 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 20 -alpha 1.0 -length_scale  0.01 -timetovec 1
#!/bin/bash


python main.py -data power_hawkes -d_model 22 -kernel_type 2  -epoch 50 -lr 0.0001 -length_scale  1.0 -alpha 0
python main.py -data power_hawkes -d_model 22 -kernel_type 2  -epoch 50 -lr 0.0001 -length_scale  0.15 -alpha 0.55
python main.py -data power_hawkes -d_model 22 -kernel_type 2 -epoch 50 -lr 0.0001 -length_scale  0.0001 -alpha 1

python main.py -data power_hawkes -d_model 22 -kernel_type 2  -epoch 50 -lr 0.0001 -length_scale  1.0 -alpha 0 -batch 1
python main.py -data power_hawkes -d_model 22 -kernel_type 2  -epoch 50 -lr 0.0001 -length_scale  0.15 -alpha 0.55 -batch 1
python main.py -data power_hawkes -d_model 22 -kernel_type 2 -epoch 50 -lr 0.0001 -length_scale  0.0001 -alpha 1 -batch 1


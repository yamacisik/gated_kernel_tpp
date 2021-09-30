#!/bin/bash

for epoch in 100 200
  do
    for data in exp_hawkes power_hawkes
    do
      for lr in 0.001 0.0001
      do
      python main.py -data $data -d_model 128 -d_type 32 -kernel_type 2  -epoch 150 -lr $lr -batch 5 -alpha 0.6  -timetovec 0 -length_scale 0.2  -sigma 0.7
      done
      done
      done

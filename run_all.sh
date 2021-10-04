#!/bin/bash




      for softmax  in 0 1
      do
      for lr  in 0.001
      do
        for data in sin_hawkes_2
        do
      python main.py -data $data -d_model 128 -d_type 32 -kernel_type 2  -epoch 10 -lr $lr -batch 1  -timetovec 0 -softmax $softmax
      done
      done
      done

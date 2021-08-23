#!/bin/bash

for lr in 0.0001 0.0005 0.001
  do
  for d_model in 32 16
  do
  python main.py -data power_hawkes -d_model $d_model -batch 40 -lr $lr -epoch 250
  python main.py -data power_hawkes -d_model $d_model -batch 4 -lr $lr -epoch 50

done
done

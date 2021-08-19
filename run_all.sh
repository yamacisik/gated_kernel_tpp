#!/bin/bash

for lr in 0.0001 0.0005 0.001 0.005
do
for batch in 16  4
  do
  for d_model in 32 16
  do
  python main.py -data power_hawkes -d_model $d_model -batch $batch -lr $lr -epoch 50
  python main.py -data power_hawkes -d_model $d_model -batch $batch -lr $lr -epoch 100
done
done
done
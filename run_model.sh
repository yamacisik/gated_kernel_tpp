#!/bin/bash


data=2_d_hawkes

for b5 in 1 5 10
do
for b3 in 1 5 10
do
python main.py -data $data -d_model 16 -epoch 100 -lr 0.0001  -batch 16 -b5 $b5 -b3 $b3

done
done
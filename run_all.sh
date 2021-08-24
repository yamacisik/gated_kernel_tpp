#!/bin/bash


for d_model in 64 128
do
for param_reg in 1 2 5 10 20
do
for d_type in 4 8
do
python main.py -d_model $d_model -d_type $d_type -batch 40 -lr 0.0001 -epoch 500 -param_reg $param_reg -normalized False
python main.py -d_model $d_model -d_type $d_type -batch 40 -lr 0.0001 -epoch 500 -param_reg $param_reg -normalized True

done
done
done

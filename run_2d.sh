#!/bin/bash

#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.005 -batch 30  -timetovec 0 -softmax 0
#python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 50 -lr 0.005  -batch 30  -timetovec 0 -softmax 0
#python main.py -data 2_d_hawkes -d_model 16 -d_type 16 -kernel_type 2  -epoch 50 -lr 0.005  -batch 30  -timetovec 0 -softmax 0
#
#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.0001  -batch 30  -timetovec 0 -softmax 0
#python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 50 -lr 0.0001  -batch 30  -timetovec 0 -softmax 0
python main.py -data 2_d_hawkes -d_model 16 -d_type 16 -kernel_type 2  -epoch 50 -lr 0.0001  -batch 15  -timetovec 0 -softmax 0


#for beta_1 in  0.6 0.4 0.2
#do
#for beta_2 in  0.6 0.4 0.2
#do
#for beta_3 in 0.6 0.4 0.2
#do
#python main.py -data 2_d_hawkes -d_model 16 -d_type 16 -kernel_type 2  -epoch 40 -lr 0.005  -batch 30  -timetovec 0 -softmax 0 -beta_1 $beta_1  -beta_2 $beta_2  -beta_3 $beta_3
#python main.py -data 2_d_hawkes -d_model 24 -d_type 24 -kernel_type 2  -epoch 40 -lr 0.005  -batch 30  -timetovec 0 -softmax 0 -beta_1 $beta_1  -beta_2 $beta_2  -beta_3 $beta_3
#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 40 -lr 0.005 -batch 30  -timetovec 0 -softmax 0 -beta_1 $beta_1  -beta_2 $beta_2  -beta_3 $beta_3
#
#python main.py -data 2_d_hawkes -d_model 16 -d_type 16 -kernel_type 2  -epoch 40 -lr 0.0001  -batch 30  -timetovec 0 -softmax 0 -beta_1 $beta_1  -beta_2 $beta_2  -beta_3 $beta_3
#python main.py -data 2_d_hawkes -d_model 24 -d_type 24 -kernel_type 2  -epoch 40 -lr 0.0001  -batch 30  -timetovec 0 -softmax 0 -beta_1 $beta_1  -beta_2 $beta_2  -beta_3 $beta_3
#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 40 -lr 0.0001  -batch 30  -timetovec 0 -softmax 0 -beta_1 $beta_1  -beta_2 $beta_2  -beta_3 $beta_3
#
#
#done
#done
#done


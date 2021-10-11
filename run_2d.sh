#!/bin/bash



#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.0005 -batch 30  -timetovec 0 -softmax 1 -regularize 0

#python main.py -data 2_d_hawkes -d_model 16 -d_type 16 -kernel_type 2  -epoch 50 -lr 0.0005 -batch 30  -timetovec 0 -softmax 1 -regularize 0
#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.0005 -batch 30  -timetovec 0 -softmax 1 -regularize 0


for reg_param in 5 4 3 2 1
do
for beta_2 in 0.4 0.37 0.33
do
for l2 in  0.0 0.0001
do
python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.00005 -batch 30  -timetovec 1 -softmax 1 -regularize 0 -reg_param $reg_param -beta_2 $beta_2 -l2 $l2
python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.00005 -batch 30  -timetovec 0 -softmax 1 -regularize 0 -reg_param $reg_param -beta_2 $beta_2 -l2 $l2
done
done
done

for reg_param in 5 4 3 2 1
do
for beta_2 in 0.4 0.37 0.33
do
for l2 in  0.0 0.0001
do
python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 50 -lr 0.00005 -batch 30  -timetovec 1 -softmax 1 -regularize 0 -reg_param $reg_param -beta_2 $beta_2 -l2 $l2
python main.py -data 2_d_hawkes -d_model 64 -d_type 64 -kernel_type 2  -epoch 50 -lr 0.00005 -batch 30  -timetovec 0 -softmax 1 -regularize 0 -reg_param $reg_param -beta_2 $beta_2 -l2 $l2
done
done
done




#python main.py -data 2_d_hawkes -d_model 16 -d_type 16 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 30  -timetovec 0 -softmax 1 -regularize 1
#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 30  -timetovec 0 -softmax 1 -regularize 1
#
#python main.py -data 2_d_hawkes -d_model 16 -d_type 16 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 30  -timetovec 1 -softmax 1 -regularize 1
#python main.py -data 2_d_hawkes -d_model 32 -d_type 32 -kernel_type 2  -epoch 50 -lr 0.0001 -batch 30  -timetovec 1 -softmax 1 -regularize 1
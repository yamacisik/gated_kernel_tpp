#!/bin/bash


data=mimic


python main.py -data $data -d_model 16 -epoch 500 -lr 0.001  -batch 32




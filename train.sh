#!/usr/bin/env bash

python train.py --problem=sin-wav --num_epochs=1000 --save_path=./sin-wav-200 --num_steps=200
python train.py --problem=sin-wav --num_epochs=1000 --save_path=./sin-wav-100 --num_steps=100
python train.py --problem=quadratic-wav --num_epochs=1000 --save_path=./quadratic-wav-100 --num_steps 100
python train.py --problem=quadratic-wav --num_epochs=1000 --save_path=./quadratic-wav-200 --num_steps 200

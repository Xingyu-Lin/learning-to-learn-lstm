#!/usr/bin/env bash
problems=(quadratic)

for i in {0..0}
do
    #python train.py --problem=quadratic-wav --num_epochs=10000 --save_path=./quadratic-wav
    python train.py --problem=sin-wav --num_epochs=10000 --save_path=./sin-wav
done

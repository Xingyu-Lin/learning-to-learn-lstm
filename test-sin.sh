#!/usr/bin/env bash
# Generate problems
# python generate_problems_sin.py

# Optimize
optimizers=("L2L" "Adam" "Momentum" "NAG" "RMSProp" "SGD")
tLen=${#optimizers[@]}

python evaluate.py --optimizer=${optimizers[0]} --problem=sin --path=./sin --problem_path=./problems/sin.npy --learning_rate=0.001 --num_steps=100

for (( i=1; i<${tLen}; i++ ));
do
    python evaluate.py --optimizer=${optimizers[$i]} --problem=sin --path=./sin --problem_path=./problems/sin.npy --learning_rate=0.1 --num_steps=100
done

# Plot
python plot_sin.py

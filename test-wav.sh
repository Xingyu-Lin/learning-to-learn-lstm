#!/usr/bin/env bash
# Generate problems
#python generate_problems.py

# Optimize
optimizers=("L2L" "Adam" "Momentum" "NAG" "RMSProp" "SGD")
tLen=${#optimizers[@]}

python evaluate.py --optimizer=${optimizers[0]} --problem=quadratic-wav --path=./quadratic-wav-200 --problem_path=./problems/quadratic.npz --learning_rate=0.001 --num_steps=200

for (( i=1; i<${tLen}; i++ ));
do
    python evaluate.py --optimizer=${optimizers[$i]} --problem=quadratic-wav --path=./quadratic-wav-200 --problem_path=./problems/quadratic.npz --learning_rate=0.1 --num_steps=200
done

# Plot
python plot.py

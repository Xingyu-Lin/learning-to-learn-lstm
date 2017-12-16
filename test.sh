#!/usr/bin/env bash
# Generate problems
python generate_problems.py

# Optimize
optimizers=( "L2L" "Adam" "Momentum" "NAG" "RMSProp" "SGD")
tLen=${#optimizers[@]}

python evaluate.py --optimizer=${optimizers[0]} --problem=quadratic --path=./quadratic --problem_path=./problems/quadratic.npz --learning_rate=0.001
python evaluate.py --optimizer=${optimizers[0]} --problem=quadratic-wav --path=./quadratic-wav-100 --problem_path=./problems/quadratic.npz --learning_rate=0.001 --num_steps=100

for (( i=1; i<${tLen}; i++ ));
do
    python evaluate.py --optimizer=${optimizers[$i]} --problem=quadratic --path=./quadratic --problem_path=./problems/quadratic.npz --learning_rate=0.1
done

# Plot
python plot.py
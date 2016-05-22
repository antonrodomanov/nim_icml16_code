#!/bin/bash

datasets=( mushrooms a9a cod-rna ijcnn1 )
alphas=( 1 1e-1 1e-3 1e-5 )

for dataset in "${datasets[@]}"
do
    for alpha in "${alphas[@]}"
    do
        echo
        echo "====================================================================================================="
        echo "Run alpha=$alpha on dataset $dataset"

        ./main --dataset $dataset --method NIM --max_epochs 200 --alpha $alpha
    done
done

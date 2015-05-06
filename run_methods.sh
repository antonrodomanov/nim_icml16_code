#!/bin/bash

datasets=( a9a )
methods=( SAG HFN newton BFGS LBFGS SGD )

for dataset in "${datasets[@]}"
do
    echo "*******************************************************************************************************"
    echo "*******************************************************************************************************"
    echo "*******************************************************************************************************"
    echo "Run SO2 on dataset $dataset"

    ./main --dataset $dataset --method SO2 --max_epochs 500

    read epoch elapsed val norm_g < <(tail -n 1 output/"$dataset"_SO2.dat)
    echo "SO2 finished in $elapsed seconds"

    opt_allowed_time=`echo "4*$elapsed" | bc`
    for method in "${methods[@]}"
    do
        echo
        echo "====================================================================================================="
        echo "Run method $method on dataset $dataset for not more than $opt_allowed_time seconds"

        ./main --dataset $dataset --method $method --max_epochs 500000 --opt_allowed_time $opt_allowed_time
    done
done

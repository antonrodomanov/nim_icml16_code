#!/bin/bash

datasets=( mushrooms a9a cod-rna ijcnn1 covtype w8a quantum protein SUSY alpha beta gamma delta )
methods=( SAG HFN newton BFGS LBFGS SGD )

for dataset in "${datasets[@]}"
do
    echo "*******************************************************************************************************"
    echo "*******************************************************************************************************"
    echo "*******************************************************************************************************"
    echo "Run NIM on dataset $dataset"

    ./main --dataset $dataset --method NIM --max_epochs 500

    read epoch elapsed val norm_g < <(tail -n 1 output/"$dataset"_NIM.dat)
    echo "NIM finished in $elapsed seconds"

    opt_allowed_time=`echo "4*$elapsed" | bc`
    for method in "${methods[@]}"
    do
        echo
        echo "====================================================================================================="
        echo "Run method $method on dataset $dataset for not more than $opt_allowed_time seconds"

        ./main --dataset $dataset --method $method --max_epochs 500000 --opt_allowed_time $opt_allowed_time --alpha 1e-2
    done
done

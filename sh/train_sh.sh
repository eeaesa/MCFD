#!/bin/bash
export PATH=/home/gpu/anaconda3/envs/py3.12/bin:$PATH
now=$(date +"%Y%m%d_%H%M%S")


cd /mnt/home/WorkSpace/Survival-Prediction/MCFD
# cd sh
# sh train_sh.sh

echo "begin training!"

epoch=20
lr=2e-4
reg=1e-5
opt=adam

### genomic
fusion=None
mod=omic

methods="SNN"
cancers="BLCA BRCA UCEC GBMLGG LUAD"
for method in $methods; do
    for cancer in $cancers; do
        echo "Running: method=$method with dataset=$cancer"
        python main_sh.py --cancer_style "$cancer" --model_type "$method" --fusion "$fusion" \
                        --mod "$mod" --max_epochs "$epoch" --opt "$opt" --lr "$lr" \
                         --reg "$reg"
    done
done



### multimodal
backbone=resnet50_trunc
fusion=concat

mod=coattn
methods="MCAT MCFD"
cancers="BLCA BRCA UCEC GBMLGG LUAD"
for method in $methods; do
    for cancer in $cancers; do
        echo "Running: method=$method with dataset=$cancer"
        python main_sh.py --cancer_style "$cancer" --model_type "$method" --fusion "$fusion" \
                         --backbone "$backbone" --mod "$mod" --max_epochs "$epoch" --opt "$opt" --lr "$lr" \
                         --reg "$reg"
    done
done


mod=path_and_geno
methods="MCFD"
cancers="BLCA BRCA UCEC GBMLGG LUAD"
for method in $methods; do
    for cancer in $cancers; do
        echo "Running: method=$method with dataset=$cancer"
        python main_sh.py --cancer_style "$cancer" --model_type "$method" --fusion "$fusion" \
                         --backbone "$backbone" --mod "$mod" --max_epochs "$epoch" --opt "$opt" --lr "$lr" \
                         --reg "$reg"
    done
done


echo "\033[32m training finish! \033[0m"

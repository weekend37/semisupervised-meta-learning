#!/bin/bash

DATASET='mini_imagenet'
FOLDER="results/"$DATASET

mkdir -p $FOLDER

for perc in 0.2
do
	file_prefix=sl_maml_gan_$DATASET
	cmd="python3 $file_prefix.py $perc"
	log_file=${FOLDER}/${file_prefix}_$perc.txt
	echo Running 
    echo "  " $ ${cmd}
	echo "and saving results in $log_file"
	$cmd > ${log_file}
done
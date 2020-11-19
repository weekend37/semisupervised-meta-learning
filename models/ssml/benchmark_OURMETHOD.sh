#!/bin/bash

mkdir -p results
for perc in 0.5
do
	echo Running 
    echo "  " $ python3 OURMETHOD.py  $perc
	echo and saving results in results/mini_imagenet/OURMETHOD_omniglot_perc$perc.txt
	python3 OURMETHOD.py $perc > results/mini_imagenet/OURMETHOD_omniglot_perc$perc.txt
done

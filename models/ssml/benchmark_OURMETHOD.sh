#!/bin/bash

mkdir -p results
for perc in 0.05 0.1
do
	echo Running 
    echo "  " $ python3 OURMETHOD.py  $perc
	echo and saving results in results/OURMETHOD_omniglot_perc$perc.txt
	python3 OURMETHOD.py $perc > results/OURMETHOD_omniglot_perc$perc.txt
done

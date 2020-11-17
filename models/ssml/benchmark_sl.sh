#!/bin/bash

mkdir -p results
# for perc in 0.01 0.05 0.07 0.1 0.3 0.4 0.7 0.9
for perc in 0.001 0.01
do
	echo Running 
    echo "  " $ python3 sl_maml_gan_omniglot.py  $perc
	echo and saving results in results/sl_omniglot_perc$perc.txt
	python3 sl_maml_gan_omniglot.py $perc > results/sl_omniglot_perc$perc.txt
done

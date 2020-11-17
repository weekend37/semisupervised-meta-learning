#!/bin/bash

mkdir -p results
for perc in 0.7 0.9
do
	echo Running 
    echo "  " $ python3 ssml_maml_gan_omniglot.py  $perc
	echo and saving results in results/ssml_omniglot_perc$perc.txt
	python3 ssml_maml_gan_omniglot.py $perc > results/ssml_omniglot_perc$perc.txt
done

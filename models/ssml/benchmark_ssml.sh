#!/bin/bash

mkdir -p results
for perc in $(seq 0.0 0.1 1.0);
do
	echo Running 
      	echo "  " $ python3 ssml_maml_gan_omniglot.py  $perc
	echo and saving results in results/ssml_omniglot_perc$perc.txt
	python3 ssml_maml_gan_omniglot.py $perc > results/ssml_omniglot_perc$perc.txt
done

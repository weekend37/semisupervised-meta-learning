#!/bin/bash

FILENAME=ssml_omniglot_benchmark
for perc in $(seq 0.4 0.1 1.0)
do
	python3 ssml_maml_gan_omniglot.py $perc $FILENAME  > benchmark_results/$FILENAME-$perc.txt
done

FILENAME=sl_omniglot_benchmark
for perc in $(seq 0.1 0.1 1.0)
do
	python3 sl_maml_gan_omniglot.py $perc $FILENAME  > benchmark_results/$FILENAME-$perc.txt
done

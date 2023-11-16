#!/bin/bash

cd ../build || exit
make -j 4
printf "Running 'default' ...\n"
./rs_test_1d -i 10
printf "Running 'advanced' ...\n"
./rs_test_1d -i 10 -a
printf "Running 'box' ...\n"
./rs_test_1d -i 10 -b
printf "Running 'pav' ...\n"
./rs_test_1d -i 10 -p

printf "\n"
printf "Running R script ...\n"
cd ../scripts || exit
Rscript measure_1d.r

printf "Finishing\n"

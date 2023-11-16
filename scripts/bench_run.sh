#!/bin/sh
sudo cpupower frequency-set --governor performance > /dev/null 2>&1
../build/benchmark --benchmark_out=bench.data 2> /dev/null
sudo cpupower frequency-set --governor powersave > /dev/null 2>&1

# to get pgfplots data run this on a file containing results from benchmark on selected distribution:
# cat <name> | grep '.*mean.*' | sed -e 's/[a-zA-Z]*\/[a-zA-Z]*\/\([0-9]*\)\/repeats:<Repetitions>_mean\s*\([0-9]*\?\.\?[0-9]*\).*/(\1,\2)/' | tr -d '\n'
# or
# cat <name> | grep '.*mean.*' | sed -e 's/[a-zA-Z]*\/\([0-9]*\)\/repeats:<Repetitions>_mean\s*\([0-9]*\?\.\?[0-9]*\).*/(\1,\2)/' | tr -d '\n'

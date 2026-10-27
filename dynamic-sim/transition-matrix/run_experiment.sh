#!/usr/bin/env bash
# Runs the experimental grid: every bin count n at load (n-1)/n, with 1 and n
# balls per transition, for each seed. Tables go to results/seed<S>/, and each
# run's printed summary to results/seed<S>/<n>bins_<K>balls.txt.
#
#   ./run_experiment.sh          seeds 1 to 5
#   ./run_experiment.sh 1        seed 1 only
set -euo pipefail
cd "$(dirname "$0")"
make -s

if [ $# -gt 0 ]; then seeds="$*"; else seeds="1 2 3 4 5"; fi

for seed in $seeds; do
    out="results/seed$seed"
    mkdir -p "$out"
    for n in 4 5 6 8 10 20 50 100; do
        for k in 1 "$n"; do
            echo "seed $seed: $n bins, load $((n - 1))/$n, $k ball(s) per transition"
            ./build/transition_matrix --bins "$n" --load "$((n - 1))/$n" --balls "$k" \
                --transitions 1000000 --seed "$seed" --out "$out" > "$out/${n}bins_${k}balls.txt"
        done
    done
done

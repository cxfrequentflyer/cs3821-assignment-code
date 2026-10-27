# Idle-bin transition matrix

Builds the transition matrix of the number of idle bins under 1 choice and
under 2 choices, using the simulator in `..`. It writes each matrix as a CSV
table, together with standard errors, a Markov check and how quickly each
chain forgets where it started.

## Run

```bash
make
./build/transition_matrix                                   # asks for the three inputs
./build/transition_matrix --bins 20 --load 19/20 --balls 20
./run_experiment.sh                                         # the full grid, seeds 1 to 5 (about 10 minutes)
./run_experiment.sh 1                                       # seed 1 only (about 2 minutes)
```

| Input | Meaning |
|---|---|
| `--bins N` | number of bins (2 or more), all with efficiency 1 |
| `--load L` | load between 0 and 1, as a decimal (`0.95`) or a fraction (`19/20`): one ball every 1 / (L × N) time units |
| `--balls K` | balls distributed per transition |
| `--transitions T` | transitions to measure (default 500,000) |
| `--seed X` | seed for the hash (default 1) |
| `--out DIR` | where to write the tables (default `results`) |
| `--max-lag M` | longest k for mixing (default 500 turns: 500 × N for one ball per transition, else 500) |
| `--batches B` | batches for the standard errors (default 20) |

Every ball has size 1 and the gaps between balls are constant. The idle count
is taken at the arrival instant of each transition's first ball, before
anything at that instant is processed, so a bin whose last ball finishes
exactly then still counts as busy. K balls are distributed, and then it is
taken again. With constant gaps nothing changes between arrivals, so the
count's mean equals the time average, n(1 − load): exactly 1 at load (n−1)/n.

Use a fraction for loads like 5/6, which no decimal gives exactly. A bin counts
as idle when it emptied more than gap / 10,000 before the counting instant.
Real finishing times are whole grid steps apart for any load with up to three
decimals or a small-denominator fraction, so the threshold only absorbs
rounding in the clock.

## Output

The program prints both matrices, then writes to `DIR`:

```
idle_20bins_load0.95_20balls_1choice.csv
idle_20bins_load0.95_20balls_2choices.csv
idle_20bins_load0.95_20balls_2choices_exact.csv   (only when the exact chain applies)
idle_20bins_load0.95_20balls_mixing.csv
```

Each matrix table has one row per current idle count and one column per next
idle count, followed by:
- the share of transitions that start from that row;
- how many transitions were observed from it. Rows seen only a few hundred
  times are noisy.

Idle counts that never occur are left out, and a row that never occurs is
blank. Both tables of a run cover the same range, so they line up.

The mixing table gives, for each k, how far the rows of the k-step matrix still
are from the share (largest total-variation distance), measured and as a
Markov chain with the measured one-step matrix would have it.
It also gives the average earth mover's distance between adjacent rows (i and
i+1) of each k-step matrix, over pairs of rows that each start at least 1% of
transitions: how much k steps ahead still depends on the starting count.

Under each matrix the program prints:
- the mean idle count and mean wait, with batch-means standard errors;
- a trend check: second-half minus first-half batch means, in standard errors.
  Anything beyond ±3 suggests the run hadn't settled;
- the correlation between consecutive counts. 0 means the next count has
  forgotten the current one;
- a Markov check: how far the two-transition matrix is from the one-transition
  matrix squared;
- mixing: after how many transitions every row is within 0.05 of the share,
  and the same for a Markov chain with this matrix.
- adjacent rows: the average earth mover's distance between adjacent rows
  after 1 transition, and after how many transitions it falls below 0.05.

`run_experiment.sh` runs every n in 4, 5, 6, 8, 10, 20, 50, 100 at load
(n−1)/n, with 1 and n balls per transition, 1,000,000 transitions each. It puts
each seed's tables and printed summaries in `results/seed<S>/`.

## Exact check

With one ball per transition and a coarse time grid, the program also solves
the exact Markov chain on sorted bin backlogs for 2 choices. It then compares
that chain with the measured matrix. For example, 6 bins at load 5/6 gives
130,780 states, agreeing to about 0.005. Above 2 million states or 9 bins the
chain is skipped, and the program prints why.

## Notes

- A warm-up of about 20 settling times of a one-choice queue runs before
  measuring, so high loads take longer. 100 bins at 99/100 with 100 balls per
  transition takes about 45 seconds.

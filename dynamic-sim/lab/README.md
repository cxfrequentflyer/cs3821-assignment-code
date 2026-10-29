# Balls-into-bins lab

An interactive shell for experiments with the simulator in `..`. Set up a
system, run it, and compare placement rules side by side. The results are
averaged over replications with 95% confidence intervals.

## Run

```bash
make
./build/lab                                          # interactive: type help
./build/lab --bins 20 --load 0.95 --sigma 0.5 --run  # set, run once, exit
./build/lab < commands.txt                           # a script of shell commands
```

A session:

```
lab> bins 20; eff 16x1,4x2; load 0.95; sigma 0.5
lab> run
lab> perbin
lab> sweep load 0.8 0.9 0.95
lab> csv results.csv
```

## Settings

| Setting | Meaning | Default |
|---|---|---|
| `bins N` | number of bins | 10 |
| `eff SPEC` | bin efficiencies: a bin with efficiency e processes a size-s ball in s/e time | `1` |
| `load L` | arrival rate as a fraction of total capacity (`0.9` or `19/20`) | 0.9 |
| `rate R` | arrival rate in balls per unit time, instead of load (`0` = use load) | 0 |
| `arrivals TYPE` | `poisson` (exponential gaps) or `constant` | poisson |
| `sigma S` | log-normal ball sizes with mean 1; `0` makes every ball size 1 | 1 |
| `choices LIST` | rules to compare: bins checked per ball, e.g. `1,2` or `1,2,3` | 1,2 |
| `metric M` | what least loaded means: `backlog` (time to clear) or `length` (balls) | backlog |
| `replace on/off` | whether a ball's checked bins may repeat | off |
| `turns T` | measured turns; one turn is one ball per bin, so n balls | 10000 |
| `warmup W` or `auto` | turns run before measuring | auto |
| `reps R` | replications, each with its own seed | 5 |
| `seed X` | base seed | 1 |

Efficiency specs:

| Spec | Bins |
|---|---|
| `1.5` | every bin 1.5 |
| `1,1,2,0.5` | one value per bin (sets the number of bins) |
| `8x1,2x2` | 8 bins at 1 and 2 at 2 (sets the number of bins) |
| `uniform:0.5:1.5` | each bin drawn uniformly from [0.5, 1.5] |
| `lognormal:0.3` | each bin drawn log-normal with mean 1 and sigma 0.3 |

Random efficiencies are drawn once from the seed, so every rule and
replication sees the same bins. Each replication uses its own seed, shared by
the rules, so the rules see identical arrivals and ball sizes.

The auto warm-up is about 20 settling times of a one-choice queue: 20/(1 −
load)² turns, between 100 and 200,000.

## Commands

| Command | Does |
|---|---|
| `show` | print the current settings |
| `set NAME VALUE`, or just `NAME VALUE` | change a setting |
| `run` | run every rule and print the results |
| `sweep NAME V1 V2 ...` | run once per value of one setting, then put it back |
| `perbin` | per-bin breakdown of the last run (grouped above 40 bins) |
| `csv FILE` / `csv off` | append every result to a CSV file |
| `reset`, `help`, `quit` | |

Several commands can share a line, separated by `;`. As flags,
`--NAME VALUE` sets a setting, and `--run`, `--perbin`, `--sweep "NAME V1 V2"`
and `--csv FILE` do the same as the commands. With `--run` or `--sweep` the
program exits afterwards.

## Results

For each rule, the mean over replications (± 95% confidence interval) of:

| Column | Meaning |
|---|---|
| mean wait | time a ball queues before its processing starts, over balls arriving after the warm-up |
| waited | fraction of balls that had to wait at all |
| median, p95, p99 | wait percentiles, accurate to about 4% |
| avg max queue | the longest queue in each turn, averaged over turns |
| global max | the longest queue in any run |
| load | measured work received / total capacity, to check the setting |

Queue length is the number of balls in a bin, including the one being
processed. A queue only grows when a ball joins it, so the longest queue is
tracked exactly at every placement.

The program also prints:
- **A theory check.** For 1 choice with Poisson arrivals, the M/G/1 mean wait
  that the simulation should match.
- **An overload warning.** It appears when a rule sends some bins more work
  than they can process: offered load above 100%. Their queues then grow
  without limit, so the averages depend on run length. With 1 choice every bin
  gets 1/n of the balls, so slow bins overload first.

`perbin` shows, for each bin (or group of similar bins):
- its share of the balls;
- its offered load: work received per unit of its capacity;
- its mean wait;
- its longest queue.

The CSV has one row per rule per run (or per sweep value), with every setting
and result.

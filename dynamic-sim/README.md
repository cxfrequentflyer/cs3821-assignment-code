# dynamic-sim: balls into bins over time

A discrete-event simulator. A single source emits balls. Each ball is hashed
into `D` of the active bins and joins the least loaded one. Each bin is a FIFO
queue that processes balls at its own efficiency. Bins can be added and retired
while the simulation runs. The output is the mean queueing delay: the time a
ball waits before processing starts, not counting the processing itself.

## Build and run

```bash
make            # builds build/dynsim and build/tests (C++17, header-only)
make test       # exact runs + queueing-theory checks, ~3 s
./build/dynsim --bins 8 --d 2 --sigma 1 --load 0.9 --time 100000
./build/dynsim --bins 8 --d 2 --load 0.6 --add 2 --add-efficiency 2 --retire 4 --time 300000
```

`--load` sets the arrival rate so that the starting bins are that busy on average.
The other flags are `--constant` (constant gaps instead of Poisson arrivals),
`--metric backlog|length`, `--replace` (sample the `D` bins with replacement)
and `--seed`.

## Using it

```cpp
auto source = make_shared<Source>(/*meanGap=*/0.125, /*poisson=*/true, /*sigma=*/1.0, /*seed=*/1);
Simulator sim(source, /*Dchoice=*/2, /*seed=*/1);   // LoadMetric::Backlog, distinct choices
for (int i = 0; i < 8; i++) sim.addBin(1.0);
sim.runUntil(1000);
int id = sim.addBin(2.0);       // takes effect at sim.now()
sim.retireBin(0);               // no new balls; it drains, then is freed
sim.runUntil(2000);
sim.averageQueueingDelay();     // also ballsProcessed(), ballsInSystem(), totalQueueingDelay()
```

## Files

| File | Role |
|---|---|
| `Source.hpp` | emits balls; gaps constant or exponential, sizes log-normal |
| `Bin.hpp` | FIFO queue; a ball of size `s` takes `s / efficiency` time |
| `Simulator.hpp` | task priority queue, hashing with D choices, add/retire bins, stats |
| `EventEntity.hpp` | the task interface shared by the source and bins |
| `Ball.hpp`, `Distribution.hpp`, `Hash.hpp` | data, seeded random streams, a 64-bit mixer |
| `tests.cpp` | checks against hand-computed runs and Pollaczek-Khinchine |
| `lab/` | interactive experiment shell: set bins, efficiencies, arrivals, ball sizes and rules, then run, sweep and compare |
| `transition-matrix/` | idle-bin transition matrix tables for 1 vs 2 choices, for any bins, load and balls per transition |
| `visualiser/index.html` | live browser view: add/retire bins, change the rate, toggle 1 or 2 choices |
| `original/` | the incomplete version this was built from |

## Visualiser

Open `visualiser/index.html` in a browser. It needs no build step. It runs the
same rules as `Simulator.hpp`, ported to JavaScript so it can step in real time.
Bins are drawn as graduated tubes, and each ball is a rectangle whose height is
its size. The page shows each bin's mean wait, and a chart tracks the mean wait
since the last reset and over the last 200 balls, against the one-choice M/G/1
prediction. The JavaScript model was checked against the same theory values as
`tests.cpp`.

## Modeling choices

- **Ball sizes have mean 1.** They are log-normal with `mu = -sigma^2 / 2`, so a
  bin at efficiency 1 processes an average ball in one unit of time, and the
  load does not change with `sigma`. `sigma = 0` gives every ball size 1.
- **Load** means either `Backlog` (default) or `QueueLength`. `Backlog` is the
  time a bin needs to clear everything it holds, which is the wait a new ball
  would see. `QueueLength` is the number of balls in the bin, counting the one
  being processed. It ignores both ball sizes and efficiency.
- **The hash** mixes the ball id, its size, the current time and a seed. The
  `k`-th choice is a rehash of that value. By default the `D` bins are all
  different (when `D` exceeds the number of bins, every bin is checked). If two
  bins tie, the ball joins the one checked first, which is a uniformly random
  choice among the tied bins.
- **Adding or retiring a bin** changes which bins the hash can map to. There is
  no consistent hashing.
- **Ties in time:** at the same instant, bins finish balls before the source
  emits a new one.
- **Delay is recorded when a ball finishes.** Balls still in a bin at the end
  are not counted. No warm-up period is removed.
- **A ball that arrives when no bins are active** throws an error.

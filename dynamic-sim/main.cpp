// Example run: balls arrive at a group of bins, and bins can be added and
// retired part way through. Prints the mean queueing delay of each phase.
//
//   ./build/dynsim --bins 8 --d 2 --sigma 1 --load 0.9 --time 100000
//   ./build/dynsim --bins 8 --d 2 --add 2 --retire 4 --time 300000
//
// --load sets the arrival rate so the starting bins are that busy on average
// (mean ball size is 1). With --add / --retire, the run has three equal
// phases: --add new bins join at time/3, and the first --retire of the
// starting bins are retired at 2*time/3.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "Simulator.hpp"

using namespace std;

static void usage() {
    fprintf(stderr,
            "usage: dynsim [--bins N] [--d D] [--sigma S] [--load RHO] [--constant]\n"
            "              [--metric backlog|length] [--replace] [--time T] [--seed X]\n"
            "              [--add K] [--add-efficiency E] [--retire K]\n");
    exit(2);
}

int main(int argc, char **argv) {
    int bins = 8, d = 2, add = 0, retire = 0;
    double sigma = 1.0, load = 0.9, endTime = 1e5, addEfficiency = 1.0;
    bool poisson = true, distinct = true;
    LoadMetric metric = LoadMetric::Backlog;
    uint64_t seed = 1;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        auto value = [&]() -> const char * {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (arg == "--bins") bins = atoi(value());
        else if (arg == "--d") d = atoi(value());
        else if (arg == "--sigma") sigma = atof(value());
        else if (arg == "--load") load = atof(value());
        else if (arg == "--constant") poisson = false;
        else if (arg == "--replace") distinct = false;
        else if (arg == "--time") endTime = atof(value());
        else if (arg == "--seed") seed = strtoull(value(), nullptr, 10);
        else if (arg == "--add") add = atoi(value());
        else if (arg == "--add-efficiency") addEfficiency = atof(value());
        else if (arg == "--retire") retire = atoi(value());
        else if (arg == "--metric") {
            string m = value();
            if (m == "backlog") metric = LoadMetric::Backlog;
            else if (m == "length") metric = LoadMetric::QueueLength;
            else usage();
        } else usage();
    }
    if (bins < 1 || load <= 0.0 || retire > bins) usage();

    double meanGap = 1.0 / (load * bins);
    Simulator sim(make_shared<Source>(meanGap, poisson, sigma, seed), d, seed, metric, distinct);
    vector<int> ids;
    for (int i = 0; i < bins; i++) ids.push_back(sim.addBin(1.0));

    printf("%d bins, D = %d (%s, by %s), sigma = %g, %s arrivals, load %g\n\n",
           bins, d, distinct ? "distinct" : "with replacement",
           metric == LoadMetric::Backlog ? "backlog" : "queue length",
           sigma, poisson ? "Poisson" : "constant", load);
    printf("%-26s %8s %8s %12s %10s\n", "phase", "active", "draining", "processed", "mean wait");

    uint64_t lastProcessed = 0;
    double lastWait = 0.0;
    auto report = [&](const char *phase) {
        uint64_t processed = sim.ballsProcessed() - lastProcessed;
        double wait = sim.totalQueueingDelay() - lastWait;
        printf("%-26s %8zu %8zu %12llu %10.4f\n", phase, sim.activeBins().size(),
               sim.retiringBins().size(), (unsigned long long)processed,
               processed ? wait / processed : 0.0);
        lastProcessed = sim.ballsProcessed();
        lastWait = sim.totalQueueingDelay();
    };

    if (add == 0 && retire == 0) {
        sim.runUntil(endTime);
        report("whole run");
    } else {
        sim.runUntil(endTime / 3);
        report("1: starting bins");
        for (int i = 0; i < add; i++) sim.addBin(addEfficiency);
        sim.runUntil(2 * endTime / 3);
        report("2: after adding");
        for (int i = 0; i < retire; i++) sim.retireBin(ids[i]);
        sim.runUntil(endTime);
        report("3: after retiring");
    }
    printf("\noverall mean queueing delay %.4f over %llu balls (%llu still in bins)\n",
           sim.averageQueueingDelay(), (unsigned long long)sim.ballsProcessed(),
           (unsigned long long)sim.ballsInSystem());
    return 0;
}

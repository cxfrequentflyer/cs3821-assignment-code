// Checks the simulator against hand-computed runs and queueing theory.
// Build and run with `make test`.

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include "Simulator.hpp"

using namespace std;

static int failures = 0;

static void check(bool ok, const string &what) {
    printf("%s  %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) failures++;
}

static void checkNear(double got, double want, double relTol, const string &what) {
    char buf[256];
    snprintf(buf, sizeof buf, "%s: got %.4f, want %.4f (+-%.0f%%)",
             what.c_str(), got, want, relTol * 100);
    check(fabs(got - want) <= relTol * fabs(want), buf);
}

// Pollaczek-Khinchine: mean queueing delay of an M/G/1 queue with arrival rate
// lambda and service time S / efficiency, where S is log-normal with mean 1,
// so E[S^2] = exp(sigma^2).
static double mg1Wait(double lambda, double sigma, double efficiency = 1.0) {
    double rho = lambda / efficiency;
    double meanServiceSq = exp(sigma * sigma) / (efficiency * efficiency);
    return lambda * meanServiceSq / (2.0 * (1.0 - rho));
}

static Simulator makeSim(double meanGap, bool poisson, double sigma, int d,
                         uint64_t seed, LoadMetric metric = LoadMetric::Backlog,
                         bool distinct = true) {
    return Simulator(make_shared<Source>(meanGap, poisson, sigma, seed), d, seed,
                     metric, distinct);
}

static void deterministicNoWait() {
    // gaps of 1.25, every ball size 1: the bin is always free when a ball arrives
    Simulator sim = makeSim(1.25, false, 0.0, 1, 1);
    sim.addBin(1.0);
    sim.runUntil(1000.0);
    check(sim.averageQueueingDelay() == 0.0, "D/D/1 under capacity: nobody waits");
    check(sim.ballsProcessed() == 800, "D/D/1: 801 balls arrive in [0, 1000], 800 finish by 1000");
}

static void deterministicOverload() {
    // a ball every 0.5, each takes 1: ball k starts at k and waits 0.5k, and
    // finishes at k + 1. By time 10 balls 0..9 are done: mean wait 0.5 * 4.5.
    Simulator sim = makeSim(0.5, false, 0.0, 1, 1);
    sim.addBin(1.0);
    sim.runUntil(10.0);
    check(sim.ballsProcessed() == 10, "overloaded D/D/1: 10 balls processed by t = 10");
    checkNear(sim.averageQueueingDelay(), 2.25, 1e-12, "overloaded D/D/1 mean wait");
    check(sim.ballsGenerated() == sim.ballsProcessed() + sim.ballsInSystem(),
          "conservation: generated = processed + in system");
}

static void efficiencyScalesService() {
    // efficiency 2 halves processing time; constant gaps of 0.4 vs service 0.5
    Simulator sim = makeSim(0.4, false, 0.0, 1, 1);
    sim.addBin(2.0);
    sim.runUntil(4.0);
    // ball k arrives 0.4k, starts 0.5k, waits 0.1k, finishes 0.5k + 0.5;
    // done by t = 4: k = 0..7, mean wait 0.1 * 3.5
    checkNear(sim.averageQueueingDelay(), 0.35, 1e-9, "efficiency 2 halves service time");
}

static void pollaczekKhinchine() {
    {
        Simulator sim = makeSim(1.0 / 0.8, true, 0.0, 1, 7);
        sim.addBin(1.0);
        sim.runUntil(2e6);
        checkNear(sim.averageQueueingDelay(), mg1Wait(0.8, 0.0), 0.04, "M/D/1 at load 0.8");
    }
    {
        Simulator sim = makeSim(1.0 / 0.5, true, 1.0, 1, 8);
        sim.addBin(1.0);
        sim.runUntil(4e6);
        checkNear(sim.averageQueueingDelay(), mg1Wait(0.5, 1.0), 0.04,
                  "M/G/1, log-normal sigma 1, load 0.5");
    }
    {
        Simulator sim = makeSim(1.0 / 1.6, true, 0.5, 1, 9);
        sim.addBin(2.0);
        sim.runUntil(2e6);
        checkNear(sim.averageQueueingDelay(), mg1Wait(1.6, 0.5, 2.0), 0.04,
                  "M/G/1, efficiency 2, load 0.8");
    }
}

static void oneChoiceSplitsPoisson() {
    // D = 1: hashing splits a Poisson stream into independent Poisson streams,
    // so each of the 4 bins is an M/D/1 queue with a quarter of the traffic
    Simulator sim = makeSim(1.0 / 3.2, true, 0.0, 1, 11);
    for (int i = 0; i < 4; i++) sim.addBin(1.0);
    sim.runUntil(1e6);
    checkNear(sim.averageQueueingDelay(), mg1Wait(0.8, 0.0), 0.04,
              "D = 1 over 4 bins behaves like 4 separate M/D/1 queues");
    bool even = true;
    for (const auto &bin : sim.activeBins()) {
        double share = static_cast<double>(bin->ballsProcessed()) / sim.ballsProcessed();
        even = even && fabs(share - 0.25) < 0.005;
    }
    check(even, "D = 1: each bin gets a quarter of the balls (+-0.5%)");
}

static void moreChoicesHelp() {
    double wait[3];
    for (int d = 1; d <= 3; d++) {
        Simulator sim = makeSim(1.0 / (0.9 * 16), true, 1.0, d, 12);
        for (int i = 0; i < 16; i++) sim.addBin(1.0);
        sim.runUntil(1e5);
        wait[d - 1] = sim.averageQueueingDelay();
    }
    char buf[128];
    snprintf(buf, sizeof buf, "16 bins at load 0.9: wait %.3f (D=1) > %.3f (D=2) > %.3f (D=3)",
             wait[0], wait[1], wait[2]);
    check(wait[0] > 3 * wait[1] && wait[1] > wait[2], buf);

    Simulator byCount = makeSim(1.0 / (0.9 * 16), true, 1.0, 2, 12, LoadMetric::QueueLength);
    for (int i = 0; i < 16; i++) byCount.addBin(1.0);
    byCount.runUntil(1e5);
    check(byCount.averageQueueingDelay() < wait[0], "D = 2 by queue length also beats D = 1");
}

static void distinctVersusReplacement() {
    // 2 bins, a ball every 0.6, each takes 1. Checking both bins, a ball always
    // finds one free. With replacement both checks sometimes hit the busy bin.
    Simulator distinct = makeSim(0.6, false, 0.0, 2, 13, LoadMetric::Backlog, true);
    distinct.addBin(1.0);
    distinct.addBin(1.0);
    distinct.runUntil(1e4);
    check(distinct.averageQueueingDelay() == 0.0, "distinct D = n: a free bin is always found");

    Simulator repeat = makeSim(0.6, false, 0.0, 2, 13, LoadMetric::Backlog, false);
    repeat.addBin(1.0);
    repeat.addBin(1.0);
    repeat.runUntil(1e4);
    check(repeat.averageQueueingDelay() > 0.0, "with replacement D = 2 sometimes misses the free bin");
}

static void retiredBinDrainsAndIsFreed() {
    Simulator sim = makeSim(1.0 / 1.8, true, 1.0, 2, 14);
    int first = sim.addBin(1.0);
    sim.addBin(1.0);
    sim.runUntil(1000.0);

    weak_ptr<Bin> retired = sim.activeBins()[0];
    size_t before = retired.lock()->queueLength();
    sim.retireBin(first);
    check(sim.activeBins().size() == 1 && sim.retiringBins().size() == (before > 0 ? 1u : 0u),
          "retired bin leaves the active set and drains separately");

    // its queue only ever shrinks, until it is gone
    bool shrinking = true;
    size_t last = before;
    while (!retired.expired()) {
        sim.runUntil(sim.now() + 0.01);
        if (auto bin = retired.lock()) {
            shrinking = shrinking && bin->queueLength() <= last;
            last = bin->queueLength();
        }
        if (sim.now() > 1e4) break;
    }
    check(shrinking, "retired bin takes no new balls");
    check(retired.expired() && sim.retiringBins().empty(), "retired bin is freed once empty");
    check(sim.ballsGenerated() == sim.ballsProcessed() + sim.ballsInSystem(),
          "conservation still holds after a bin is freed");

    // an idle bin is freed the moment it is retired
    Simulator idle = makeSim(10.0, false, 0.0, 1, 15);
    idle.addBin(1.0);
    int spare = idle.addBin(1.0);
    weak_ptr<Bin> spareBin = idle.activeBins()[1];
    idle.retireBin(spare);
    check(spareBin.expired(), "an idle bin is freed as soon as it is retired");
}

static void addedBinGetsBalls() {
    Simulator sim = makeSim(1.0 / 0.9, true, 1.0, 2, 16);
    sim.addBin(1.0);
    sim.runUntil(1000.0);
    sim.addBin(1.0);
    sim.runUntil(2000.0);
    check(sim.activeBins()[1]->ballsProcessed() > 300, "a bin added mid-run starts taking balls");
}

static void noBinsThrows() {
    Simulator sim = makeSim(1.0, false, 0.0, 1, 17);
    bool threw = false;
    try {
        sim.runUntil(1.0);
    } catch (const runtime_error &) {
        threw = true;
    }
    check(threw, "a ball with no active bins is an error");
}

static void deterministic() {
    auto run = [](uint64_t seed) {
        Simulator sim = makeSim(1.0 / 7.0, true, 1.5, 2, seed);
        for (int i = 0; i < 8; i++) sim.addBin(1.0);
        sim.runUntil(1e4);
        return sim.averageQueueingDelay();
    };
    check(run(5) == run(5), "same seed, same result");
    check(run(5) != run(6), "different seed, different result");
}

int main() {
    deterministicNoWait();
    deterministicOverload();
    efficiencyScalesService();
    pollaczekKhinchine();
    oneChoiceSplitsPoisson();
    moreChoicesHelp();
    distinctVersusReplacement();
    retiredBinDrainsAndIsFreed();
    addedBinGetsBalls();
    noBinsThrows();
    deterministic();
    printf("\n%s (%d failed)\n", failures ? "FAILED" : "all passed", failures);
    return failures ? 1 : 0;
}

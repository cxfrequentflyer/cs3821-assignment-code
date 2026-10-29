// An interactive lab for balls-into-bins experiments on ../Simulator.hpp.
//
// Set up a system (how many bins and how efficient each is, how balls arrive,
// how big they are, which placement rules to compare), run it for a number of
// turns, and read off waiting times and queue lengths, averaged over
// replications with 95% confidence intervals. One turn is n balls: one per bin
// on average.
//
//   ./build/lab                                   interactive; type `help`
//   ./build/lab --bins 20 --load 0.9 --run        set, run once, exit
//   ./build/lab < commands.txt                    a script of the same commands

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include "Simulator.hpp"

using namespace std;

// ---------- settings ----------

struct Settings {
    int bins = 10;
    string eff = "1";           // efficiency spec, see parseEfficiencies
    double load = 0.9;          // used when rate is 0
    double rate = 0.0;          // balls per unit time; 0 means load x capacity
    bool poisson = true;
    double sigma = 1.0;         // log-normal ball sizes with mean 1
    vector<int> choices = {1, 2};
    LoadMetric metric = LoadMetric::Backlog;
    bool distinct = true;
    long long turns = 10000;
    long long warmup = -1;      // turns; -1 means auto
    int reps = 5;
    uint64_t seed = 1;
};

static vector<string> split(const string &s, char sep) {
    vector<string> out;
    string cur;
    for (char c : s) {
        if (c == sep) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

static vector<string> words(const string &line) {
    istringstream in(line);
    vector<string> out;
    for (string w; in >> w;) out.push_back(w);
    return out;
}

static bool toInt(const string &s, long long &v) {
    if (s.empty()) return false;
    char *end;
    v = strtoll(s.c_str(), &end, 10);
    return *end == '\0';
}

// A number as a decimal (0.95) or a fraction (19/20).
static bool toNumber(const string &s, double &v) {
    vector<string> parts = split(s, '/');
    if (parts.size() > 2 || parts[0].empty()) return false;
    char *end;
    v = strtod(parts[0].c_str(), &end);
    if (*end != '\0') return false;
    if (parts.size() == 2) {
        double den = strtod(parts[1].c_str(), &end);
        if (parts[1].empty() || *end != '\0' || den == 0.0) return false;
        v /= den;
    }
    return isfinite(v);
}

// Bin efficiencies from a spec, or false with an error message:
//   1.5               every bin 1.5
//   1,1,2,0.5         one value per bin (sets the number of bins)
//   8x1,2x2           8 bins at 1 and 2 bins at 2 (sets the number of bins)
//   uniform:0.5:1.5   each bin drawn uniformly from [0.5, 1.5]
//   lognormal:0.3     each bin log-normal with mean 1 and sigma 0.3
// Random draws use the seed, so every rule and replication sees the same bins.
static bool parseEfficiencies(const string &spec, int bins, uint64_t seed, vector<double> &out, string &error) {
    out.clear();
    if (spec.find(':') != string::npos) {
        vector<string> p = split(spec, ':');
        mt19937_64 gen(hashCombine(seed, 4));
        double a, b;
        if (p[0] == "uniform" && p.size() == 3 && toNumber(p[1], a) && toNumber(p[2], b) && a > 0 && b >= a) {
            uniform_real_distribution<double> u(a, b);
            for (int i = 0; i < bins; i++) out.push_back(u(gen));
            return true;
        }
        if (p[0] == "lognormal" && p.size() == 2 && toNumber(p[1], a) && a >= 0) {
            lognormal_distribution<double> ln(-0.5 * a * a, a > 0 ? a : 1.0);
            for (int i = 0; i < bins; i++) out.push_back(a > 0 ? ln(gen) : 1.0);
            return true;
        }
        error = "use uniform:LOW:HIGH (LOW above 0) or lognormal:SIGMA (SIGMA 0 or more)";
        return false;
    }
    bool list = spec.find(',') != string::npos || spec.find('x') != string::npos;
    for (const string &item : split(spec, ',')) {
        long long count = 1;
        string value = item;
        size_t x = item.find('x');
        if (x != string::npos) {
            if (!toInt(item.substr(0, x), count) || count < 1) { error = "in COUNTxVALUE, COUNT must be a whole number, 1 or more"; return false; }
            value = item.substr(x + 1);
        }
        double v;
        if (!toNumber(value, v) || v <= 0) { error = "every efficiency must be a number above 0, such as 1 or 0.5"; return false; }
        for (long long i = 0; i < count; i++) out.push_back(v);
    }
    if (!list) out.assign(bins, out[0]);
    if (out.empty()) { error = "no bins"; return false; }
    return true;
}

static bool efficiencySetsBins(const string &spec) {
    return spec.find(':') == string::npos && (spec.find(',') != string::npos || spec.find('x') != string::npos);
}

// Turns of warm-up before measuring: about 20 settling times of a one-choice
// queue, 1 / (1 - load)^2 service times, at roughly one service time a turn.
static long long warmupTurns(const Settings &s, double load) {
    if (s.warmup >= 0) return s.warmup;
    if (load >= 1.0) return 100;
    return static_cast<long long>(min(200000.0, max(100.0, 20.0 / ((1.0 - load) * (1.0 - load)))));
}

// ---------- statistics ----------

// Waits on a log scale, 16 buckets per doubling from about 1e-6 to 1e12, plus
// an exact count of zero waits. Quantiles come out within about 4%.
class WaitHistogram {
    static constexpr int PER = 16, LOW = -20 * PER, HIGH = 40 * PER;
    vector<uint64_t> counts = vector<uint64_t>(HIGH - LOW + 1, 0);
    uint64_t zeros = 0, total = 0;

public:
    void add(double w) {
        total++;
        if (w <= 1e-9) { zeros++; return; }
        int k = static_cast<int>(floor(PER * log2(w)));
        counts[min(max(k, LOW), HIGH) - LOW]++;
    }
    double quantile(double q) const {
        if (!total) return nan("");
        uint64_t target = static_cast<uint64_t>(ceil(q * total)), seen = zeros;
        if (target <= zeros) return 0.0;
        for (size_t i = 0; i < counts.size(); i++) {
            seen += counts[i];
            if (seen >= target) return exp2((static_cast<int>(i) + LOW + 0.5) / PER);
        }
        return INFINITY;
    }
    double waitedFraction() const { return total ? 1.0 - static_cast<double>(zeros) / total : nan(""); }
};

struct Summary {
    double mean = nan(""), half = nan("");   // half-width of a 95% confidence interval
};

static Summary summarize(const vector<double> &x) {
    static const double t[] = {0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
                               2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
                               2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
    Summary s;
    size_t n = x.size();
    if (!n) return s;
    s.mean = 0.0;
    for (double v : x) s.mean += v;
    s.mean /= n;
    if (n < 2) return s;
    double var = 0.0;
    for (double v : x) var += (v - s.mean) * (v - s.mean);
    var /= (n - 1);
    s.half = (n - 1 <= 30 ? t[n - 1] : 1.96) * sqrt(var / n);
    return s;
}

// ---------- one run ----------

struct BinStats {
    double eff = 1.0;
    uint64_t balls = 0;
    double waitSum = 0.0, busy = 0.0;   // busy: processing time of the balls it received
    int maxQueue = 0;
};

struct RunResult {
    double meanWait = 0.0, avgMaxQueue = 0.0, maxWait = 0.0, duration = 0.0, measuredLoad = 0.0;
    int globalMaxQueue = 0;
    uint64_t balls = 0;
    WaitHistogram hist;
    vector<BinStats> perBin;
};

// Queue length is the number of balls in a bin, including the one being
// processed. A queue only grows when a ball joins it, so the longest queue in
// a turn is the longest at the turn's start or right after one of its balls.
static RunResult runOnce(const Settings &s, const vector<double> &effs, int d, uint64_t seed, long long warmup) {
    const int n = static_cast<int>(effs.size());
    double capacity = 0.0;
    for (double e : effs) capacity += e;
    const double lambda = s.rate > 0 ? s.rate : s.load * capacity;

    auto source = make_shared<Source>(1.0 / lambda, s.poisson, s.sigma, seed);
    Simulator sim(source, d, seed, s.metric, s.distinct);
    for (double e : effs) sim.addBin(e);

    RunResult r;
    r.perBin.resize(n);
    for (int i = 0; i < n; i++) r.perBin[i].eff = effs[i];
    bool measuring = false;
    int turnMax = 0;
    double waitSum = 0.0, work = 0.0;
    sim.onPlace = [&](const Bin &bin, const Ball &ball) {
        int q = static_cast<int>(bin.queueLength());
        if (q > turnMax) turnMax = q;
        if (!measuring) return;
        double w = ball.queueingDelay();
        r.hist.add(w);
        waitSum += w;
        r.balls++;
        r.maxWait = max(r.maxWait, w);
        work += ball.size;
        BinStats &b = r.perBin[bin.id];
        b.balls++;
        b.waitSum += w;
        b.busy += ball.size / bin.getEfficiency();
        b.maxQueue = max(b.maxQueue, q);
    };

    double start = 0.0, sumTurnMax = 0.0;
    for (long long t = 0; t < warmup + s.turns; t++) {
        if (t == warmup) { measuring = true; start = sim.now(); }
        turnMax = 0;
        for (const auto &bin : sim.activeBins()) turnMax = max(turnMax, static_cast<int>(bin->queueLength()));
        for (int b = 0; b < n; b++) sim.runUntil(source->nextActionTime());   // place the next ball
        if (measuring) {
            sumTurnMax += turnMax;
            r.globalMaxQueue = max(r.globalMaxQueue, turnMax);
        }
    }
    r.duration = sim.now() - start;
    r.meanWait = r.balls ? waitSum / r.balls : nan("");
    r.avgMaxQueue = sumTurnMax / s.turns;
    r.measuredLoad = r.duration > 0 ? work / (r.duration * capacity) : nan("");
    return r;
}

// ---------- experiments ----------

struct PolicyResult {
    int d = 1;
    vector<RunResult> runs;
};

struct Experiment {
    Settings s;
    vector<double> effs;
    double lambda = 0.0, load = 0.0;
    long long warmup = 0;
    vector<PolicyResult> policies;
    double seconds = 0.0;
};

static bool prepare(const Settings &s, Experiment &e, string &error) {
    e.s = s;
    if (!parseEfficiencies(s.eff, s.bins, s.seed, e.effs, error)) return false;
    double capacity = 0.0;
    for (double x : e.effs) capacity += x;
    e.lambda = s.rate > 0 ? s.rate : s.load * capacity;
    e.load = e.lambda / capacity;
    e.warmup = warmupTurns(s, e.load);
    return true;
}

static Experiment runExperiment(const Settings &s, bool progress) {
    Experiment e;
    string error;
    prepare(s, e, error);
    auto t0 = chrono::steady_clock::now();
    for (int d : s.choices) {
        PolicyResult p;
        p.d = d;
        for (int rep = 0; rep < s.reps; rep++) {
            if (progress && isatty(2)) fprintf(stderr, "\r  running %d choice%s, replication %d of %d ", d, d == 1 ? "" : "s", rep + 1, s.reps);
            p.runs.push_back(runOnce(s, e.effs, d, hashCombine(s.seed, static_cast<uint64_t>(rep)), e.warmup));
        }
        e.policies.push_back(p);
    }
    if (progress && isatty(2)) fprintf(stderr, "\r%60s\r", "");
    e.seconds = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    return e;
}

// ---------- output ----------

static string sig(double v, int digits = 4) {
    if (isnan(v)) return "-";
    if (isinf(v)) return "inf";
    char b[32];
    snprintf(b, sizeof b, "%.*g", digits, v);
    return b;
}

static string pm(const Summary &x) {
    if (isnan(x.half)) return sig(x.mean);
    return sig(x.mean) + " ± " + sig(x.half, 2);
}

// Pad to a width in characters, counting a multi-byte character such as ± once.
static string pad(const string &s, size_t width) {
    size_t chars = 0;
    for (unsigned char c : s) chars += (c & 0xC0) != 0x80;
    return chars >= width ? s + " " : s + string(width - chars, ' ');
}

static string ruleName(int d) { return d == 1 ? "1 choice" : to_string(d) + " choices"; }

static void printSettings(const Settings &s) {
    Experiment e;
    string error;
    if (!prepare(s, e, error)) { printf("  efficiencies: %s\n", error.c_str()); return; }
    const vector<double> &f = e.effs;
    double lo = *min_element(f.begin(), f.end()), hi = *max_element(f.begin(), f.end()), sum = 0.0;
    for (double x : f) sum += x;
    printf("  bins         %zu", f.size());
    if (lo == hi) printf(", efficiency %s each\n", sig(lo).c_str());
    else printf(", efficiencies %s (min %s, mean %s, max %s)\n", s.eff.c_str(), sig(lo).c_str(), sig(sum / f.size()).c_str(), sig(hi).c_str());
    if (efficiencySetsBins(s.eff)) printf("               (the number of bins comes from the efficiency list)\n");
    printf("  capacity     %s balls per unit time (average ball size 1)\n", sig(sum).c_str());
    printf("  arrivals     %s, %s balls per unit time: load %.3f%s\n", s.poisson ? "Poisson" : "constant gaps",
           sig(e.lambda).c_str(), e.load, e.load >= 1.0 ? "  (1 or more: queues grow without limit)" : "");
    if (s.sigma > 0) printf("  ball sizes   log-normal, mean 1, sigma %s\n", sig(s.sigma).c_str());
    else printf("  ball sizes   all size 1\n");
    string rules;
    for (size_t i = 0; i < s.choices.size(); i++) rules += (i ? ", " : "") + ruleName(s.choices[i]);
    printf("  rules        %s; least %s wins; %s\n", rules.c_str(), s.metric == LoadMetric::Backlog ? "backlog" : "queue length",
           s.distinct ? "checked bins are distinct" : "checked bins may repeat");
    printf("  length       %lld turns of %zu balls, after %lld warm-up turns%s\n", s.turns, f.size(), e.warmup,
           s.warmup < 0 ? " (auto)" : "");
    printf("  replications %d, seed %llu\n", s.reps, (unsigned long long)s.seed);
}

// M/G/1 mean wait for 1 choice with Poisson arrivals: the hash gives every bin
// lambda / n, so each bin is its own M/G/1 queue (Pollaczek-Khinchine).
static double mg1Prediction(const Experiment &e) {
    if (!e.s.poisson) return nan("");
    double n = e.effs.size(), lam = e.lambda / n, es2 = exp(e.s.sigma * e.s.sigma), total = 0.0;
    for (double f : e.effs) {
        double rho = lam / f;
        if (rho >= 1.0) return nan("");
        total += lam * es2 / (f * f) / (2.0 * (1.0 - rho));
    }
    return total / n;
}

struct Row {
    Summary wait, waited, p50, p95, p99, avgMax, load;
    int globalMax = 0;
    double meanRunMax = 0.0;
};

// The largest offered load of any bin over all runs: work received per unit of
// that bin's capacity. Above 1 its queue grows without limit, so averages
// depend on how long the run is rather than settling.
static double maxOfferedLoad(const PolicyResult &p) {
    if (p.runs.empty()) return 0.0;
    double duration = 0.0, worst = 0.0;
    vector<double> busy(p.runs[0].perBin.size(), 0.0);
    for (const RunResult &r : p.runs) {
        duration += r.duration;
        for (size_t i = 0; i < busy.size(); i++) busy[i] += r.perBin[i].busy;
    }
    for (double b : busy) worst = max(worst, duration > 0 ? b / duration : 0.0);
    return worst;
}

static Row summarizePolicy(const PolicyResult &p) {
    vector<double> wait, waited, p50, p95, p99, avgMax, load;
    Row row;
    for (const RunResult &r : p.runs) {
        wait.push_back(r.meanWait);
        waited.push_back(r.hist.waitedFraction());
        p50.push_back(r.hist.quantile(0.50));
        p95.push_back(r.hist.quantile(0.95));
        p99.push_back(r.hist.quantile(0.99));
        avgMax.push_back(r.avgMaxQueue);
        load.push_back(r.measuredLoad);
        row.globalMax = max(row.globalMax, r.globalMaxQueue);
        row.meanRunMax += r.globalMaxQueue / static_cast<double>(p.runs.size());
    }
    row.wait = summarize(wait);
    row.waited = summarize(waited);
    row.p50 = summarize(p50);
    row.p95 = summarize(p95);
    row.p99 = summarize(p99);
    row.avgMax = summarize(avgMax);
    row.load = summarize(load);
    return row;
}

static void printResults(const Experiment &e) {
    printf("\n  Results: %d replication%s of %lld turns, %s%s\n", e.s.reps, e.s.reps == 1 ? "" : "s", e.s.turns,
           e.s.reps > 1 ? "mean ± 95% confidence interval" : "one run, so no confidence intervals",
           "");
    printf("  %s%s%s%s%s%s%s%s%s\n", pad("rule", 11).c_str(), pad("mean wait", 19).c_str(), pad("waited", 8).c_str(),
           pad("median", 8).c_str(), pad("p95", 8).c_str(), pad("p99", 8).c_str(), pad("avg max queue", 19).c_str(),
           pad("global max", 13).c_str(), "load");
    for (const PolicyResult &p : e.policies) {
        Row r = summarizePolicy(p);
        char waited[16];
        snprintf(waited, sizeof waited, "%.1f%%", 100.0 * r.waited.mean);
        printf("  %s%s%s%s%s%s%s%s%.3f\n", pad(ruleName(p.d), 11).c_str(), pad(pm(r.wait), 19).c_str(), pad(waited, 8).c_str(),
               pad(sig(r.p50.mean), 8).c_str(), pad(sig(r.p95.mean), 8).c_str(), pad(sig(r.p99.mean), 8).c_str(),
               pad(pm(r.avgMax), 19).c_str(), pad(to_string(r.globalMax), 13).c_str(), r.load.mean);
    }
    for (const PolicyResult &p : e.policies) {
        double theory = p.d == 1 ? mg1Prediction(e) : nan("");
        if (!isnan(theory)) printf("  check: M/G/1 theory gives a 1-choice mean wait of %s\n", sig(theory).c_str());
    }
    for (const PolicyResult &p : e.policies) {
        double worst = maxOfferedLoad(p);
        if (worst > 1.0)
            printf("  warning: under %s some bins receive more work than they can process (offered load up to %.0f%%),\n"
                   "           so their queues grow without limit and its averages keep rising with run length\n",
                   ruleName(p.d).c_str(), 100.0 * worst);
    }
    printf("  Wait is the time a ball queues before processing starts. Queue length counts the balls in a bin,\n"
           "  including the one being processed. avg max queue = the longest queue in each turn, averaged over\n"
           "  turns; global max = the longest queue in any run. load = measured work / capacity.  (%.1f s)\n\n",
           e.seconds);
}

static void printPerBin(const Experiment &e) {
    for (const PolicyResult &p : e.policies) {
        size_t n = e.effs.size();
        vector<BinStats> sum(n);
        double duration = 0.0;
        uint64_t total = 0;
        for (const RunResult &r : p.runs) {
            duration += r.duration;
            for (size_t i = 0; i < n; i++) {
                sum[i].eff = r.perBin[i].eff;
                sum[i].balls += r.perBin[i].balls;
                sum[i].waitSum += r.perBin[i].waitSum;
                sum[i].busy += r.perBin[i].busy;
                sum[i].maxQueue = max(sum[i].maxQueue, r.perBin[i].maxQueue);
                total += r.perBin[i].balls;
            }
        }
        // More than 40 bins are grouped: by efficiency value when there are at
        // most 20 distinct values, otherwise into 10 bands of similar efficiency.
        struct Group {
            string label;
            int size = 0;
            BinStats stats;
        };
        auto addTo = [](Group &g, const BinStats &b) {
            g.size++;
            g.stats.balls += b.balls;
            g.stats.waitSum += b.waitSum;
            g.stats.busy += b.busy;
            g.stats.maxQueue = max(g.stats.maxQueue, b.maxQueue);
        };
        vector<size_t> order(n);
        for (size_t i = 0; i < n; i++) order[i] = i;
        sort(order.begin(), order.end(), [&](size_t a, size_t b) { return sum[a].eff < sum[b].eff; });
        vector<Group> groups;
        bool grouped = n > 40;
        string how = "per bin";
        if (grouped) {
            map<double, int> distinct;
            for (size_t i : order) distinct[round(sum[i].eff * 1000) / 1000]++;
            if (distinct.size() <= 20) {
                how = "per efficiency";
                for (size_t i : order) {
                    string label = sig(round(sum[i].eff * 1000) / 1000);
                    if (groups.empty() || groups.back().label != label) groups.push_back(Group{label, 0, BinStats{}});
                    addTo(groups.back(), sum[i]);
                }
            } else {
                how = "in 10 bands of similar efficiency";
                for (size_t band = 0; band < 10; band++) {
                    size_t a = band * n / 10, b = (band + 1) * n / 10;
                    Group g{sig(sum[order[a]].eff, 3) + "-" + sig(sum[order[b - 1]].eff, 3), 0, BinStats{}};
                    for (size_t k = a; k < b; k++) addTo(g, sum[order[k]]);
                    groups.push_back(g);
                }
            }
        }
        printf("\n  %s, %s (all replications)\n", ruleName(p.d).c_str(), how.c_str());
        printf("  %s%s%s%s%s%s\n", pad(grouped ? "efficiency" : "bin", 14).c_str(), pad(grouped ? "bins" : "efficiency", 12).c_str(),
               pad("share of balls", 16).c_str(), pad("offered load", 14).c_str(), pad("mean wait", 11).c_str(), "max queue");
        // offered load: work received per unit of the bin's capacity; above 100% its queue grows without limit
        auto line = [&](const string &a, const string &b, const BinStats &x, int count) {
            char share[16], offered[16];
            snprintf(share, sizeof share, "%.2f%%", total ? 100.0 * x.balls / total : 0.0);
            snprintf(offered, sizeof offered, "%.1f%%", duration > 0 ? 100.0 * x.busy / (duration * count) : 0.0);
            printf("  %s%s%s%s%s%d\n", pad(a, 14).c_str(), pad(b, 12).c_str(), pad(share, 16).c_str(), pad(offered, 14).c_str(),
                   pad(sig(x.balls ? x.waitSum / x.balls : nan("")), 11).c_str(), x.maxQueue);
        };
        if (grouped) {
            for (const Group &g : groups) line(g.label, to_string(g.size), g.stats, g.size);
        } else {
            for (size_t i = 0; i < n; i++) line(to_string(i), sig(sum[i].eff), sum[i], 1);
        }
    }
    printf("\n");
}

// ---------- CSV log ----------

static void logCsv(const string &path, const Experiment &e, const string &sweepName = "", const string &sweepValue = "") {
    if (path.empty()) return;
    bool fresh;
    {
        ifstream in(path);
        fresh = !in.good() || in.peek() == ifstream::traits_type::eof();
    }
    ofstream out(path, ios::app);
    if (!out) { printf("  cannot write %s\n", path.c_str()); return; }
    if (fresh)
        out << "sweep,sweep_value,bins,efficiency,rate,load,arrivals,sigma,metric,distinct,turns,warmup,reps,seed,"
               "choices,mean_wait,mean_wait_ci,waited_fraction,median_wait,p95_wait,p99_wait,avg_max_queue,"
               "avg_max_queue_ci,global_max_queue,mean_run_max_queue,measured_load\n";
    for (const PolicyResult &p : e.policies) {
        Row r = summarizePolicy(p);
        out << sweepName << "," << sweepValue << "," << e.effs.size() << ",\"" << e.s.eff << "\"," << e.lambda << "," << e.load
            << "," << (e.s.poisson ? "poisson" : "constant") << "," << e.s.sigma << ","
            << (e.s.metric == LoadMetric::Backlog ? "backlog" : "length") << "," << (e.s.distinct ? 1 : 0) << ","
            << e.s.turns << "," << e.warmup << "," << e.s.reps << "," << e.s.seed << "," << p.d << "," << r.wait.mean << ","
            << r.wait.half << "," << r.waited.mean << "," << r.p50.mean << "," << r.p95.mean << "," << r.p99.mean << ","
            << r.avgMax.mean << "," << r.avgMax.half << "," << r.globalMax << "," << r.meanRunMax << "," << r.load.mean << "\n";
    }
}

// ---------- commands ----------

static const char *SETTING_NAMES[] = {"bins", "eff", "load", "rate", "arrivals", "sigma", "choices",
                                      "metric", "replace", "turns", "warmup", "reps", "seed"};

static bool isSetting(const string &name) {
    for (const char *s : SETTING_NAMES)
        if (name == s) return true;
    return false;
}

// Apply one setting; returns an error message, or "" if it worked.
static string applySetting(Settings &s, const string &name, const string &value) {
    double v;
    long long k;
    if (name == "bins") {
        if (!toInt(value, k) || k < 1 || k > 100000) return "bins must be a whole number from 1 to 100000";
        s.bins = static_cast<int>(k);
    } else if (name == "eff") {
        vector<double> e;
        string error;
        if (!parseEfficiencies(value, s.bins, s.seed, e, error)) return error;
        s.eff = value;
    } else if (name == "load") {
        if (!toNumber(value, v) || v <= 0) return "load must be above 0, as a decimal (0.9) or a fraction (9/10)";
        s.load = v;
        s.rate = 0.0;
    } else if (name == "rate") {
        if (!toNumber(value, v) || v < 0) return "rate must be 0 or more balls per unit time (0 means: use the load)";
        s.rate = v;
    } else if (name == "arrivals") {
        if (value == "poisson") s.poisson = true;
        else if (value == "constant") s.poisson = false;
        else return "arrivals must be poisson or constant";
    } else if (name == "sigma") {
        if (!toNumber(value, v) || v < 0 || v > 5) return "sigma must be from 0 to 5 (0 makes every ball size 1)";
        s.sigma = v;
    } else if (name == "choices") {
        vector<int> c;
        for (const string &part : split(value, ',')) {
            if (!toInt(part, k) || k < 1 || k > 1000) return "choices must be a list of whole numbers, such as 1,2";
            c.push_back(static_cast<int>(k));
        }
        s.choices = c;
    } else if (name == "metric") {
        if (value == "backlog") s.metric = LoadMetric::Backlog;
        else if (value == "length") s.metric = LoadMetric::QueueLength;
        else return "metric must be backlog (time to clear the queue) or length (balls in the bin)";
    } else if (name == "replace") {
        if (value == "on") s.distinct = false;
        else if (value == "off") s.distinct = true;
        else return "replace must be on (checked bins may repeat) or off (always distinct)";
    } else if (name == "turns") {
        if (!toInt(value, k) || k < 1) return "turns must be a whole number, 1 or more";
        s.turns = k;
    } else if (name == "warmup") {
        if (value == "auto") s.warmup = -1;
        else if (!toInt(value, k) || k < 0) return "warmup must be a whole number of turns, or auto";
        else s.warmup = k;
    } else if (name == "reps") {
        if (!toInt(value, k) || k < 1 || k > 1000) return "reps must be a whole number from 1 to 1000";
        s.reps = static_cast<int>(k);
    } else if (name == "seed") {
        if (!toInt(value, k) || k < 0) return "seed must be a whole number, 0 or more";
        s.seed = static_cast<uint64_t>(k);
    } else {
        return "there is no setting called '" + name + "'; type help for the list";
    }
    return "";
}

static void help() {
    printf(
        "\n  Commands\n"
        "    show                       current settings\n"
        "    set NAME VALUE             change a setting (or just: NAME VALUE)\n"
        "    run                        run every rule and print the results\n"
        "    sweep NAME V1 V2 ...       run once per value of one setting, then restore it\n"
        "    perbin                     per-bin breakdown of the last run\n"
        "    csv FILE | csv off         append every result to a CSV file\n"
        "    reset                      default settings\n"
        "    help, quit\n"
        "  Several commands can go on one line, separated by ;\n\n"
        "  Settings\n"
        "    bins N           number of bins (default 10)\n"
        "    eff SPEC         bin efficiencies (default 1): a bin with efficiency e processes a size-s\n"
        "                     ball in s/e time. SPEC is one of\n"
        "                       1.5              every bin the same\n"
        "                       1,1,2,0.5        one per bin (sets the number of bins)\n"
        "                       8x1,2x2          8 bins at 1 and 2 at 2 (sets the number of bins)\n"
        "                       uniform:0.5:1.5  drawn uniformly per bin\n"
        "                       lognormal:0.3    drawn per bin, mean 1\n"
        "    load L           arrival rate as a fraction of total capacity, e.g. 0.9 or 19/20 (default 0.9)\n"
        "    rate R           arrival rate in balls per unit time instead of load (0 = use load)\n"
        "    arrivals TYPE    poisson (exponential gaps) or constant (default poisson)\n"
        "    sigma S          ball sizes are log-normal with mean 1 and this sigma; 0 = all size 1 (default 1)\n"
        "    choices LIST     rules to compare: bins checked per ball, e.g. 1,2 or 1,2,3 (default 1,2)\n"
        "    metric M         backlog (default) or length: what 'least loaded' means\n"
        "    replace on|off   whether a ball's checked bins may repeat (default off)\n"
        "    turns T          measured turns; one turn = one ball per bin (default 10000)\n"
        "    warmup W|auto    turns run before measuring (default auto)\n"
        "    reps R           replications, each with its own seed (default 5)\n"
        "    seed X           base seed (default 1)\n\n"
        "  Example:  bins 20; eff 16x1,4x2; load 0.95; sigma 0.5; run; perbin\n\n");
}

int main(int argc, char **argv) {
    Settings s;
    Experiment last;
    bool haveLast = false;
    string csvPath;
    int errors = 0;

    // returns false to quit
    auto execute = [&](const vector<string> &w) -> bool {
        if (w.empty()) return true;
        const string &cmd = w[0];
        auto fail = [&](const string &msg) { printf("  %s\n", msg.c_str()); errors++; };
        if (cmd == "quit" || cmd == "exit") return false;
        if (cmd == "help" || cmd == "?") { help(); return true; }
        if (cmd == "show") { printSettings(s); return true; }
        if (cmd == "reset") { s = Settings(); printSettings(s); return true; }
        if (cmd == "set" || isSetting(cmd)) {
            size_t at = cmd == "set" ? 1 : 0;
            if (w.size() != at + 2) { fail("usage: set NAME VALUE (one value, no spaces)"); return true; }
            string error = applySetting(s, w[at], w[at + 1]);
            if (!error.empty()) fail(error);
            return true;
        }
        if (cmd == "run") {
            Experiment probe;
            string error;
            if (!prepare(s, probe, error)) { fail(error); return true; }
            printf("\n");
            printSettings(s);
            last = runExperiment(s, true);
            haveLast = true;
            printResults(last);
            logCsv(csvPath, last);
            return true;
        }
        if (cmd == "sweep") {
            if (w.size() < 3 || !isSetting(w[1])) { fail("usage: sweep NAME V1 V2 ...  (NAME is a setting)"); return true; }
            Settings saved = s;
            printf("\n");
            printSettings(s);
            printf("\n  Sweep over %s (mean over replications)\n", w[1].c_str());
            printf("  %s%s%s%s%s%s\n", pad(w[1], 12).c_str(), pad("rule", 11).c_str(), pad("mean wait", 19).c_str(),
                   pad("waited", 8).c_str(), pad("avg max queue", 19).c_str(), "global max");
            for (size_t i = 2; i < w.size(); i++) {
                string error = applySetting(s, w[1], w[i]);
                Experiment probe;
                if (error.empty() && !prepare(s, probe, error)) {}
                if (!error.empty()) { fail(w[1] + " " + w[i] + ": " + error); break; }
                Experiment e = runExperiment(s, true);
                for (const PolicyResult &p : e.policies) {
                    Row r = summarizePolicy(p);
                    char waited[16];
                    snprintf(waited, sizeof waited, "%.1f%%", 100.0 * r.waited.mean);
                    printf("  %s%s%s%s%s%d%s\n", pad(w[i], 12).c_str(), pad(ruleName(p.d), 11).c_str(), pad(pm(r.wait), 19).c_str(),
                           pad(waited, 8).c_str(), pad(pm(r.avgMax), 19).c_str(), r.globalMax,
                           maxOfferedLoad(p) > 1.0 ? "  (some bins overloaded: queues grow)" : "");
                }
                logCsv(csvPath, e, w[1], w[i]);
                last = e;
                haveLast = true;
            }
            printf("\n");
            s = saved;
            return true;
        }
        if (cmd == "perbin") {
            if (!haveLast) fail("nothing has run yet: type run first");
            else printPerBin(last);
            return true;
        }
        if (cmd == "csv") {
            if (w.size() != 2) { fail("usage: csv FILE  or  csv off"); return true; }
            csvPath = w[1] == "off" ? "" : w[1];
            printf("  %s\n", csvPath.empty() ? "CSV logging off" : ("results will be appended to " + csvPath).c_str());
            return true;
        }
        fail("unknown command '" + cmd + "'; type help");
        return true;
    };

    auto executeLine = [&](const string &line) -> bool {
        for (const string &part : split(line, ';'))
            if (!execute(words(part))) return false;
        return true;
    };

    // Flags: --NAME VALUE sets a setting; --run, --perbin, --sweep "NAME V1 V2", --csv FILE.
    // Any of --run / --sweep runs those and exits; otherwise the shell starts.
    bool batch = false;
    vector<string> script;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") { help(); return 0; }
        if (arg == "--run") { script.push_back("run"); batch = true; continue; }
        if (arg == "--perbin") { script.push_back("perbin"); continue; }
        if (arg.rfind("--", 0) == 0 && i + 1 < argc) {
            string name = arg.substr(2), value = argv[++i];
            if (name == "sweep") { script.push_back("sweep " + value); batch = true; }
            else if (name == "csv") script.push_back("csv " + value);
            else script.push_back("set " + name + " " + value);
            continue;
        }
        fprintf(stderr, "unrecognised argument '%s'; try --help\n", arg.c_str());
        return 2;
    }
    for (const string &line : script)
        if (!executeLine(line)) return errors ? 1 : 0;
    if (batch) return errors ? 1 : 0;

    bool interactive = isatty(0);
    if (interactive) {
        printf("Balls-into-bins lab. Type help for commands, show for the settings, run to run.\n");
        printSettings(s);
    }
    for (string line;;) {
        if (interactive) { printf("lab> "); fflush(stdout); }
        if (!getline(cin, line)) break;
        if (!executeLine(line)) break;
    }
    return errors && !interactive ? 1 : 0;
}

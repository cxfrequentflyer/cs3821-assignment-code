// Transition matrix of the number of idle bins, 1 choice vs 2 choices.
//
// Every bin has efficiency 1 and every ball has size 1. A ball arrives every
// 1 / (load * bins) time units, so the bins are busy a fraction `load` of the
// time. One transition is one turn of K balls: the number of idle bins is
// counted at the arrival instant of the turn's first ball, before any bin finishing
// at that instant is processed; the K balls are distributed, and then it is
// counted again.
//
// The program runs ../Simulator.hpp once with each rule, prints both matrices
// and writes each one to a CSV table. Every run also reports:
//  - batch-means standard errors and a first-half/second-half trend check;
//  - the two-transition matrix against the one-transition matrix squared,
//    which would agree if the idle count were a Markov chain;
//  - mixing: how far the k-step matrix's rows still are from the share, for
//    k up to --max-lag, measured and as a Markov chain would have it;
//  - for one ball per turn on a coarse enough grid, the two-choice matrix
//    against the exact Markov chain on sorted bin backlogs.
//
//   ./build/transition_matrix                                (asks for the inputs)
//   ./build/transition_matrix --bins 20 --load 0.95 --balls 20

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "Simulator.hpp"

using namespace std;

using Matrix = vector<vector<double>>;

struct Result {
    Matrix P;                   // P[i][j]: the next count is j, given this one is i
    Matrix P2;                  // the same, two transitions ahead (measured only)
    vector<double> share;       // fraction of transitions that start from i idle bins
    vector<uint64_t> observed;  // how many transitions start from i (measured only)
    double meanWait = 0.0;

    // convergence (measured only)
    vector<int> lags;           // k for each entry of `distance`
    vector<double> distance;    // largest total-variation distance from the share of a row of the k-step matrix
    vector<double> adjacent;    // average earth mover's distance between adjacent rows of the k-step matrix
    int batches = 0;            // batches behind the standard errors (0: run too short for them)
    double batchLength = 0.0;   // service times per batch
    double seIdle = 0.0, seWait = 0.0;        // batch-means standard errors of the mean idle count and mean wait
    double trendIdle = 0.0, trendWait = 0.0;  // second half minus first half, in standard errors
};

// k = 1, 2, 3, ... growing by about 30% a step, up to maxLag.
static vector<int> lagSet(int maxLag) {
    vector<int> lags;
    for (int k = 1; k <= maxLag; k = max(k + 1, static_cast<int>(lround(k * 1.3)))) lags.push_back(k);
    return lags;
}

// Row-normalize a flattened size x size table of counts.
static Matrix rowNormalize(const vector<uint64_t> &counts, size_t size) {
    Matrix P(size, vector<double>(size, 0.0));
    for (size_t i = 0; i < size; i++) {
        uint64_t total = accumulate(counts.begin() + i * size, counts.begin() + (i + 1) * size, uint64_t(0));
        for (size_t j = 0; total && j < size; j++) {
            P[i][j] = static_cast<double>(counts[i * size + j]) / total;
        }
    }
    return P;
}

// Largest total-variation distance between a row of P and the share, over
// rows that start at least 1% of transitions: how far the chain still is from
// having forgotten where it started.
static double distanceFromShare(const Matrix &P, const vector<double> &share) {
    double worst = 0.0;
    for (size_t i = 0; i < P.size(); i++) {
        if (share[i] < 0.01) continue;
        double tv = 0.0;
        for (size_t j = 0; j < P.size(); j++) tv += fabs(P[i][j] - share[j]);
        worst = max(worst, tv / 2);
    }
    return worst;
}

// Average earth mover's distance between adjacent rows i and i+1, over pairs
// where both rows start at least 1% of transitions. On a line it is the area
// between the two rows' cumulative distributions. 0 means the next count
// doesn't depend on whether the chain started one bin higher or lower.
static double adjacentRowDistance(const Matrix &P, const vector<double> &share) {
    double total = 0.0;
    int pairs = 0;
    for (size_t i = 0; i + 1 < P.size(); i++) {
        if (share[i] < 0.01 || share[i + 1] < 0.01) continue;
        double gap = 0.0, Fa = 0.0, Fb = 0.0;
        for (size_t j = 0; j + 1 < P.size(); j++) {
            Fa += P[i][j];
            Fb += P[i + 1][j];
            gap += fabs(Fa - Fb);
        }
        total += gap;
        pairs++;
    }
    return pairs ? total / pairs : 0.0;
}

// Mean and batch-means standard error of a series of batch averages, plus the
// gap between its second and first halves in standard errors (a trend check).
static void batchStats(const vector<double> &x, double &se, double &trend) {
    size_t b = x.size(), h = b / 2;
    double mean = accumulate(x.begin(), x.end(), 0.0) / b, var = 0.0;
    for (double v : x) var += (v - mean) * (v - mean);
    double sd = sqrt(var / (b - 1));
    se = sd / sqrt(static_cast<double>(b));
    double first = accumulate(x.begin(), x.begin() + h, 0.0) / h;
    double second = accumulate(x.begin() + h, x.end(), 0.0) / (b - h);
    double seGap = sd * sqrt(1.0 / h + 1.0 / (b - h));
    trend = seGap > 0.0 ? (second - first) / seGap : 0.0;
}

static Matrix multiply(const Matrix &A, const Matrix &B) {
    size_t n = A.size();
    Matrix C(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < n; k++)
            for (size_t j = 0; j < n; j++) C[i][j] += A[i][k] * B[k][j];
    return C;
}

// P^k by repeated squaring.
static Matrix matrixPower(Matrix base, int k) {
    Matrix result(base.size(), vector<double>(base.size(), 0.0));
    for (size_t i = 0; i < base.size(); i++) result[i][i] = 1.0;
    for (; k > 0; k >>= 1) {
        if (k & 1) result = multiply(result, base);
        base = multiply(base, base);
    }
    return result;
}

// Run the simulator with d choices and record the idle count once per turn.
static Result measure(int bins, double load, int balls, uint64_t transitions, int d, uint64_t seed, int maxLag,
                      int batches) {
    const double gap = 1.0 / (load * bins);
    auto source = make_shared<Source>(gap, false, 0.0, seed);
    Simulator sim(source, d, seed);
    for (int i = 0; i < bins; i++) sim.addBin(1.0);

    // Count before processing anything at instant t: a bin whose last ball
    // finishes exactly at t is still busy. A real finishing time is a whole
    // number of grid steps, and the grid is at least gap / 1000 for a load with
    // up to three decimals, so a bin is idle only if it emptied more than
    // `tolerance` before t. Anything closer is rounding in the floating-point
    // clock around a finish at t itself.
    const double tolerance = gap * 1e-4;
    auto idleAt = [&](double t) {
        int idle = 0;
        for (const auto &bin : sim.activeBins()) idle += bin->backlog(t - tolerance) <= 0.0;
        return idle;
    };

    // Let the queues settle first: roughly 20 relaxation times of 1 / (1 - load)^2.
    double settle = 20.0 * load * bins / ((1.0 - load) * (1.0 - load));
    uint64_t warmupBalls = static_cast<uint64_t>(min(2e7, max(1e5, settle)));
    uint64_t warmup = max<uint64_t>(2, warmupBalls / balls);

    // Pairs of measured counts k transitions apart, for every lag k; lag 1 is
    // the transition matrix and lag 2 the Markov check. A ring buffer keeps the
    // counts the longest lag still needs.
    const size_t size = bins + 1;
    const vector<int> lags = lagSet(max(2, maxLag));
    const size_t depth = lags.back() + 1;
    vector<vector<uint64_t>> pairs(lags.size(), vector<uint64_t>(size * size, 0));
    vector<int> history(depth, 0);
    vector<uint64_t> seen(size, 0);

    // Batch means: the measured period is cut into equal batches. Each batch
    // needs enough transitions to hold a few balls, or its averages are noise.
    const uint64_t perBatch = transitions / max(1, batches);
    const bool useBatches = batches >= 4 && perBatch >= 100;
    vector<double> batchIdle, batchWait;
    double idleSum = 0.0, waitMark = 0.0, waitBefore = 0.0;
    uint64_t finishedMark = 0, finishedBefore = 0;

    for (uint64_t k = 0; k < warmup + transitions; k++) {
        if (k == warmup) {   // the mean wait covers only balls finished after the warm-up
            waitBefore = waitMark = sim.totalQueueingDelay();
            finishedBefore = finishedMark = sim.ballsProcessed();
        }
        double t = source->nextActionTime();   // the turn's first ball arrives at t; every earlier ball is placed
        int idle = idleAt(t);
        if (k >= warmup) {
            uint64_t m = k - warmup;           // index of this count among the measured ones
            seen[idle]++;
            for (size_t l = 0; l < lags.size() && static_cast<uint64_t>(lags[l]) <= m; l++)
                pairs[l][history[(m - lags[l]) % depth] * size + idle]++;
            history[m % depth] = idle;
            idleSum += idle;
            if (useBatches && (m + 1) % perBatch == 0 && batchIdle.size() < static_cast<size_t>(batches)) {
                double w = sim.totalQueueingDelay();
                uint64_t f = sim.ballsProcessed();
                batchIdle.push_back(idleSum / perBatch);
                batchWait.push_back((w - waitMark) / static_cast<double>(max<uint64_t>(1, f - finishedMark)));
                idleSum = 0.0;
                waitMark = w;
                finishedMark = f;
            }
        }
        for (int b = 0; b < balls; b++) sim.runUntil(source->nextActionTime() + gap / 2);   // place this turn's balls
    }

    Result r;
    r.P = rowNormalize(pairs[0], size);
    r.P2 = rowNormalize(pairs[1], size);
    for (size_t i = 0; i < size; i++) {
        r.share.push_back(static_cast<double>(seen[i]) / transitions);
        r.observed.push_back(seen[i]);
    }
    r.meanWait = (sim.totalQueueingDelay() - waitBefore) / static_cast<double>(sim.ballsProcessed() - finishedBefore);
    r.lags = lags;
    for (size_t l = 0; l < lags.size(); l++) {
        Matrix Pk = rowNormalize(pairs[l], size);
        r.distance.push_back(distanceFromShare(Pk, r.share));
        r.adjacent.push_back(adjacentRowDistance(Pk, r.share));
    }
    if (useBatches) {
        r.batches = static_cast<int>(batchIdle.size());
        r.batchLength = perBatch * balls * gap;
        batchStats(batchIdle, r.seIdle, r.trendIdle);
        batchStats(batchWait, r.seWait, r.trendWait);
    }
    return r;
}

// Exact two-choice matrix for one ball per turn. Time is counted in steps: a
// ball arrives every G steps and takes S steps. The state is, for every bin,
// the time until it empties, taken at an arrival instant before anything there
// is processed: 0 means its last ball finishes right now (still busy), -1 means
// it emptied earlier (idle). Sorted, since bins are interchangeable; values
// above capS service times are cut to the cap. Stored as value + 1.
static Result exactTwoChoice(int n, int G, int S, int capS, size_t &states, double &capMass) {
    const int CAP = capS * S;
    if (n > 9 || CAP > 126) throw invalid_argument("it supports at most 9 bins and a cap of at most 126 grid steps");
    auto pack = [](vector<int> v) {
        sort(v.begin(), v.end());
        uint64_t key = 0;
        for (int x : v) key = key << 7 | static_cast<uint64_t>(x + 1);
        return key;
    };
    vector<uint64_t> code;
    unordered_map<uint64_t, uint32_t> index;
    auto id = [&](const vector<int> &v) {
        auto [it, fresh] = index.try_emplace(pack(v), static_cast<uint32_t>(code.size()));
        if (fresh) code.push_back(it->first);
        return it->second;
    };
    id(vector<int>(n, -1));   // every bin long empty

    vector<vector<pair<uint32_t, double>>> next;
    vector<int> idle;
    vector<double> capHit;
    const double p = 2.0 / (n * (n - 1));
    for (size_t s = 0; s < code.size(); s++) {
        vector<int> v(n);
        uint64_t key = code[s];
        for (int i = n - 1; i >= 0; i--) { v[i] = static_cast<int>(key & 127) - 1; key >>= 7; }
        map<uint32_t, double> out;
        double hit = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {   // v is sorted, so bin i has the smaller backlog
                vector<int> u = v;
                for (int &x : u) x = max(0, x);   // process the finishes at this instant
                u[i] += S;                         // place the ball
                if (u[i] > CAP) { u[i] = CAP; hit += p; }
                for (int &x : u) x = max(-1, x - G);   // the next arrival, before it is processed
                out[id(u)] += p;
            }
        }
        next.emplace_back(out.begin(), out.end());
        idle.push_back(static_cast<int>(count_if(v.begin(), v.end(), [](int x) { return x < 0; })));
        capHit.push_back(hit);
        if (code.size() > 2000000) throw runtime_error("it has more than 2 million states, too many to solve here");
    }

    // stationary distribution of the lazy chain (same answer, never periodic)
    size_t N = code.size();
    vector<double> pi(N, 0.0), nx(N);
    pi[0] = 1.0;
    for (int it = 0; it < 1000000; it++) {
        fill(nx.begin(), nx.end(), 0.0);
        for (size_t s = 0; s < N; s++) {
            nx[s] += 0.5 * pi[s];
            for (auto [t, q] : next[s]) nx[t] += 0.5 * pi[s] * q;
        }
        double total = accumulate(nx.begin(), nx.end(), 0.0), diff = 0.0;
        for (size_t s = 0; s < N; s++) { nx[s] /= total; diff += fabs(nx[s] - pi[s]); }
        swap(pi, nx);
        if (diff < 1e-14) break;
    }

    // lump the chain onto the idle count
    Result r;
    r.P.assign(n + 1, vector<double>(n + 1, 0.0));
    r.share.assign(n + 1, 0.0);
    capMass = 0.0;
    for (size_t s = 0; s < N; s++) {
        r.share[idle[s]] += pi[s];
        capMass += pi[s] * capHit[s];
        for (auto [t, q] : next[s]) r.P[idle[s]][idle[t]] += pi[s] * q;
    }
    for (int i = 0; i <= n; i++)
        for (int j = 0; r.share[i] > 0 && j <= n; j++) r.P[i][j] /= r.share[i];
    states = N;
    return r;
}

// ---------- summaries ----------

static double meanOf(const vector<double> &share) {
    double m = 0.0;
    for (size_t i = 0; i < share.size(); i++) m += static_cast<double>(i) * share[i];
    return m;
}

// Correlation between one count and the next: 0 means the next count has
// forgotten the current one, 1 means it never moves.
static double lagCorrelation(const Result &r) {
    double mean = meanOf(r.share), var = 0.0, cov = 0.0;
    for (size_t i = 0; i < r.share.size(); i++) {
        double expectNext = 0.0;
        for (size_t j = 0; j < r.P.size(); j++) expectNext += r.P[i][j] * static_cast<double>(j);
        var += r.share[i] * (static_cast<double>(i) - mean) * (static_cast<double>(i) - mean);
        cov += r.share[i] * (static_cast<double>(i) - mean) * (expectNext - mean);
    }
    return var > 0.0 ? cov / var : 0.0;
}

// Largest entry-wise gap over rows that start at least 1% of transitions in
// both, so that rarely visited rows (and their sampling noise) don't dominate.
static double maxDiff(const Matrix &A, const Matrix &B, const vector<double> &shareA, const vector<double> &shareB) {
    double worst = 0.0;
    for (size_t i = 0; i < A.size(); i++) {
        if (shareA[i] < 0.01 || shareB[i] < 0.01) continue;
        for (size_t j = 0; j < A.size(); j++) worst = max(worst, fabs(A[i][j] - B[i][j]));
    }
    return worst;
}

// The idle counts that ever occur in any of the results, as a state or a destination.
static pair<int, int> occurringRange(const vector<const Result *> &results) {
    int n = static_cast<int>(results[0]->P.size()) - 1, lo = n, hi = 0;
    for (const Result *r : results) {
        for (int i = 0; i <= n; i++) {
            if (r->share[i] < 1e-12) continue;
            lo = min(lo, i);
            hi = max(hi, i);
            for (int j = 0; j <= n; j++)
                if (r->P[i][j] > 0) { lo = min(lo, j); hi = max(hi, j); }
        }
    }
    return {lo, hi};
}

// ---------- output ----------

static void printMatrix(const string &title, const Result &r, int lo, int hi) {
    printf("%s\n", title.c_str());
    printf("  idle now \\ next");
    for (int j = lo; j <= hi; j++) printf("%8d", j);
    printf("     share\n");
    for (int i = lo; i <= hi; i++) {
        printf("  %15d ", i);
        for (int j = lo; j <= hi; j++) {
            if (r.share[i] < 1e-12) printf("%8s", "-");
            else if (r.P[i][j] == 0.0) printf("%8s", ".");
            else printf("%8.4f", r.P[i][j]);
        }
        printf("   %7.2f%%\n", 100.0 * r.share[i]);
    }
}

// One row per current idle count, one column per next idle count. Rows that
// never occur are left blank.
static void writeCsv(const string &path, const Result &r, int lo, int hi) {
    ofstream out(path);
    if (!out) throw runtime_error("cannot write " + path);
    out << "idle now";
    for (int j = lo; j <= hi; j++) out << ",next " << j;
    out << ",share of transitions";
    if (!r.observed.empty()) out << ",transitions observed";
    out << "\n";
    char buf[32];
    for (int i = lo; i <= hi; i++) {
        out << i;
        bool occurs = r.share[i] >= 1e-12;
        for (int j = lo; j <= hi; j++) {
            out << ",";
            if (occurs) { snprintf(buf, sizeof buf, "%.6f", r.P[i][j]); out << buf; }
        }
        snprintf(buf, sizeof buf, "%.6f", r.share[i]);
        out << "," << buf;
        if (!r.observed.empty()) out << "," << r.observed[i];
        out << "\n";
    }
}

// How many transitions until every row is within MIX_EPS of the share,
// measured and as a Markov chain with the same one-step matrix would have it.
static const double MIX_EPS = 0.05;
// `reached` and `still` word the result, e.g. "every row is within 0.05 of the share".
static string mixingSummary(const vector<int> &lags, const vector<double> &measured, const vector<double> &model,
                            const char *reached = "every row is within %.2f of the share",
                            const char *still = "rows are still %.2f from the share") {
    auto firstWithin = [&](const vector<double> &dist) {
        for (size_t l = 0; l < lags.size(); l++)
            if (dist[l] <= MIX_EPS) return lags[l];
        return -1;
    };
    int k = firstWithin(measured), km = firstWithin(model);
    char phrase[120], buf[200];
    if (k > 0) {
        snprintf(phrase, sizeof phrase, reached, MIX_EPS);
        snprintf(buf, sizeof buf, "%s after %d transition%s", phrase, k, k == 1 ? "" : "s");
    } else {
        snprintf(phrase, sizeof phrase, still, measured.back());
        snprintf(buf, sizeof buf, "%s after %d transitions (raise --max-lag)", phrase, lags.back());
    }
    string s = buf;
    if (km > 0) snprintf(buf, sizeof buf, " (a Markov chain with this matrix: %d)", km);
    else snprintf(buf, sizeof buf, " (a Markov chain with this matrix: more than %d)", lags.back());
    return s + buf;
}

static string fmtSetting(double v) {
    char buf[32];
    snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

// ---------- input ----------

static void usage() {
    fprintf(stderr,
            "usage: transition_matrix [--bins N] [--load L] [--balls K]\n"
            "                         [--transitions T] [--seed X] [--out DIR] [--cap C]\n"
            "                         [--max-lag M] [--batches B]\n"
            "Any of --bins, --load and --balls left out is asked for.\n");
    exit(2);
}

// A whole number, or false if the text isn't one.
static bool parseInt(const string &text, int &v) {
    istringstream in(text);
    return in >> v && (in >> ws).eof();
}

// A load as a decimal (0.95) or a fraction (19/20), or false if it is neither.
// Fractions matter for loads like 5/6, which no decimal gives exactly.
static bool parseLoad(const string &text, double &v) {
    istringstream in(text);
    double num = 0.0, den = 1.0;
    char slash = 0;
    if (!(in >> num)) return false;
    if (in >> slash && (slash != '/' || !(in >> den) || den == 0.0)) return false;
    in.clear();
    if (!(in >> ws).eof()) return false;
    v = num / den;
    return true;
}

// Ask until the answer parses and passes `ok`.
template <class T, class Parse, class Check>
static T ask(const string &prompt, Parse parse, Check ok, const string &rule) {
    for (;;) {
        cout << prompt << flush;
        string line;
        if (!getline(cin, line)) { cerr << "\nNo input given.\n"; exit(2); }
        T v{};
        if (parse(line, v) && ok(v)) return v;
        cout << "  " << rule << "\n";
    }
}

int main(int argc, char **argv) {
    int bins = 0, balls = 0, capS = 4, maxLag = 0, batches = 20;
    double load = 0.0;
    uint64_t transitions = 500000, seed = 1;
    string outDir = "results";
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (i + 1 >= argc) usage();
        if (arg == "--bins") bins = atoi(argv[++i]);
        else if (arg == "--load") { if (!parseLoad(argv[++i], load)) load = -1.0; }
        else if (arg == "--balls") balls = atoi(argv[++i]);
        else if (arg == "--transitions") transitions = strtoull(argv[++i], nullptr, 10);
        else if (arg == "--seed") seed = strtoull(argv[++i], nullptr, 10);
        else if (arg == "--out") outDir = argv[++i];
        else if (arg == "--cap") capS = atoi(argv[++i]);
        else if (arg == "--max-lag") maxLag = atoi(argv[++i]);
        else if (arg == "--batches") batches = atoi(argv[++i]);
        else usage();
    }

    const string binRule = "Enter a whole number of bins, 2 or more.";
    const string loadRule = "Enter a load between 0 and 1, as a decimal (0.95) or a fraction (19/20). At 1 or above "
                            "the queues grow without limit.";
    const string ballRule = "Enter a whole number of balls, 1 or more.";
    auto binsOk = [](int v) { return v >= 2; };
    auto loadOk = [](double v) { return v > 0.0 && v < 1.0; };
    auto ballsOk = [](int v) { return v >= 1; };
    if (argc > 1 && ((bins && !binsOk(bins)) || (load != 0.0 && !loadOk(load)) || (balls && !ballsOk(balls)))) {
        if (bins && !binsOk(bins)) fprintf(stderr, "%s\n", binRule.c_str());
        if (load != 0.0 && !loadOk(load)) fprintf(stderr, "%s\n", loadRule.c_str());
        if (balls && !ballsOk(balls)) fprintf(stderr, "%s\n", ballRule.c_str());
        return 2;
    }
    if (!bins) bins = ask<int>("Number of bins: ", parseInt, binsOk, binRule);
    if (load == 0.0) load = ask<double>("Load factor (0 to 1, e.g. 0.95 or 19/20): ", parseLoad, loadOk, loadRule);
    if (!balls) balls = ask<int>("Balls distributed per transition: ", parseInt, ballsOk, ballRule);
    if (transitions < 10) usage();
    if (maxLag <= 0) maxLag = balls == 1 ? 500 * bins : 500;   // about 500 turns of n balls either way

    const double gap = 1.0 / (load * bins);
    printf("\n%d bins, load %s: one size-1 ball every %.6g time units. %d ball%s per transition, %llu transitions.\n",
           bins, fmtSetting(load).c_str(), gap, balls, balls == 1 ? "" : "s", (unsigned long long)transitions);
    printf("Idle bins are counted at each transition's first arrival, before anything at that instant is processed. Their time average is %.3f under "
           "either rule.\n\n", bins * (1.0 - load));

    Result measured[3];
    for (int d = 1; d <= 2; d++) {
        printf("Running %s...\n", d == 1 ? "1 choice" : "2 choices");
        fflush(stdout);
        measured[d] = measure(bins, load, balls, transitions, d, seed, maxLag, batches);
    }
    printf("\n");

    // exact two-choice chain, when it applies
    Result exact;
    bool haveExact = false;
    size_t exactStates = 0;
    double capMass = 0.0;
    string exactNote;
    int S = 0;
    for (int s = 1; s <= 64 && !S; s++)
        if (fabs(gap * s - round(gap * s)) < 1e-9) S = s;
    if (balls != 1) {
        exactNote = "only computed for one ball per transition";
    } else if (!S || round(gap * S) < 1) {
        exactNote = "the gap is not a multiple of 1/S for any S up to 64";
    } else {
        try {
            exact = exactTwoChoice(bins, static_cast<int>(round(gap * S)), S, capS, exactStates, capMass);
            haveExact = true;
        } catch (const exception &e) {
            exactNote = "backlogs sit on a grid of 1/" + to_string(S) + ", giving " + to_string(capS * S) +
                        " levels per bin with a cap of " + to_string(capS) + " service times, and " + e.what();
        }
    }

    vector<const Result *> all = {&measured[1], &measured[2]};
    if (haveExact) all.push_back(&exact);
    auto [lo, hi] = occurringRange(all);
    if (lo > 0 || hi < bins) printf("(idle counts outside %d to %d never occur, so they are left out)\n\n", lo, hi);

    const double settle = 1.0 / ((1.0 - load) * (1.0 - load));   // service times a one-choice queue takes to settle
    vector<double> modelDistance[3], modelAdjacent[3];
    for (int d = 1; d <= 2; d++) {
        const Result &r = measured[d];
        // the same distances for a Markov chain with matrix P, from its powers
        for (int lag : r.lags) {
            Matrix Pk = matrixPower(r.P, lag);
            modelDistance[d].push_back(distanceFromShare(Pk, r.share));
            modelAdjacent[d].push_back(adjacentRowDistance(Pk, r.share));
        }

        printMatrix(d == 1 ? "1 choice" : "2 choices", r, lo, hi);
        if (r.batches) {
            printf("  mean idle bins %.4f ± %.4f, mean wait %.4f ± %.4f (batch-means standard errors, %d batches "
                   "of %.3g service times)\n", meanOf(r.share), r.seIdle, r.meanWait, r.seWait, r.batches, r.batchLength);
            printf("  trend check, second half minus first in standard errors: idle %+.1f, wait %+.1f\n",
                   r.trendIdle, r.trendWait);
            if (r.batchLength < 5 * settle)
                printf("  (each batch is shorter than 5 x %.3g service times, the settling time of a one-choice queue, so "
                       "these errors may be too small: compare seeds)\n", settle);
        } else {
            printf("  mean idle bins %.3f, mean wait %.4f (too few transitions for batch-means errors)\n",
                   meanOf(r.share), r.meanWait);
        }
        printf("  correlation with the next count %.3f\n", lagCorrelation(r));
        printf("  Markov check: two-transition matrix vs one-transition squared, largest gap %.4f\n",
               maxDiff(r.P2, multiply(r.P, r.P), r.share, r.share));
        printf("  mixing: %s\n", mixingSummary(r.lags, r.distance, modelDistance[d]).c_str());
        printf("  adjacent rows: earth mover's distance %.3f after 1 transition; %s\n\n", r.adjacent[0],
               mixingSummary(r.lags, r.adjacent, modelAdjacent[d], "below %.2f", "still %.3f").c_str());
    }
    if (haveExact) {
        printMatrix("2 choices, exact (" + to_string(exactStates) + "-state chain)", exact, lo, hi);
        printf("  probability of reaching the backlog cap of %d service times %.1e\n", capS, capMass);
        printf("  largest gap from the measured matrix %.4f\n\n",
               maxDiff(exact.P, measured[2].P, exact.share, measured[2].share));
    } else {
        printf("(no exact chain: %s)\n\n", exactNote.c_str());
    }

    // tables
    filesystem::create_directories(outDir);
    string stem = outDir + "/idle_" + to_string(bins) + "bins_load" + fmtSetting(load) + "_" + to_string(balls) +
                  (balls == 1 ? "ball_" : "balls_");
    vector<pair<string, const Result *>> tables = {{stem + "1choice.csv", &measured[1]}, {stem + "2choices.csv", &measured[2]}};
    if (haveExact) tables.push_back({stem + "2choices_exact.csv", &exact});
    printf("Tables written:\n");
    for (auto &[path, r] : tables) {
        writeCsv(path, *r, lo, hi);
        printf("  %s\n", path.c_str());
    }

    // mixing: distance from the share after k transitions, measured and as a Markov chain would have it
    string mixPath = stem + "mixing.csv";
    ofstream mix(mixPath);
    if (!mix) throw runtime_error("cannot write " + mixPath);
    mix << "transitions k,1 choice measured,1 choice if Markov,2 choices measured,2 choices if Markov,"
           "1 choice adjacent EMD measured,1 choice adjacent EMD if Markov,"
           "2 choices adjacent EMD measured,2 choices adjacent EMD if Markov\n";
    char buf[160];
    for (size_t l = 0; l < measured[1].lags.size(); l++) {
        snprintf(buf, sizeof buf, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", measured[1].distance[l], modelDistance[1][l],
                 measured[2].distance[l], modelDistance[2][l], measured[1].adjacent[l], modelAdjacent[1][l],
                 measured[2].adjacent[l], modelAdjacent[2][l]);
        mix << measured[1].lags[l] << "," << buf << "\n";
    }
    printf("  %s\n", mixPath.c_str());
    return 0;
}

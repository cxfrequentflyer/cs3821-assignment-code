#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "EventEntity.hpp"
#include "Source.hpp"
#include "Bin.hpp"
#include "Hash.hpp"

using namespace std;


// What "lowest load" means when a ball picks among its D bins.
enum class LoadMetric {
    QueueLength,    // balls in the bin, including the one being processed
    Backlog,        // time the bin needs to finish all its balls (Bin::backlog)
};

class Simulator {
private:
    // A task in the priority queue. The time is fixed when the task is
    // scheduled, so the heap order can never change under it.
    struct Event {
        double time;
        int rank;           // at equal times, bins (0) finish before the source (1) emits
        uint64_t seq;       // then first scheduled, first run, so runs are deterministic
        shared_ptr<EventEntity> entity;
    };

    static bool runsLater(const Event &a, const Event &b) {
        return tie(a.time, a.rank, a.seq) > tie(b.time, b.rank, b.seq);
    }

    shared_ptr<Source> source;
    vector<shared_ptr<Bin>> bins;           // bins taking new balls; balls are hashed into these
    vector<shared_ptr<Bin>> retiring;       // retired bins still processing their balls
    int Dchoice = 1;
    LoadMetric metric;
    bool distinctChoices;
    uint64_t hashSeed;

    double time = 0.0;
    uint64_t nextSeq = 0;
    int nextBinId = 0;
    vector<size_t> order;                   // a permutation of bin indices, for distinct choices

    // Queueing delays of balls processed by bins that have since been freed.
    uint64_t freedProcessed = 0;
    double freedWait = 0.0;

    // A priority queue of tasks; the earliest one is on top.
    priority_queue<Event, vector<Event>, bool (*)(const Event &, const Event &)>
        events{runsLater};

    void schedule(const shared_ptr<EventEntity> &entity);
    void step();
    void place(const Ball &ball);
    size_t chooseBin(const Ball &ball);
    double load(const Bin &bin) const;
    void freeIfRetired(const EventEntity *entity);
    void recordFreed(const Bin &bin);

public:
    // Dchoice          how many bins each ball checks before joining the least loaded.
    // seed             seeds the hash that maps (ball, time) to bins.
    // metric           what counts as load (see LoadMetric).
    // distinctChoices  if true the D bins checked are all different (D is capped
    //                  at the number of bins); if false they are D independent
    //                  hashes and may repeat.
    Simulator(shared_ptr<Source> source, int Dchoice, uint64_t seed = 0,
              LoadMetric metric = LoadMetric::Backlog, bool distinctChoices = true);

    // Called after every ball is placed, with the bin it joined and the ball as
    // stored there (its start time gives its wait). Optional.
    function<void(const Bin &, const Ball &)> onPlace;

    // Add a bin at the current time. Returns its id.
    int addBin(double efficiency = 1.0);

    // Stop sending balls to a bin at the current time. It finishes the balls
    // it already holds, then it is freed.
    void retireBin(int id);

    // Run every task due at or before endTime, then move the clock to endTime.
    void runUntil(double endTime);

    double now() const { return this->time; }

    uint64_t ballsGenerated() const { return this->source->ballsEmitted(); }
    uint64_t ballsProcessed() const;
    uint64_t ballsInSystem() const;
    double totalQueueingDelay() const;

    // Mean time processed balls spent waiting in a queue (not counting their
    // own processing time). NaN until a ball has been processed.
    double averageQueueingDelay() const;

    const vector<shared_ptr<Bin>> &activeBins() const { return this->bins; }
    const vector<shared_ptr<Bin>> &retiringBins() const { return this->retiring; }
};


inline Simulator::Simulator(shared_ptr<Source> source, int Dchoice, uint64_t seed,
                            LoadMetric metric, bool distinctChoices)
    : source(std::move(source)), Dchoice(Dchoice), metric(metric),
      distinctChoices(distinctChoices), hashSeed(hashCombine(seed, 3)) {
    if (!this->source) {
        throw invalid_argument("Simulator needs a source");
    }
    if (Dchoice < 1) {
        throw invalid_argument("Dchoice must be at least 1");
    }
    this->schedule(this->source);
}

inline void Simulator::schedule(const shared_ptr<EventEntity> &entity) {
    int rank = entity == this->source ? 1 : 0;
    this->events.push(Event{entity->nextActionTime(), rank, this->nextSeq++, entity});
}

inline int Simulator::addBin(double efficiency) {
    this->bins.push_back(make_shared<Bin>(this->nextBinId, efficiency));
    this->order.resize(this->bins.size());
    iota(this->order.begin(), this->order.end(), 0);
    return this->nextBinId++;
}

inline void Simulator::retireBin(int id) {
    auto it = find_if(this->bins.begin(), this->bins.end(),
                      [id](const shared_ptr<Bin> &bin) { return bin->id == id; });
    if (it == this->bins.end()) {
        throw invalid_argument("No active bin with id " + to_string(id));
    }
    shared_ptr<Bin> bin = *it;
    this->bins.erase(it);
    this->order.resize(this->bins.size());
    iota(this->order.begin(), this->order.end(), 0);

    bin->retire();
    if (bin->idle()) {
        this->recordFreed(*bin);        // nothing left to process: freed right away
    } else {
        this->retiring.push_back(bin);  // its pending task keeps it running
    }
}

inline void Simulator::runUntil(double endTime) {
    if (endTime < this->time) {
        throw invalid_argument("Cannot run backwards in time");
    }
    while (!this->events.empty() && this->events.top().time <= endTime) {
        this->step();
    }
    this->time = endTime;
}

// Run the earliest task. Every entity has at most one task queued: the source
// always, and a bin while it holds balls.
inline void Simulator::step() {
    Event event = this->events.top();
    this->events.pop();
    this->time = event.time;

    optional<Ball> ball = event.entity->processEvent(this->time);
    if (ball) {
        this->place(*ball);
    }

    if (event.entity->nextActionTime() != EventEntity::NEVER) {
        this->schedule(event.entity);
    } else {
        this->freeIfRetired(event.entity.get());
    }
}

inline void Simulator::place(const Ball &ball) {
    if (this->bins.empty()) {
        throw runtime_error("A ball arrived at time " + to_string(this->time) +
                            " but there are no active bins");
    }
    shared_ptr<Bin> &bin = this->bins[this->chooseBin(ball)];
    bool wasIdle = bin->idle();
    const Ball &placed = bin->addBall(ball, this->time);
    if (wasIdle) {
        this->schedule(bin);            // its next task: this ball finishing
    }
    if (this->onPlace) {
        this->onPlace(*bin, placed);
    }
}

// Hash the ball and the current time into Dchoice bins and return the index
// of the least loaded one. Ties go to the bin checked first; the choices are
// exchangeable, so that is a uniformly random tied bin.
inline size_t Simulator::chooseBin(const Ball &ball) {
    uint64_t h = hashCombine(this->hashSeed, ball.id);
    h = hashCombine(h, doubleBits(ball.size));
    h = hashCombine(h, doubleBits(this->time));

    size_t n = this->bins.size();
    size_t d = this->distinctChoices ? min<size_t>(this->Dchoice, n) : this->Dchoice;
    size_t best = 0;
    double bestLoad = EventEntity::NEVER;
    for (size_t k = 0; k < d; k++) {
        uint64_t hk = mix64(h + k);     // the k-th hash of this ball
        size_t i;
        if (this->distinctChoices) {
            // one step of a partial Fisher-Yates shuffle: order[0..d) ends up
            // a uniformly random sample without replacement, whatever order
            // was left in by earlier balls
            size_t j = k + hk % (n - k);
            swap(this->order[k], this->order[j]);
            i = this->order[k];
        } else {
            i = hk % n;
        }
        double l = this->load(*this->bins[i]);
        if (l < bestLoad) {
            bestLoad = l;
            best = i;
        }
    }
    return best;
}

inline double Simulator::load(const Bin &bin) const {
    if (this->metric == LoadMetric::QueueLength) {
        return static_cast<double>(bin.queueLength());
    }
    return bin.backlog(this->time);
}

// Called when an entity runs out of work. If it is a retired bin, drop it: the
// task being run holds the last reference, so the bin is freed once step()
// returns.
inline void Simulator::freeIfRetired(const EventEntity *entity) {
    auto it = find_if(this->retiring.begin(), this->retiring.end(),
                      [entity](const shared_ptr<Bin> &bin) { return bin.get() == entity; });
    if (it != this->retiring.end()) {
        this->recordFreed(**it);
        this->retiring.erase(it);
    }
}

inline void Simulator::recordFreed(const Bin &bin) {
    this->freedProcessed += bin.ballsProcessed();
    this->freedWait += bin.totalQueueingDelay();
}

inline uint64_t Simulator::ballsProcessed() const {
    uint64_t total = this->freedProcessed;
    for (const auto &bin : this->bins) total += bin->ballsProcessed();
    for (const auto &bin : this->retiring) total += bin->ballsProcessed();
    return total;
}

inline uint64_t Simulator::ballsInSystem() const {
    uint64_t total = 0;
    for (const auto &bin : this->bins) total += bin->queueLength();
    for (const auto &bin : this->retiring) total += bin->queueLength();
    return total;
}

inline double Simulator::totalQueueingDelay() const {
    double total = this->freedWait;
    for (const auto &bin : this->bins) total += bin->totalQueueingDelay();
    for (const auto &bin : this->retiring) total += bin->totalQueueingDelay();
    return total;
}

inline double Simulator::averageQueueingDelay() const {
    uint64_t processed = this->ballsProcessed();
    if (processed == 0) {
        return nan("");
    }
    return this->totalQueueingDelay() / static_cast<double>(processed);
}

#endif

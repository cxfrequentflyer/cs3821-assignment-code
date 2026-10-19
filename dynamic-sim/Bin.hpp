#ifndef BIN_HPP
#define BIN_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include "EventEntity.hpp"

using namespace std;

// A FIFO queue that processes one ball at a time. A ball of size s takes
// s / efficiency units of time, so at efficiency 1 an average (size 1) ball
// takes one unit of time.
//
// The bin never idles while it holds balls and its efficiency never changes,
// so a ball's start and finish times are already known when it joins: it
// starts when the ball ahead of it finishes, or straight away if the bin is
// empty. The bin's next action is always its front ball finishing.
class Bin : public EventEntity {
private:
    queue<Ball> balls;
    double efficiency = 1.0;
    double drainTime = 0.0;     // when the last ball in the queue finishes
    bool isRetired = false;

    // Processed balls are deleted; only their queueing delays are kept.
    uint64_t processed = 0;
    double totalWait = 0.0;

public:
    const int id;

    Bin(int id, double efficiency);

    double nextActionTime() const override;

    optional<Ball> processEvent(double now) override;

    void addBall(Ball ball, double now);

    // A retired bin takes no new balls but finishes the ones it holds.
    void retire() { this->isRetired = true; }

    bool retired() const { return this->isRetired; }
    bool idle() const { return this->balls.empty(); }
    double getEfficiency() const { return this->efficiency; }

    // Balls in the bin, including the one being processed.
    size_t queueLength() const { return this->balls.size(); }

    // Time the bin needs to finish every ball it holds: the queueing delay a
    // ball joining now would see.
    double backlog(double now) const { return max(0.0, this->drainTime - now); }

    uint64_t ballsProcessed() const { return this->processed; }
    double totalQueueingDelay() const { return this->totalWait; }
};




inline Bin::Bin(int id, double efficiency) : EventEntity(), efficiency(efficiency), id(id) {
    if (!(efficiency > 0.0) || !isfinite(efficiency)) {
        throw invalid_argument("Efficiency must be greater than 0");
    }
}

inline double Bin::nextActionTime() const {
    if (this->balls.empty()) {
        return NEVER;
    }
    return this->balls.front().finishTime;
}

inline void Bin::addBall(Ball ball, double now) {
    if (this->isRetired) {
        throw logic_error("A retired bin cannot take new balls");
    }
    ball.startTime = max(now, this->drainTime);
    ball.finishTime = ball.startTime + ball.size / this->efficiency;
    this->drainTime = ball.finishTime;
    this->balls.push(ball);
}

inline optional<Ball> Bin::processEvent(double now) {
    if (this->balls.empty()) {
        return nullopt;
    }

    // the front ball is done: record how long it waited, then delete it
    const Ball &ball = this->balls.front();
    assert(ball.finishTime <= now);
    (void)now;
    this->processed++;
    this->totalWait += ball.queueingDelay();
    this->balls.pop();
    return nullopt;
}
#endif

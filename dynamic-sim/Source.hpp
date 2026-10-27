#ifndef SOURCE_HPP
#define SOURCE_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include "EventEntity.hpp"
#include "Ball.hpp"
#include "Distribution.hpp"
#include "Hash.hpp"

using namespace std;

// The single source of balls. Balls arrive from time 0 onwards; the gaps
// between them are either constant or exponential (a Poisson process), and
// their sizes are log-normal.
class Source : public EventEntity {
private:

    unique_ptr<Distribution> timeDist;
    unique_ptr<Distribution> sizeDist;
    double nextBallTime = 0.0;
    uint64_t emitted = 0;
    double constantGap = 0.0;   // the gap, when gaps are constant

public:

    // meanGap  mean time between balls.
    // poisson  exponential gaps if true, otherwise every gap is exactly meanGap.
    // sigma    shape of the log-normal sizes. mu is set to -sigma^2 / 2 so the
    //          mean size is 1: at efficiency 1 an average ball takes one unit
    //          of time. sigma = 0 makes every ball size 1.
    // seed     the gap and size streams are derived from it independently.
    Source(double meanGap, bool poisson, double sigma, uint64_t seed = 0) : EventEntity() {
        if (meanGap <= 0.0) {
            throw invalid_argument("Mean gap must be greater than 0");
        }
        if (sigma < 0.0) {
            throw invalid_argument("Sigma must be non-negative");
        }
        if (poisson) {
            this->timeDist = make_unique<ExponentialDistribution>(1.0 / meanGap, hashCombine(seed, 1));
        } else {
            this->timeDist = make_unique<ConstantDistribution>(meanGap);
            this->constantGap = meanGap;
        }
        if (sigma > 0.0) {
            this->sizeDist = make_unique<LognormalDistribution>(-0.5 * sigma * sigma, sigma, hashCombine(seed, 2));
        } else {
            this->sizeDist = make_unique<ConstantDistribution>(1.0);
        }
    };

    double nextActionTime() const override {
        return this->nextBallTime;
    }

    optional<Ball> processEvent(double now) override;

    uint64_t ballsEmitted() const { return this->emitted; }
};


inline optional<Ball> Source::processEvent(double now) {
    Ball ball(this->emitted++, this->sizeDist->sample(), now);
    // With constant gaps ball k arrives at exactly k * gap; adding up the gaps
    // instead would let rounding build up over millions of balls.
    this->nextBallTime = this->constantGap > 0.0 ? static_cast<double>(this->emitted) * this->constantGap
                                                 : now + this->timeDist->sample();
    return ball;
}

#endif

#ifndef SOURCE_HPP
#define SOURCE_HPP

#include <stdexcept>
#include "EventEntity.hpp"
#include "Ball.hpp"
#include "Distribution.hpp"
#include <queue>
#include <memory>
#include <iostream>
#include "Simulator.hpp"

using namespace std;

class Source : public EventEntity {
private:

    unique_ptr<Distribution> timeDist;
    unique_ptr<Distribution> sizeDist;
    hash<double> hashFn = hash<double>();
    double nextBallTime = 0.0;

public:

    Source(double mean, bool poisson = true) : EventEntity() {
        if (poisson) {
            this->timeDist = make_unique<ExponentialDistribution>(1.0 / mean);
        } else {
            this->timeDist = make_unique<ConstantDistribution>(mean);
        }

    };

    double nextActionTime() const override;

    optional<Ball> processEvent(double &time) override;
};




#endif

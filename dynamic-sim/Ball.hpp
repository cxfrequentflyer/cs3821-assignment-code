#ifndef BALL_HPP
#define BALL_HPP

#include <cstdint>
#include <stdexcept>

// A ball in the balls-into-bins simulation. Its id, size and arrival time are
// fixed when it is created; the bin it joins fills in when it starts and
// finishes being processed.
class Ball {
public:
    Ball(uint64_t id, double size, double arrivalTime)
        : id(id), size(size), arrivalTime(arrivalTime) {
        if (size < 0.0) {
            throw std::invalid_argument("Ball size must be non-negative");
        }
    }

    uint64_t id;
    double size;
    double arrivalTime;
    double startTime = 0.0;
    double finishTime = 0.0;

    // Time spent waiting in the queue, excluding processing time.
    double queueingDelay() const { return startTime - arrivalTime; }
};

#endif

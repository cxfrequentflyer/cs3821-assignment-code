#ifndef BALL_HPP
#define BALL_HPP

#include <stdexcept>

// A ball in the balls-into-bins simulation. Its size and spawn time are
// fixed when it is created.
class Ball {
public:
    explicit Ball(double nextActionTime, double size = 1.0)
        : size(size), nextActionTime(nextActionTime) {
        if (size < 0.0) {
            this->size = 1.0;
        }
    }

    double size;
    double nextActionTime;
};

#endif

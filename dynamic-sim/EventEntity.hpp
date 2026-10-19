#ifndef EVENTENTITY_HPP
#define EVENTENTITY_HPP

#include <limits>
#include <optional>
#include "Ball.hpp"

using namespace std;

// Something the simulator runs as a task: the source, or a bin.
class EventEntity {
public:
    // nextActionTime() of an entity with nothing to do.
    static constexpr double NEVER = numeric_limits<double>::infinity();

    virtual ~EventEntity() = default;

    // When this entity next needs to act, or NEVER if it is idle.
    virtual double nextActionTime() const = 0;

    // Carry out the action due at time `now`. Returns the ball this action
    // releases into the system, if any: the source returns each new ball so
    // the simulator can place it, a bin returns nothing (it deletes the ball
    // it has finished).
    virtual optional<Ball> processEvent(double now) = 0;
};


#endif

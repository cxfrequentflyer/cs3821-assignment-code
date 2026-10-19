#ifndef EVENTENTITY_HPP
#define EVENTENTITY_HPP

#include <stdexcept>
#include "Ball.hpp"
#include <memory>
#include <optional>

using namespace std;

class EventEntity {
public:
    virtual ~EventEntity() = default;
    virtual double nextActionTime() const;

    // given the current time, process the event and return the new time after processing
    virtual optional<Ball> processEvent(double &time);

    // Priority queue comparator: the entity with the earliest action time ends
    // up on top (min-heap).
    static bool actsLater(const shared_ptr<EventEntity> a, const shared_ptr<EventEntity> b) {
        return a->nextActionTime() > b->nextActionTime();
    }
};


#endif
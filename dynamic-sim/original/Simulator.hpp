#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <memory>
#include <queue>
#include <vector>
#include "EventEntity.hpp"
#include "Source.hpp"
#include "Bin.hpp"

using namespace std;



class Simulator {
private:
    shared_ptr<Source> source;
    vector<shared_ptr<Bin>> bins;
    int Dchoice = 1;
    // A priority queue of events, sorted by their action time.
    priority_queue<shared_ptr<EventEntity>, vector<shared_ptr<EventEntity>>,
        bool (*)(const shared_ptr<EventEntity>, const shared_ptr<EventEntity>)>
        events{EventEntity::actsLater};

public:

};

#endif

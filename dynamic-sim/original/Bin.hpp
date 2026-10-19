#ifndef BIN_HPP
#define BIN_HPP

#include <stdexcept>
#include "EventEntity.hpp"
#include <queue>

using namespace std;

class Bin : public EventEntity {
private:
    queue<Ball> balls  = queue<Ball>();;
    double efficiency = 1.0;

public:
    bool isProcessing = false;
    bool retired = false;
    Bin(double efficiency);
    double nextActionTime() const override;

    optional<Ball> processEvent(double &time) override;

    void addBall(Ball ball, double time) {
        this->balls.push(ball);
        this->isProcessing = true;

        // edge case
        if (this->balls.size() == 1) {
            this->balls.front().nextActionTime = time + ball.size / this->efficiency;
        }
    }
};




Bin::Bin(double efficiency) : EventEntity(), efficiency(efficiency) {
    if (efficiency <= 0.0) {
        throw invalid_argument("Efficiency must be greater than 0");
    }
}

double Bin::nextActionTime() const {
    if (this->balls.empty()) {
        return -1.0;
    }
    return this->balls.front().nextActionTime;
}

optional<Ball> Bin::processEvent(double &time) {
    if (this->balls.empty()) {
        return nullopt;
    }

    // remove the ball from the bin and process it
    Ball ball = this->balls.front();
    this->balls.pop();
    time += ball.size / this->efficiency;
    delete &ball;


    // update the next ball's action time
    if (this->balls.empty()) {
        this->isProcessing = false;
        return nullopt;
    }


    Ball nextBall = this->balls.front();
    nextBall.nextActionTime = time + nextBall.size / this->efficiency;
    return nullopt;
}
#endif
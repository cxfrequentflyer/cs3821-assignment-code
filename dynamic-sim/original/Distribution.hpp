#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <random>
#include <memory>


class Distribution {
    public:
        virtual ~Distribution() = default;
        virtual double sample() = 0;
};

class ExponentialDistribution : public Distribution {
    private:
        std::default_random_engine generator;
        std::exponential_distribution<double> distribution;

    public:
        ExponentialDistribution(double lambda) : distribution(lambda) {}

        double sample() override {
            return distribution(generator);
        }
};

class ConstantDistribution : public Distribution {
    private:
        double value;

    public:
        ConstantDistribution(double value) : value(value) {}

        double sample() override {
            return value;
        }
};

class LognormalDistribution : public Distribution {
    private:
        std::default_random_engine generator;
        std::lognormal_distribution<double> distribution;

    public:
        LognormalDistribution(double mean, double stddev) : distribution(mean, stddev) {}

        double sample() override {
            return distribution(generator);
        }
};

#endif
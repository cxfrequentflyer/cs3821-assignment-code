#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <cstdint>
#include <random>
#include <memory>


// Each random distribution owns its own generator, seeded explicitly, so two
// streams (e.g. ball gaps and ball sizes) never share random numbers.
class Distribution {
    public:
        virtual ~Distribution() = default;
        virtual double sample() = 0;
};

class ExponentialDistribution : public Distribution {
    private:
        std::mt19937_64 generator;
        std::exponential_distribution<double> distribution;

    public:
        ExponentialDistribution(double lambda, uint64_t seed)
            : generator(seed), distribution(lambda) {}

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

// exp(N(mu, sigma^2)). Note mu and sigma belong to the underlying normal: the
// mean of the samples themselves is exp(mu + sigma^2 / 2).
class LognormalDistribution : public Distribution {
    private:
        std::mt19937_64 generator;
        std::lognormal_distribution<double> distribution;

    public:
        LognormalDistribution(double mu, double sigma, uint64_t seed)
            : generator(seed), distribution(mu, sigma) {}

        double sample() override {
            return distribution(generator);
        }
};

#endif

#ifndef NODE_HPP
#define NODE_HPP

#include <array>
#include <cstdint>
#include <map>
#include <vector>

using namespace std;

// A configuration of bins: one hex digit per bin holding its number of balls,
// sorted so equal configurations share a code. The fullest bin is the most
// significant digit and empty bins are the trailing zeros, e.g. 0x31100.
class Node {
public:
    inline static int numBins = 5;      // at most 8
    inline static int binCap = 15;      // at most 15
    inline static int choices = 2;      // bins each ball checks
    inline static bool distinct = true; // checked bins are all different

    uint32_t state;
    vector<pair<uint32_t, uint64_t>> nextStates;   // (next configuration, frequency), sorted by configuration
    uint64_t total = 0;                            // sum of the frequencies: divide by it for probabilities

    explicit Node(uint32_t s) : state(encode(counts(s))) {}

    // Fills nextStates with the configurations one turn later, and how many ordered
    // choice sequences reach each one. In a turn, `balls` balls (default n - 1)
    // arrive one at a time, each joining the least loaded of its checked bins (a
    // ball finding its bin full is turned away), then every non-empty bin
    // processes one ball. Frequencies sum to orderedChoices(numBins)^balls.
    void findNextStates(int balls = -1) {
        map<uint32_t, uint64_t> table = {{state, 1}};
        for (int b = 0; b < (balls < 0 ? numBins - 1 : balls); ++b) {
            map<uint32_t, uint64_t> next;
            for (auto [s, freq] : table) {
                array<int, 16> c = counts(s);
                for (int k = 0, atLeast = numBins; k <= binCap; atLeast -= c[k++]) {
                    // ways the least loaded checked bin has exactly k balls
                    uint64_t ways = orderedChoices(atLeast) - orderedChoices(atLeast - c[k]);
                    if (!ways) continue;
                    array<int, 16> after = c;
                    if (k < binCap) after[k]--, after[k + 1]++;
                    next[encode(after)] += freq * ways;
                }
            }
            table = std::move(next);
        }
        // then every non-empty bin processes one ball: a bin with k balls ends with k - 1
        map<uint32_t, uint64_t> processed;
        for (auto [s, freq] : table) {
            array<int, 16> c = counts(s), after{};
            after[0] = c[0];
            for (int k = 1; k < 16; ++k) after[k - 1] += c[k];
            processed[encode(after)] += freq;
        }
        nextStates.assign(processed.begin(), processed.end());
        total = 0;
        for (auto [s, freq] : nextStates) total += freq;
    }

    // Ordered ways to check `choices` of `bins` bins.
    static uint64_t orderedChoices(int bins) {
        uint64_t ways = 1;
        for (int i = 0; i < choices; ++i) ways *= max(0, distinct ? bins - i : bins);
        return ways;
    }

private:
    // c[k] = bins holding k balls
    static array<int, 16> counts(uint32_t s) {
        array<int, 16> c{};
        c[0] = numBins;
        for (; s; s >>= 4)
            if (s & 0xF) c[s & 0xF]++, c[0]--;
        return c;
    }

    // fullest bins first, so they end up in the most significant digits
    static uint32_t encode(const array<int, 16> &c) {
        uint32_t s = 0;
        for (int k = 15; k >= 0; --k)
            for (int i = 0; i < c[k]; ++i) s = s << 4 | k;
        return s;
    }
};

#endif

#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include "node.hpp"

using namespace std;

class Solver {
public:
    vector<Node> nodes;
    unordered_map<uint32_t, size_t> stateToIndex;

    // Finds every configuration reachable from `state`, depth first.
    void solve(uint32_t state) {
        state = Node(state).state;   // sorted, as stored
        if (!stateToIndex.count(state)) addNode(state);
    }

    // Long-run probability of each node, by power iteration on the lazy chain
    // (I + P) / 2, which has the same answer and is never periodic.
    vector<double> stationary(double tolerance = 1e-13) const {
        vector<double> pi(nodes.size(), 1.0 / nodes.size()), next(nodes.size());
        for (double change = 1; change > tolerance;) {
            for (size_t i = 0; i < nodes.size(); ++i) next[i] = pi[i] / 2;
            for (size_t i = 0; i < nodes.size(); ++i)
                for (auto [s, freq] : nodes[i].nextStates)
                    next[stateToIndex.at(s)] += pi[i] / 2 * double(freq) / nodes[i].total;
            change = 0;
            for (size_t i = 0; i < nodes.size(); ++i) change += abs(next[i] - pi[i]);
            swap(pi, next);
        }
        return pi;
    }

    // Saves as a CSV where each row is a node: first the state, then pairs of
    // next state and the probability of that transition. States are written in
    // hexadecimal with one digit per bin, e.g. 00113.
    void save(const string &path = "chain.csv") const {
        ofstream out(path);
        auto hexState = [&](uint32_t s) -> ostream & {
            return out << hex << setw(Node::numBins) << setfill('0') << s << dec;
        };
        for (const Node &node : nodes) {
            hexState(node.state);
            for (auto [next, freq] : node.nextStates) {
                out << ',';
                hexState(next) << ',' << double(freq) / node.total;
            }
            out << '\n';
        }
    }

private:
    void addNode(uint32_t state) {
        size_t i = nodes.size();
        stateToIndex[state] = i;
        nodes.emplace_back(state);
        nodes[i].findNextStates();
        // copied: solving the next states grows `nodes`, which can move nodes[i]
        for (auto [next, freq] : vector(nodes[i].nextStates)) solve(next);
    }
};

#endif

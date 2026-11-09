// Builds the Markov chain from empty bins for each rule and saves it.
//   ./build/markov [bins] [cap]          defaults: 5 bins, cap 15
#include <cstdio>
#include <cstdlib>
#include "solver.hpp"

int main(int argc, char **argv) {
    Node::numBins = argc > 1 ? atoi(argv[1]) : 5;
    Node::binCap = argc > 2 ? atoi(argv[2]) : 15;
    for (int d : {1, 2}) {
        Node::choices = d;
        Solver solver;
        solver.solve(0);
        string path = "results/chain_" + to_string(Node::numBins) + "bins_cap" + to_string(Node::binCap) + "_" +
                      to_string(d) + (d == 1 ? "choice.csv" : "choices.csv");
        solver.save(path);
        printf("%d choice%s: %zu states -> %s\n", d, d == 1 ? " " : "s", solver.nodes.size(), path.c_str());
    }
}

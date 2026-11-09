CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic

build/markov: main.cpp node.hpp solver.hpp | build
	$(CXX) $(CXXFLAGS) -o $@ main.cpp

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: clean

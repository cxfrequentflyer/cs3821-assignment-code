CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
HEADERS := $(wildcard *.hpp)

all: build/dynsim build/tests transition-matrix

build/dynsim: main.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -o $@ main.cpp

transition-matrix:
	$(MAKE) -C transition-matrix

build/tests: tests.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -o $@ tests.cpp

build:
	mkdir -p build

test: build/tests
	./build/tests

clean:
	rm -rf build
	$(MAKE) -C transition-matrix clean

.PHONY: all test clean transition-matrix

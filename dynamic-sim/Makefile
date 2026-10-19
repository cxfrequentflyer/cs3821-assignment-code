CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
HEADERS := $(wildcard *.hpp)

all: build/dynsim build/tests

build/dynsim: main.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -o $@ main.cpp

build/tests: tests.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -o $@ tests.cpp

build:
	mkdir -p build

test: build/tests
	./build/tests

clean:
	rm -rf build

.PHONY: all test clean

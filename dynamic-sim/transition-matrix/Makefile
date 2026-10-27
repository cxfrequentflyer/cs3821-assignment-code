CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
HEADERS := $(wildcard ../*.hpp)

all: build/transition_matrix

build/transition_matrix: transition_matrix.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -I.. -o $@ transition_matrix.cpp

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean

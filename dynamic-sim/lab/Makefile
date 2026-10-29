CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
HEADERS := $(wildcard ../*.hpp)

all: build/lab

build/lab: lab.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -I.. -o $@ lab.cpp

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean

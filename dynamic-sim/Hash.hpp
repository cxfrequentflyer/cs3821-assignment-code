#ifndef HASH_HPP
#define HASH_HPP

#include <cstdint>
#include <cstring>

// splitmix64's mixing step: a fast, well-tested 64-bit bit mixer. Used both to
// hash balls into bins and to derive independent seeds for random streams.
inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline uint64_t hashCombine(uint64_t h, uint64_t value) {
    return mix64(h ^ mix64(value));
}

inline uint64_t doubleBits(double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return bits;
}

#endif

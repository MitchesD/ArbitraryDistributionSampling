/**
 * Copyright (C) 2023 by Michal Vlnas <ivlnas@fit.vutbr.cz>
 *
 * Permission to use, copy, modify, and/or distribute this
 * software for any purpose with or without fee is hereby granted.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
 * THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
 * CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
 * OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef IMPORTANCE_LCG_H__
#define IMPORTANCE_LCG_H__

#include <shared/config.h>
#include <glm/glm.hpp>

#define INV_UINT63_MAX 1.0842e-19

namespace Importance
{

template<uint64_t A, uint64_t C, uint64_t M>
class LinearCongruentialGenerator
{
public:
    static constexpr uint64_t multiplier   = A;
    static constexpr uint64_t increment    = C;
    static constexpr uint64_t modulus      = M;
    static constexpr uint64_t mask         = M - 1ULL;
    static constexpr uint64_t default_seed = 1ULL;

    explicit LinearCongruentialGenerator(uint64_t seed_ = default_seed) : seed_backup(seed_), seed(seed_) { }

    LinearCongruentialGenerator(const LinearCongruentialGenerator& rhs)
    {
        seed = rhs.seed;
        seed_backup = rhs.seed_backup;
    }

    FORCE_INLINE Float next1D()
    {
        seed = (multiplier * seed + increment) & mask;
        return seed * INV_UINT63_MAX;
    }

    FORCE_INLINE uint64_t next1DInt()
    {
        seed = (multiplier * seed + increment) & mask;
        return seed;
    }

    FORCE_INLINE void skip1D()
    {
        seed = (multiplier * seed + increment) & mask;
    }

    glm::vec2 next2D()
    {
        return { next1D(), next1D() };
    }

    std::tuple<uint64_t, uint64_t> next2DInt()
    {
        return { next1DInt(), next1DInt() };
    }

    void skip2D()
    {
        skip1D();
        skip1D();
    }

    void skipN(uint64_t ns)
    {
        if (ns == 0)
            return;

        int64_t nskip = ns & mask;

        // The algorithm here to determine the parameters used to skip ahead is
        // described in the paper F. Brown, "Random Number Generation with Arbitrary Stride,"
        // Trans. Am. Nucl. Soc. (Nov. 1994). This algorithm is able to skip ahead in
        // O(log2(N)) operations instead of O(N). It computes parameters
        // M and C which can then be used to find x_N = M*x_0 + C mod 2^M.

        // initialize constants
        uint64_t m = multiplier; // original multiplicative constant
        uint64_t c = increment;  // original additive constant

        uint64_t m_next = 1ULL; // new effective multiplicative constant
        uint64_t c_next = 0ULL; // new effective additive constant

        while (nskip > 0LL)
        {
            if (nskip & 1LL) // check least significant bit for being 1
            {
                m_next = (m_next * m) & mask;
                c_next = (c_next * m + c) & mask;
            }

            c = ((m + 1ULL) * c) & mask;
            m = (m * m) & mask;

            nskip = nskip >> 1LL; // shift right, dropping least significant bit
        }

        // with M and C, we can now find the new seed
        seed = (m_next * seed + c_next) & mask;
    }

    void reset()
    {
        seed = seed_backup;
    }

    void print_state()
    {
        std::cout << std::to_string(seed) << std::endl;
    }

    std::string get_state_string() const
    {
        return std::to_string(seed);
    }

private:
    uint64_t seed_backup;
    uint64_t seed;
};

typedef LinearCongruentialGenerator<2806196910506780709ULL, 1ULL, 1ULL << 63UL> FastLCG;

} // namespace Importance

#endif // IMPORTANCE_LCG_H__

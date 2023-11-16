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

#ifndef IMPORTANCE_CONFIG_H__
#define IMPORTANCE_CONFIG_H__

#include <iostream>

namespace Importance
{

#define F_PI    3.14159265358979323846f
#define F_PI_2  1.570796327f
#define F_E		2.7182818284590452354f

#ifdef IMPORTANCE_DOUBLE_PRECISION
    typedef double Float;
#else
    typedef float Float;
#endif

#define FORCE_INLINE __attribute__((always_inline))

#ifndef NDEBUG
#define ASSERT_MESSAGE(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#define ASSERT_MESSAGE(condition, message) do { } while (false)
#endif

#ifndef NDEBUG
#define ASSERT(condition) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#define ASSERT(condition) do { } while (false)
#endif

enum GridHeuristics
{
    G_HEURISTIC_AREA = 1,
    G_HEURISTIC_MAX  = 2,
};

namespace Constants
{

// number of samples per grid interval, to estimate the integral and local maxima
uint32_t const intervalSamples = 100;
// static vector size in grid partitioning
std::size_t const defaultVectorSize = 2048;

}

}

#endif // IMPORTANCE_CONFIG_H__

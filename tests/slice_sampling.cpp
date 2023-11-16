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

#include <fstream>
#include <cmath>
#include <random>
#include <shared/config.h>

using namespace Importance;

// normal distribution PDF
Float pdf(Float x)
{
    return std::exp(-(x * x) / 2.0f) / (std::sqrt(2.0f * F_PI));
}

int main()
{
    int samples = 500;
    std::ofstream str("slice_normal_distr.data");

    std::random_device rd;
    std::mt19937 gen(rd());

    // initial 'x'
    Float x = 0.0f;
    for (int i = 0; i < samples; i++)
    {
        // generate 'y' sample using 'x'
        std::uniform_real_distribution<float> d(0.0f, pdf(x));
        Float y = d(gen);

        // now we have sliced function on 'y' coordinate
        Float alpha = std::sqrt(-2.0f * std::log(y * std::sqrt(2.0f * F_PI)));

        // sample from the slice
        std::uniform_real_distribution<Float> slice_distribution(-alpha, alpha);
        x = slice_distribution(gen);
        str << x << ", " << pdf(x) << std::endl;
    }

    str.close();

    return 0;
}
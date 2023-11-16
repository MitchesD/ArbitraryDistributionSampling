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

/**
 * Ratio of uniforms applied on No(0, 1)
 * @return
 */
int main()
{
    int samples = 500;
    std::ofstream str("ratio_normal_distr.data");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    Float e1 = 1./ 2.71828182845905f;

    for (int i = 0; i < samples; i++)
    {
        while (true)
        {
            Float rn1 = uniform(gen);
            Float rn2 = uniform(gen);
            rn2 = (2.0f * rn2 - 1.0f) * std::sqrt(2.0f * e1);
            if (-4.0f * rn1 * rn1 * std::log(rn1) < rn2 * rn2)
                continue;

            Float rn = rn2 / rn1;
            str << rn << ", " << pdf(rn) << std::endl;
            break;
        }
    }

    str.close();

    return 0;
}
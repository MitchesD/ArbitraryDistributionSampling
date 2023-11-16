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

#include <iostream>
#include <shared/spherical_distributions.h>
#include <generators/xorshift.h>
#include <shared/mapping.h>
#include <fstream>
#include <random>
#include <map>

float k = 6;
Importance::SphericalGaussian sg(k, glm::vec3(0.0, 0, 1));

void test_sample_equidistant(uint32_t samples)
{
    std::ofstream str_square("eq_square.data");
    std::ofstream str_world("eq_world.data");

    float inc = 1.0f / float(samples);
    for (uint32_t i = 0; i < samples; ++i)
    {
        for (uint32_t j = 0; j < samples; j++)
        {
            float a = inc * i;
            float b = inc * j;
            auto vec = sg.sample(glm::vec2(a, b));
            auto square = sg.sampleSquare(glm::vec2(a, b));
            str_square << square.x << ", " << square.y << std::endl;
            str_world << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        }
    }

    str_square.close();
    str_world.close();
}

void test_sample_importance(uint32_t samples)
{
    std::ofstream str_square("imp_square.data");
    std::ofstream str_world("imp_world.data");

    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<> rng(0.0, 1.0);
    for (uint32_t i = 0; i < samples; i++)
    {
        float a = rng(gen);
        float b = rng(gen);
        auto vec = sg.sample(glm::vec2(a, b));
        auto square = sg.sampleSquare(glm::vec2(a, b));
        str_square << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        str_world << square.x << ", " << square.y << std::endl;
    }

    str_square.close();
    str_world.close();
}

void test_shirley_chu_mapping(uint32_t const np)
{
    std::ofstream shirley("shirley.data");
    for (uint32_t i = 0; i < np; i++)
    {
        for (uint32_t j = 0; j < np; j++)
        {
            float px = float(i) / float(np);
            float py = float(j) / float(np);
            auto vec = Importance::uniformHemisphereShirley2(px, py);
            shirley << vec.x << ", " << vec.y << std::endl;
        }
    }

    shirley.close();
}

int main()
{
    test_sample_equidistant(150);
    test_sample_importance(10000);
    test_shirley_chu_mapping(30);

    return 0;
}

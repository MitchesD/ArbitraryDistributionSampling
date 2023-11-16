/**
 * Copyright (C) 2024 by Michal Vlnas <ivlnas@fit.vutbr.cz>
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
#include <is/ibuffer.h>
#include <generators/lcg.h>
#include <map>
#include <shared/api.h>
#include <shared/distributions_2d.h>

using namespace Importance;


int main(int argc, char** argv)
{
    std::string input = "courtyard.png";
    if (argc > 1)
        input = std::string(argv[1]);
    std::string output = input.substr(0, input.find('.')) + "-out.ppm";

    int generate = 20000;
    int samples = 5000;
    glm::vec3 green(0, 1, 0.5f);
    glm::vec3 red(1.0f, 0.2f, 0.0f);
    glm::vec3 blue(0.0f, 0.2f, 1.0f);

    IEnvironmentMap<FastLCG> map(input, glm::vec2(0,0), glm::vec2(1024,768));

    auto buffer = new RSWrappedGridBuffer<EnvironmentMap, FastLCG, glm::vec2, Grid2D<Constants::defaultVectorSize>>(150, 150, map.getMax(), map.getMin());
    auto gen = new FastLCG();

    InputData inputData = { };
    buffer->filterUniversal(generate, &map, gen, inputData);
    gen->reset();

    std::vector<glm::vec2> points;
    std::vector<glm::vec3> colors;
    std::ofstream str_default("envmap.data");
    for (int i = 0; i < samples; i++)
    {
        auto sample = map.sample(buffer, gen);
        points.push_back(sample);
        glm::vec color = blue;
        colors.push_back(color);

        str_default << sample.x << ", " << -sample.y << "\n";
    }
    str_default.close();
    gen->reset();

    map.store(output, points, colors);

    return 0;
}
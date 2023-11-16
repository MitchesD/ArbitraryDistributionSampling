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
#include <shared/test_setup.h>
#include <shared/spherical_distributions.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace Importance;
namespace po = boost::program_options;

struct TestData
{
    uint64_t samples;
    std::string output;
    BufferType bufferType;
    Float error;
    uint32_t phiNodes;
    uint32_t thetaNodes;
};

template<typename TGenerator, typename TDistribution, template<class> class TIDistribution, class... Args>
void test_distribution(TestData const& data, Args... args)
{
    std::stringstream ss;
    ss << __func__ << "<" << TypeName<TGenerator>::Get() << ", " << TypeName<TDistribution>::Get() << ", " <<
       TypeName<TIDistribution<TDistribution>>::Get() << ">";
    TDistribution distribution(args...);
    TestSetupUniversal<TGenerator, TDistribution, glm::vec3, GridSpherical<Constants::defaultVectorSize>> const setup(&distribution,
                                                                               ss.str(), data.output, data.samples,
                                                                               data.bufferType, data.error,
                                                                               data.phiNodes, data.thetaNodes);

    TIDistribution<TGenerator> i_distribution(args...);
    std::ofstream str_out(data.output);

    Float avg = 0.0f;
    Float max = 0.0f;

    for (uint64_t i = 0; i < data.samples; i++)
    {
        glm::vec3 sample = i_distribution.sample(setup.buffer, setup.gen);
        Float const value = i_distribution.f(sample);
        avg += value;

        if (value > max)
            max = value;
        str_out << sample.x << ", " << sample.y << ", " << sample.z << std::endl;
    }

    str_out.close();

    std::cout << "Avg: " << avg / Float(data.samples) << std::endl;
    std::cout << "Max: " << i_distribution.max() << std::endl;
    std::cout << "Mesaured max: " << max << std::endl;
}

int main(int argc, char** argv)
{
    uint64_t samples = 300;
    BufferType bufferType = Importance::BUFFER_RS;
    Float error = 0.05f;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("samples,s", po::value<uint64_t>(&samples)->implicit_value(300), "Number of samples per test")
        ("error,e", po::value<Float>(&error)->implicit_value(0.05f), "Accept samples with error")
        ("advanced,a", "Enables advanced buffer")
        ("box,b", "Enables RS box buffer")
        ("pav,p", "Enables 2nd variant of RS box buffer");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if (vm.count("advanced"))
        bufferType = Importance::BUFFER_RS_ADV;
    else if (vm.count("box"))
        bufferType = Importance::BUFFER_RS_BOX;
    else if (vm.count("pav"))
        bufferType = Importance::BUFFER_RS_BOX_PAV;

    test_distribution<FastLCG, SphericalGaussian, ISphericalGaussian>({
        .samples = samples,
        .output = "spherical_isg_lcg.data",
        .bufferType = bufferType, .error = error, .phiNodes = 15, .thetaNodes = 10 },
        6, glm::vec3(0, 0, 1));
    return 0;
}

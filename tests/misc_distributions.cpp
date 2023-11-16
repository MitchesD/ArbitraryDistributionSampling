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
#include <shared/distributions_1d.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace Importance;
namespace po = boost::program_options;

template<typename TBase>
struct TestData
{
    uint64_t samples;
    std::string output;
    BufferType bufferType;
    Float error;
    uint32_t nodeCount;
};

template<typename TGenerator, typename TDistribution, template<class> class TIDistribution, class... Args>
void test_distribution(TestData<TDistribution> const& data, Args... args)
{
    std::stringstream ss;
    ss << __func__ << "<" << TypeName<TGenerator>::Get() << ", " << TypeName<TDistribution>::Get() << ", " <<
        TypeName<TIDistribution<TDistribution>>::Get() << ">";
    TDistribution d(args...);
    TestSetupUniversal<TGenerator, TDistribution, Float, Grid1D<Constants::defaultVectorSize>> const setup(&d, ss.str(),
                                                                            data.output, data.samples,
                                                                            data.bufferType, data.error,
                                                                            // grid input parameters
                                                                            data.nodeCount, d.getXMin(), d.getXMax());

    TIDistribution<TGenerator> i_distribution(args...);
    std::ofstream str_vec(data.output);

    Float avg = 0.0f;
    Float max = 0.0f;

    for (uint64_t i = 0; i < data.samples; i++)
    {
        Float sample = i_distribution.sample(setup.buffer, setup.gen);
        Float value = i_distribution.f(sample);
        avg += value;

        if (value > max)
            max = value;
        str_vec << sample << ", " << value << std::endl;
    }

    std::cout << "Avg: " << avg / Float(data.samples) << std::endl;
    std::cout << "Max: " << i_distribution.max() << std::endl;
    std::cout << "Mesaured max: " << max << std::endl;
}

int main(int argc, char** argv)
{
    uint64_t samples = 300;
    BufferType bufferType = BUFFER_RS;
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

    test_distribution<FastLCG, NormalDistribution, INormalDistribution>(
        { .samples = samples, .output = "normal_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 10 },
        0.0f, 1, -8, 8);
    test_distribution<FastLCG, LandauDistribution, ILandauDistribution>(
        { .samples = samples, .output = "landau_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 25 },
        -3, 15);
    test_distribution<FastLCG, HyperbolicSecantDistribution, IHyperbolicSecantDistribution>(
        { .samples = samples, .output = "hypersecant_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 10 },
        -4, 4);
    test_distribution<FastLCG, GumbelDistribution, IGumbelDistribution>(
        { .samples = samples, .output = "gumbel_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 25 },
        1.5f, 3.0f, -10, 20);
    test_distribution<FastLCG, LogisticDistribution, ILogisticDistribution>(
        { .samples = samples, .output = "logistic_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 25 },
        2.0f, 1.0f, -10, 10);
    test_distribution<FastLCG, ExponentialDistribution, IExponentialDistribution>(
        { .samples = samples, .output = "exponential_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 10 },
        1.5f, 0, 7);
    test_distribution<FastLCG, CauchyDistribution, ICauchyDistribution>(
        { .samples = samples, .output = "cauchy_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 25 },
        0.0f, 0.5f, -50, 50);
    test_distribution<FastLCG, WeibullDistribution, IWeibullDistribution>(
        { .samples = samples, .output = "weibull_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 25 },
        1.0f, 5.0f, 0.0001, 3);
    test_distribution<FastLCG, LaplaceDistribution, ILaplaceDistribution>(
        { .samples = samples, .output = "laplace_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 25 },
        0.0f, 1.0f, -10, 10);
    test_distribution<FastLCG, RayleighDistribution, IRayleighDistribution>(
        { .samples = samples, .output = "rayleigh_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 10 },
        1.0f, 0, 10);
    test_distribution<FastLCG, FrechetDistribution, IFrechetDistribution>(
        { .samples = samples, .output = "frechet_lcg.data", .bufferType = bufferType, .error = error, .nodeCount = 10 },
        2.0f, 1.0f, 0.0f, 0.001f, 10);

    return 0;
}
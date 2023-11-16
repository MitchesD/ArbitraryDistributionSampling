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

#include <is/grid.h>
#include <generators/lcg.h>
#include <fstream>
#include <shared/test_setup.h>
#include <shared/distributions_1d.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <fmt/core.h>

using namespace Importance;
namespace po = boost::program_options;

struct TestData
{
    BufferType bufferType;
    uint64_t samples;
    std::string output;
    Float error;
    // sampled sub intervals of samples
    std::vector<uint64_t> intervals;
    uint32_t iterations;
    uint32_t nodeCount;
};

template<typename TGenerator, typename TDistribution, template<class> class TIDistribution, class... Args>
void test_distribution(TestData const& data, Args... args)
{
    std::stringstream ss;
    ss << __func__ << "<" << TypeName<TGenerator>::Get() << ", " << TypeName<TDistribution>::Get() << ", " <<
       TypeName<TIDistribution<TDistribution>>::Get() << ">";
    TDistribution d(args...);
    TestSetupUniversal<TGenerator, TDistribution, Float, Grid1D<Constants::defaultVectorSize>> const setup(&d, ss.str(), data.output,
                                                                     data.samples, data.bufferType, data.error,
                                                                     // grid input parameters
                                                                     data.nodeCount, d.getXMin(), d.getXMax());
    TIDistribution<TGenerator> i_distribution(args...);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<Float> uni_dis;

    for (std::size_t int_id = 0; int_id < data.intervals.size(); ++int_id)
    {
        for (uint32_t iter = 1; iter < data.iterations + 1; iter++)
        {
            // reset state and seq. id
            i_distribution.seq_id = 0;
            setup.gen->reset();

            auto interval = data.intervals[int_id];
            auto maxOffset = data.samples - interval;
            auto offset = static_cast<uint32_t>(uni_dis(generator) * (Float)maxOffset);
            if (offset > 0)
            {
                // skip to offset -> strides + samples
                auto skips = setup.buffer->getSkipsUpToIndex(offset);
                setup.gen->skipN(skips + offset);
                i_distribution.seq_id = offset;
            }

            std::string const s = fmt::format(fmt::runtime(data.output), int_id + 1, iter);
            std::ofstream str(s);
            std::cout << s << std::endl;
            std::cout << "offset: " << offset << " out of " << maxOffset << ", interval: " << interval
                      << ", max seq id: " << data.samples - 1 << std::endl << std::endl;
            for (uint64_t j = 0; j < interval; ++j)
            {
                Float const sample = i_distribution.sample(setup.buffer, setup.gen);
                str << sample << "\n";
            }
            str.close();
        }
    }

    std::string const chainFile = fmt::format(fmt::runtime(data.output), 0, 0);
    setup.dumpBufferSequence(chainFile, false, data.bufferType == Importance::BUFFER_RS_BOX);
}

int main(int argc, char** argv)
{
    Float error = 0.0f;
    uint32_t iterations = 1;
    BufferType bufferType = Importance::BUFFER_RS;
    std::string testName = "rsdef";

    std::vector<uint64_t> const pre_defined_samples = { 100, 500, 1000, 1500, 2000, 2500, 5000,
                                                        7500, 10000, 12500, 15000, 25000, 50000,
                                                        75000, 100000, 250000, 500000,
                                                        750000, 1000000, 1250000, 1500000, 2000000 };

    uint64_t samples = *std::max_element(pre_defined_samples.begin(), pre_defined_samples.end());

    po::options_description desc("Allowed options");
    desc.add_options()
        ("samples,s", po::value<uint64_t>(&samples)->implicit_value(200000), "Number of samples per test")
        ("iter,i", po::value<uint32_t>(&iterations)->implicit_value(1), "Number of iterations per test")
        ("error,e", po::value<Float>(&error)->implicit_value(0.0f), "Accept samples with error")
        ("advanced,a", "Enables advanced buffer")
        ("box,b", "Enables RS box buffer")
        ("pav,p", "Enables 2nd variant of RS box buffer");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if (vm.count("advanced"))
    {
        bufferType = Importance::BUFFER_RS_ADV;
        testName = "rsadv";
    }
    else if (vm.count("box"))
    {
        bufferType = Importance::BUFFER_RS_BOX;
        testName = "rsbox";
    }
    else if (vm.count("pav"))
    {
        bufferType = Importance::BUFFER_RS_BOX_PAV;
        testName = "rspav";
    }

    TestData data = { .bufferType = bufferType, .samples = samples, .output = "", .error = error,
        .intervals = pre_defined_samples, .iterations = iterations, .nodeCount = 0 };

    data.output = "data/test-" + testName +"-normal-{}-{}.data";
    data.nodeCount = 10;
    test_distribution<FastLCG, NormalDistribution, INormalDistribution>(data, 0.0f, 1, -7, 7);

    data.output = "data/test-" + testName + "-gumbel-{}-{}.data";
    data.nodeCount = 25;
    test_distribution<FastLCG, GumbelDistribution, IGumbelDistribution>(data,1.5f, 3.0f, -20, 50);

    data.output = "data/test-" + testName + "-logistic-{}-{}.data";
    data.nodeCount = 25;
    test_distribution<FastLCG, LogisticDistribution, ILogisticDistribution>(data,2.0f, 1.0f, -10, 20);

    data.output = "data/test-" + testName + "-exponential-{}-{}.data";
    data.nodeCount = 10;
    test_distribution<FastLCG, ExponentialDistribution, IExponentialDistribution>(data, 1.5f, 0, 7);

    data.output = "data/test-" + testName + "-cauchy-{}-{}.data";
    data.nodeCount = 25;
    test_distribution<FastLCG, CauchyDistribution, ICauchyDistribution>(data, 0, 1.0f, -100, 100);

    data.output = "data/test-" + testName + "-weibull-{}-{}.data";
    data.nodeCount = 10;
    test_distribution<FastLCG, WeibullDistribution, IWeibullDistribution>(data, 1, 5.f, 0.01f, 3);

    data.output = "data/test-" + testName + "-sech-{}-{}.data";
    data.nodeCount = 10;
    test_distribution<FastLCG, HyperbolicSecantDistribution, IHyperbolicSecantDistribution>(data, -6, 6);

    return 0;
}
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

#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <generators/xorshift.h>
#include <generators/lcg.h>
#include "shared/spherical_distributions.h"
#include <shared/distributions_1d.h>
#include <shared/api.h>

using namespace Importance;

constexpr uint32_t const JUMP_STEP = 200;

static void XS_JumpLog(benchmark::State& state)
{
    XorShiftGenerator gen;

    for (auto _ : state)
    {
        gen.skipN(JUMP_STEP);
    }
}

static void XS_JumpSeq(benchmark::State& state)
{
    XorShiftGenerator gen;

    for (auto _ : state)
    {
        for (uint32_t i = 0; i < JUMP_STEP; i++)
            gen.skip1D();
    }
}

static void LCG_JumpLog(benchmark::State& state)
{
    FastLCG gen;

    for (auto _ : state)
    {
        gen.skipN(JUMP_STEP);
    }
}

static void LCG_JumpSeq(benchmark::State& state)
{
    FastLCG gen;

    for (auto _ : state)
    {
        for (uint32_t i = 0; i < JUMP_STEP; i++)
            gen.skip1D();
    }
}

static void XS_Next(benchmark::State& state)
{
    XorShiftGenerator gen;

    for (auto _ : state)
    {
        gen.next1D();
    }
}

static void LCG_Next(benchmark::State& state)
{
    FastLCG gen;

    for (auto _ : state)
    {
        gen.next1D();
    }
}

static void SliceSampling(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    auto pdf = [](Float x) -> Float
    {
        return std::exp(-x * x / 2.0f) / std::sqrt(2.0f * F_PI);
    };

    // initial 'x'
    Float x = 0.0f;
    for (auto _ : state)
    {
        for (int i = 0; i < state.range(0); i++)
        {
            // generate 'y' sample using 'x'
            std::uniform_real_distribution<float> d(0.0f, pdf(x));
            Float y = d(gen);

            // now we have sliced function on 'y' coordinate
            Float alpha = std::sqrt(-2.0f * std::log(y * std::sqrt(2.0f * F_PI)));

            // sample from the slice
            std::uniform_real_distribution<Float> slice_distribution(-alpha, alpha);
            x = slice_distribution(gen);
            benchmark::DoNotOptimize(x);
        }
    }
}

static void RatioOfUniforms(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    Float e1 = 1./ 2.71828182845905f;

    for (auto _ : state)
    {
        for (int i = 0; i < state.range(0); i++)
        {
            while (true)
            {
                Float rn1 = uniform(gen);
                Float rn2 = uniform(gen);
                rn2 = (2.0f * rn2 - 1.0f) * std::sqrt(2.0f * e1);
                if (-4.0f * rn1 * rn1 * std::log(rn1) < rn2 * rn2)
                    continue;

                Float rn = rn2 / rn1;
                benchmark::DoNotOptimize(rn);
                break;
            }
        }
    }
}

static void BoxMueller(benchmark::State& state)
{
    Importance::SphericalGaussian sg(6, glm::vec3(0.0, 0, 1));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rng(0.0f, 1.0f);

    for (auto _ : state)
    {
        for (int i = 0; i < state.range(0); i++)
        {
            float a = rng(gen);
            float b = rng(gen);
            auto vec = sg.sample(glm::vec2(a, b));
            (void) vec;
        }
    }
}

static void InverseTransformMapping(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rng(0.0f, 1.0f);
    float lambda = 1.5f;

    for (auto _ : state)
    {
        for (int i = 0; i < state.range(0); i++)
        {
            float y0 = rng(gen);
            float x = -1 / lambda * std::log(1 - y0);
            (void) x;
        }
    }
}

static void RejectionSampling(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rng(0.0f, 1.0f);

    NormalDistribution d(0, 1, -5.0f, 5.0f);

    for (auto _ : state)
    {
        for (int i = 0; i < state.range(0); i++)
        {
            while (true)
            {
                Float x = rng(gen) * 10.0f - 5.0f;
                Float value = d.f(x);
                if (rng(gen) * d.max() > value)
                    continue;

                benchmark::DoNotOptimize(x);
                break;
            }
        }
    }
}

class WrappedAdvancedBufferBench : public benchmark::Fixture
{
public:
    WrappedAdvancedBufferBench()
    {
        NormalDistribution d(0, 1.0f, -5, 5);
        api = new API<FastLCG, NormalDistribution, Float, Grid1D<Constants::defaultVectorSize>>(BUFFER_RS_BOX,
                                                                                     0.0f, 10, d.getXMin(), d.getXMax());
        api->prepare(10000, &d);
    }

    ~WrappedAdvancedBufferBench() { delete api; }

    void SetUp(const ::benchmark::State& /*state*/) override { api->getGenerator()->reset(); }

    void TearDown(const ::benchmark::State& /*state*/) override { }

    API<FastLCG, NormalDistribution, Float, Grid1D<Constants::defaultVectorSize>>* api;
};

BENCHMARK_DEFINE_F(WrappedAdvancedBufferBench, NormalDistribution)(benchmark::State& state)
{
    INormalDistribution<FastLCG> d(0, 1, -5, 5);
    api->getGenerator()->reset();
    // run
    for (auto _ : state)
    {
        for (int i = 0; i < state.range(0); i++)
        {
            auto sample = d.sample(api->getBuffer(), api->getGenerator());
            benchmark::DoNotOptimize(sample);
        }
    }
}

BENCHMARK_DEFINE_F(WrappedAdvancedBufferBench, NormalFilter)(benchmark::State& state)
{
    INormalDistribution<FastLCG> d(0, 1, -5, 5);
    api->getGenerator()->reset();
    // run
    for (auto _ : state)
    {
        api->prepare(state.range(0), &d);
        api->getGenerator()->reset();
        for (int i = 0; i < state.range(0); i++)
        {
            auto sample = d.sample(api->getBuffer(), api->getGenerator());
            benchmark::DoNotOptimize(sample);
        }
    }
}

class WrappedGridBufferBench : public benchmark::Fixture
{
public:
    WrappedGridBufferBench()
    {
        NormalDistribution d(0, 1.0f, -5, 5);

        api = new API<FastLCG, NormalDistribution, Float, Grid1D<Constants::defaultVectorSize>>(BUFFER_RS_BOX_PAV,
            0.0f, 10, d.getXMin(), d.getXMax());
        api->prepare(1000000, &d);
    }

    ~WrappedGridBufferBench() { delete api; }

    void SetUp(const ::benchmark::State& /*state*/) override { api->getGenerator()->reset(); }

    void TearDown(const ::benchmark::State& /*state*/) override { }

    API<FastLCG, NormalDistribution, Float, Grid1D<Constants::defaultVectorSize>>* api;
};

BENCHMARK_DEFINE_F(WrappedGridBufferBench, NormalDistribution)(benchmark::State& state)
{
    INormalDistribution<FastLCG> d(0, 1, -5, 5);
    api->getGenerator()->reset();
    // run
    for (auto _ : state)
    {
        for (int i = 0; i < state.range(0); i++)
        {
            auto sample = d.sample(api->getBuffer(), api->getGenerator());
            benchmark::DoNotOptimize(sample);
        }
    }
}

BENCHMARK_DEFINE_F(WrappedGridBufferBench, NormalFilter)(benchmark::State& state)
{
    INormalDistribution<FastLCG> d(0, 1, -5, 5);
    api->getGenerator()->reset();
    // run
    for (auto _ : state)
    {
        api->prepare(state.range(0), &d);
        api->getGenerator()->reset();
        for (int i = 0; i < state.range(0); i++)
        {
            auto sample = d.sample(api->getBuffer(), api->getGenerator());
            benchmark::DoNotOptimize(sample);
        }
    }
}

BENCHMARK(XS_JumpLog);
BENCHMARK(XS_JumpSeq);
BENCHMARK(LCG_JumpLog);
BENCHMARK(LCG_JumpSeq);
BENCHMARK(XS_Next);
BENCHMARK(LCG_Next);

uint32_t const repetitions = 50;

std::vector<int64_t> const sizes = { 1, 2, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100,
                                     125, 150, 200, 250, 350, 500, 750, 1000, 1250, 1500,
                                     2000, 2500, 3000, 3500, 4000, 4500, 5000, 7500, 10000,
                                     12500, 15000, 17500, 20000, 25000, 30000, 40000, 50000, 75000, 100000,
                                     125000, 150000, 175000, 200000, 250000, 275000, 300000,
                                     350000, 400000, 450000, 500000, 550000, 600000, 650000,
                                     700000, 750000, 800000, 850000, 900000, 950000, 1000000 };

static void measureArguments(benchmark::internal::Benchmark* b)
{
    for (auto const& item : sizes)
        b->Arg(item);
}

BENCHMARK(SliceSampling)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK(RatioOfUniforms)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BoxMueller)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK(InverseTransformMapping)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK(RejectionSampling)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK_REGISTER_F(WrappedAdvancedBufferBench, NormalDistribution)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK_REGISTER_F(WrappedAdvancedBufferBench, NormalFilter)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK_REGISTER_F(WrappedGridBufferBench, NormalDistribution)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK_REGISTER_F(WrappedGridBufferBench, NormalFilter)
    ->Unit(benchmark::kMillisecond)
    ->Apply(measureArguments)
    ->Repetitions(repetitions)
    ->DisplayAggregatesOnly(true);

BENCHMARK_MAIN();

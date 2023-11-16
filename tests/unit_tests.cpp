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

#include <shared/config.h>
#include <is/ibuffer.h>
#include <gtest/gtest.h>
#include <shared/spherical_distributions.h>
#include <generators/lcg.h>
#include <generators/xorshift.h>
#include <is/grid.h>
#include <shared/api.h>
#include <bitset>

using namespace Importance;

TEST(BitMasks, BasicTests) {
    EXPECT_EQ(boxBits + strideBits, sizeof(BufferUnit) * 8);
}

class SphericalBufferTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        buffer = new RSWrappedAdvancedBuffer<SphericalGaussian, FastLCG,
            glm::vec3, GridSpherical<Constants::defaultVectorSize>>(5, 5);

        gen = new FastLCG();
        distribution = new ISphericalGaussian<FastLCG>(6, glm::vec3(0, 0, 1));

        InputData const inputData = { };
        buffer->filterUniversal(100, distribution, gen, inputData);
        gen->reset();
    }

    RSWrappedAdvancedBuffer<SphericalGaussian, FastLCG, glm::vec3, GridSpherical<Constants::defaultVectorSize>>* buffer;
    FastLCG* gen;
    ISphericalGaussian<FastLCG>* distribution;
};

TEST_F(SphericalBufferTest, Initialization) {
    auto grid = buffer->getGrid();
    EXPECT_EQ(grid->getNodesCount(), 25);
    GridRange range = grid->getNodeGridRange(0);
    RangeVector r = std::get<RangeVector>(range);
    // phi
    EXPECT_FLOAT_EQ(r.min.x, 0.0f);
    EXPECT_FLOAT_EQ(r.max.x, 2.0f * F_PI / 5.0f);
    // theta
    EXPECT_FLOAT_EQ(r.min.y, 0.0f);
    EXPECT_FLOAT_EQ(r.max.y, F_PI / 5.0f);

    r = std::get<RangeVector>(grid->getNodeGridRange(1));
    // phi
    EXPECT_FLOAT_EQ(r.min.x, (2.0f * F_PI) / 5.0f);
    EXPECT_FLOAT_EQ(r.max.x, 2.0f * (2.0f * F_PI) / 5.0f);
    // theta
    EXPECT_FLOAT_EQ(r.min.y, 0.0f);
    EXPECT_FLOAT_EQ(r.max.y, F_PI / 5.0f);
}

TEST(Generators, JumpTest) {
    XorShiftGenerator g1;
    XorShiftGenerator g2;

    int to_skip = 1000;
    g1.skipN(to_skip);

    for (int i = 0; i < to_skip; ++i)
        g2.skip1D();

    // states should be equal
    EXPECT_EQ(g1.get_state_string(), g2.get_state_string());

    g1.reset();
    g1.skipN(to_skip);
    g1.reset();
    for (int i = 0; i < 100; i++)
    {
        g1.skipN(0);
        g1.next2D();
    }
    g1.reset();
    g1.skipN(to_skip);
    EXPECT_EQ(g1.get_state_string(), g2.get_state_string());

    Importance::FastLCG lcg;
    Importance::FastLCG lcg2;
    lcg.skipN(5000);

    for (int i = 0; i < 5000; ++i)
        lcg2.skip1D();

    EXPECT_EQ(lcg.get_state_string(), lcg2.get_state_string());
}

TEST(Generators, Uniformity) {
    std::uniform_int_distribution<uint32_t> seedGen;
    std::random_device rd;
    std::mt19937 gen(rd());
    auto seed = seedGen(gen);
#if DETAILED_DEBUG
    std::cout << "Seed: " << seed << std::endl;
#endif
    FastLCG lcg(seed);
    double sum_left = 0.0f;
    double sum_right = 0.0f;
    double main_sum = 0.0f;

    int const samples = 100000;
    std::ofstream str1("left.rng");
    std::ofstream str2("right.rng");

    Float max = 1.0f / Float(0x7FFFFFFF);

    for (int i = 0; i < samples; i++)
    {
        uint64_t r = lcg.next1DInt();
        uint32_t const left = (r & 0x7FFFFFFF00000000u) >> 32u;
        uint32_t const right = r & 0xFFFFFFFFu;
        sum_left += left * max;
        str1 << left * max << (i % 2 == 0 ? " " : "\n");
        sum_right += right * INV_UNSIGNED_RAND_MAX;
        str2 << right * INV_UNSIGNED_RAND_MAX << (i % 2 == 0 ? " " : "\n");
    }

    str1.close();
    str2.close();

    for (int i = 0; i < samples; i++)
        main_sum += lcg.next1D();

    sum_left /= samples;
    sum_right /= samples;
    main_sum /= samples;
    EXPECT_NEAR(sum_left, sum_right, 0.01);
    EXPECT_NEAR(sum_left, main_sum, 0.01);
    EXPECT_NEAR(sum_right, main_sum, 0.01);
}

TEST(Generators, Masking) {
    uint64_t x = 0x5555555555555555llu;
    uint64_t y = x;
    x = ((x >> 1llu) | x) & 0x3333333333333333llu;
    x = ((x >> 2llu) | x) & 0x0F0F0F0F0F0F0F0Fllu;
    x = ((x >> 4llu) | x) & 0x00FF00FF00FF00FFllu;
    x = ((x >> 8llu) | x) & 0x0000FFFF0000FFFFllu;
    x = ((x >> 16llu) | x) & 0x00000000FFFFFFFFllu;

    std::bitset<64>  b(x);
    std::cout << b << std::endl;
    b = y;
    std::cout << b << std::endl;
    EXPECT_EQ(x, 0xFFFFFFFF);
}

TEST(Generators, Ranges) {
    NormalDistribution d1(0, 1.0f, -5, 5);
    EXPECT_EQ(d1.getRange(), 10.0f);
    NormalDistribution d2(0, 1.0f, 0, 5);
    EXPECT_EQ(d2.getRange(), 5.0f);
    NormalDistribution d3(0, 1.0f, -5, 0);
    EXPECT_EQ(d3.getRange(), 5.0f);
    NormalDistribution d4(0, 1.0f, -5, -1);
    EXPECT_EQ(d4.getRange(), 4.0f);
    NormalDistribution d5(0, 1.0f, 5, 10);
    EXPECT_EQ(d5.getRange(), 5.0f);

    Float r1 = 0.5f;
    Float p1 = d1.getRange() * r1 + d1.getXMin();
    EXPECT_EQ(p1, 0.0f);

    Float r2 = 0.5f;
    Float p2 = d2.getRange() * r2 + d2.getXMin();
    EXPECT_EQ(p2, 2.5f);

    Float r3 = 0.5f;
    Float p3 = d3.getRange() * r3 + d3.getXMin();
    EXPECT_EQ(p3, -2.5f);

    Float r4 = 0.5f;
    Float p4 = d4.getRange() * r4 + d4.getXMin();
    EXPECT_EQ(p4, -3.0f);

    Float r5 = 0.5f;
    Float p5 = d5.getRange() * r5 + d5.getXMin();
    EXPECT_EQ(p5, 7.5f);

    r1 = r2 = r3 = r4 = r5 = 1.0f;
    p1 = d1.getRange() * r1 + d1.getXMin();
    EXPECT_EQ(p1, 5.0f);

    p2 = d2.getRange() * r2 + d2.getXMin();
    EXPECT_EQ(p2, 5.0f);

    p3 = d3.getRange() * r3 + d3.getXMin();
    EXPECT_EQ(p3, 0.0f);

    p4 = d4.getRange() * r4 + d4.getXMin();
    EXPECT_EQ(p4, -1.0f);

    p5 = d5.getRange() * r5 + d5.getXMin();
    EXPECT_EQ(p5, 10.0f);

    r1 = r2 = r3 = r4 = r5 = 0.0f;
    p1 = d1.getRange() * r1 + d1.getXMin();
    EXPECT_EQ(p1, -5.0f);

    p2 = d2.getRange() * r2 + d2.getXMin();
    EXPECT_EQ(p2, 0.0f);

    p3 = d3.getRange() * r3 + d3.getXMin();
    EXPECT_EQ(p3, -5.0f);

    p4 = d4.getRange() * r4 + d4.getXMin();
    EXPECT_EQ(p4, -5.0f);

    p5 = d5.getRange() * r5 + d5.getXMin();
    EXPECT_EQ(p5, 5.0f);
}

TEST(API, LoadStore) {
    NormalDistribution d(0, 1.0f, -5, 5);
    API<FastLCG, NormalDistribution, Float,
        Grid1D<Constants::defaultVectorSize>> api(BUFFER_RS_BOX_PAV, 0.0f, 10, d.getXMin(), d.getXMax());

    api.prepare(10000, &d);
    api.save("loadStoreTest");

    API<FastLCG, NormalDistribution, Float,
        Grid1D<Constants::defaultVectorSize>> apiLoad(BUFFER_RS_BOX_PAV, 0.0f, 10, d.getXMin(), d.getXMax());
    apiLoad.load("loadStoreTest");

    auto ch1 = api.getBuffer()->getChain();
    auto ch2 = apiLoad.getBuffer()->getChain();
    for (uint32_t i = 0; i < ch1.size(); ++i)
        EXPECT_EQ(ch1[i], ch2[i]);
}

TEST(IVector, Pick) {
    std::vector<Float> values = { 0.05f, 0.05f, 0.25f, 0.5f, 0.15f };
    StaticVector<Float, 4> vec(values);
    EXPECT_EQ(vec.sample(0.03f), 0.05f);
    EXPECT_EQ(vec.sample(0.06f), 0.05f);
    EXPECT_EQ(vec.sample(0.33f), 0.25f);
    EXPECT_EQ(vec.sample(0.4f), 0.5f);
    EXPECT_EQ(vec.sample(0.8f), 0.5f);
    EXPECT_EQ(vec.sample(0.9f), 0.15f);
}

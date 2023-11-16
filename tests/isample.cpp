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
#include <shared/spherical_distributions.h>
#include <is/ibuffer.h>
#include <generators/lcg.h>
#include <generators/xorshift.h>
#include <map>
#include <utility>
#include <shared/test_setup.h>

using namespace Importance;

using MxData = std::vector<std::pair<float, glm::vec3>>;

struct TestData
{
    SphericalGaussian* base;
    MxData mixture_data = {};
    std::vector<Float> weights = {};
    uint64_t samples;
    std::string output;
    BufferType bufferType;
    Float error = 0.05f;
};

template<typename TGenerator>
void test_sample_single_isg(TestData const& data)
{
    TestSetupUniversal<TGenerator, SphericalGaussian, glm::vec3,
        GridSpherical<Constants::defaultVectorSize>> const setup(data.base,
                                                                 __func__, data.output, data.samples,
                                                                 data.bufferType, data.error,
                                                                 4, 2);

    Float const k = 6;
    ISphericalGaussian<TGenerator> isg(k, glm::vec3(0, 0, 1));
    std::ofstream str_vec(data.output);

    Float avg = 0.0f;
    Float max = 0.0f;

    for (uint64_t i = 0; i < data.samples; i++)
    {
        glm::vec3 const sample = isg.sample(setup.buffer, setup.gen);
        Float const value = isg.f(sample);
        avg += value;

        float phi, theta;
        spherical_phi_theta(sample, &phi, &theta);

        if (value > max)
            max = value;
        str_vec << sample.x << ", " << sample.y << ", " << sample.z << std::endl;
    }

    std::cout << "Avg: " << avg / Float(data.samples) << std::endl;
    std::cout << "Max: " << isg.max() << std::endl;
    std::cout << "Mesaured max: " << max << std::endl;
}

template<typename TGenerator, typename TDistribution, template<class> class TIDistribution, template<class> class TIMixture>
void test_mixture_distribution(TestData const& data)
{
    TestSetupUniversal<TGenerator, TDistribution, glm::vec3,
    GridSpherical<Constants::defaultVectorSize>> const setup(data.base,
                                                             __func__, data.output, data.samples,
                                                             data.bufferType, data.error,
                                                             20, 10);

    std::vector<TIDistribution<TGenerator>*> lobes;
    for (auto const& item : data.mixture_data)
        lobes.push_back(new TIDistribution<TGenerator>(item.first, item.second));

    TIMixture<TGenerator> mixture(lobes, data.weights, setup.buffer, *setup.gen);

    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<Float> distribution(0.0, 1.0);

    std::ofstream str_mix(data.output);
    for (uint64_t i = 0; i < data.samples; i++)
    {
        auto vector = mixture.sample(distribution(rng));

        float phi, theta;
        spherical_phi_theta(vector, &phi, &theta);
        str_mix << phi << ", " << theta << std::endl;
    }
    str_mix.close();

    for (auto& item : lobes)
        delete item;
}

void test_native_rejection_sampling(TestData const& data)
{
    Float const k = 6;
    SphericalGaussian isg(k, glm::vec3(0, 0, 1));
    std::ofstream str_vec(data.output);

    std::ofstream str_phi(data.output + ".phi");
    std::ofstream str_theta(data.output + ".theta");

    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<Float> distribution(0.0, 1.0);

    for (uint64_t i = 0; i < data.samples; i++)
    {
        while (true)
        {
            glm::vec3 sample = uniformSphereSample({distribution(rng), distribution(rng)});

            Float value = isg.f(sample);
            if (distribution(rng) * isg.max() > value)
                continue;

            float phi, theta;
            spherical_phi_theta(sample, &phi, &theta);

            str_vec << sample.x << ", " << sample.y << ", " << sample.z << std::endl;

            str_phi << phi << "\n";
            str_theta << theta << "\n";
            break;
        }
    }
}

int main()
{
    std::vector<std::pair<float, glm::vec3>> const mixture_data =
    {
        { 6, glm::vec3(0, 0, 1) },
        { 2, glm::vec3(0, -1, 0) },
        { 8, glm::vec3(0.5, 0.0, 0.6) },
        { 4, glm::vec3(-0.5, -0.3, 0.6) },
        { 12, glm::vec3(1, 0, 0) },
    };

    std::vector<float> const weights = { 0.35f, 0.3f, 0.2f, 0.05f, 0.1f };

    Float const k = 6;

    // used as generalization for filtering, for all bases!!!
    Importance::SphericalGaussian sg(k, glm::vec3(0, 0, 1));

    test_native_rejection_sampling({.base = &sg, .samples = 50000, .output = "isample_lcg_def.data", .bufferType = Importance::BUFFER_RS });
    test_sample_single_isg<FastLCG>({.base = &sg, .samples = 50000, .output = "isample_lcg_def.data", .bufferType = Importance::BUFFER_RS_ADV });
    test_sample_single_isg<FastLCG>({.base = &sg, .samples = 50000, .output = "isample_lcg_box.data", .bufferType = Importance::BUFFER_RS_BOX });
    test_sample_single_isg<FastLCG>({.base = &sg, .samples = 50000, .output = "isample_lcg_pav.data", .bufferType = Importance::BUFFER_RS_BOX_PAV });
    test_sample_single_isg<XorShiftGenerator>({.base = &sg, .samples = 50000, .output = "isample_xor.data", .bufferType = Importance::BUFFER_RS });
    test_mixture_distribution<FastLCG, SphericalGaussian, ISphericalGaussian, ISphericalGaussianMixture>({
        .base = &sg,
        .mixture_data = mixture_data,
        .weights = weights,
        .samples = 5000,
        .output = "mixture_lcg.data",
        .bufferType = Importance::BUFFER_RS_BOX });
    test_mixture_distribution<XorShiftGenerator, SphericalGaussian, ISphericalGaussian, ISphericalGaussianMixture>({
        .base = &sg,
        .mixture_data = mixture_data,
        .weights = weights,
        .samples = 50000,
        .output = "mixture_xor.data",
        .bufferType = Importance::BUFFER_RS });

    return 0;
}

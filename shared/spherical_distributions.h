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

#ifndef IMPORTANCE_SG_FACTORY_H__
#define IMPORTANCE_SG_FACTORY_H__

#include <utility>
#include <vector>
#include <cmath>
#include <shared/config.h>
#include <shared/fast_float.h>
#include <shared/mapping.h>
#include <shared/distribution.h>
#include <is/ibuffer.h>
#include <is/ivector.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>

namespace Importance
{

class SphericalGaussian : public SphericalDistribution
{
public:
    /**
     * SG constructor
     * @param k_ concentration parameter
     * @param m_ mean direction
     */
    SphericalGaussian(Float k_, glm::vec3 const& m_) : SphericalDistribution(), k(k_), m(m_), rot(1.0f)
    {
        prepareRotationMatrix();
    }

    /**
     * Computes the Spherical Gaussian value
     * @param x evaluated direction
     * @return the value of SG at given direction
     */
    Float f(glm::vec3 const& x) const
    {
        Float product = glm::dot(x, m);
        return constant() * std::exp(k * product);
    }

    /**
     * Sample Gaussian with Box-Muller transform, in square domain
     * @param random sample
     * @return sample mapped on square
     */
    glm::vec2 sampleSquare(glm::vec2 random) const
    {
        ASSERT( random.x >= 0.f && random.x <= 1.f );
        random.x = std::max(random.x, std::numeric_limits<Float>::min());
        const Float mult = std::sqrt(-2.0f * std::log(random.x));
        const Float r_angle = 2 * M_PI * random.y;

        Float sin_angle, cos_angle;
        Ff::sincos(r_angle, sin_angle, cos_angle);
        glm::vec2 x = glm::vec2(mult * cos_angle, mult * sin_angle);

        // transform according to concentration parameter
        // C = k * I = A
        glm::vec2 dir;
        float invDetSqrt = 1.0f / std::sqrt( k * k);
        float a11 = std::sqrt(k);
        float a22 = std::sqrt(k);
        dir.x = invDetSqrt * a11 * x.x;
        dir.y = invDetSqrt * a22 * x.y;

        return dir;
    }

    /**
     * Sample Gaussian using Box-Muller transform to sample on square and
     * Shirley-Chi mapping to hemisphere. The sample is then rotated according
     * to mean direction frame
     * @param random sample
     * @return sample
     */
    glm::vec3 sample(glm::vec2 random)
    {
        const glm::vec2 x = sampleSquare(random);
        glm::vec3 x3;
        uniformHemisphereShirley(x, x3);
        x3 = x3 * rot;

        return x3;
    }

    [[nodiscard]] Float max() const override
    {
        return f(m);
    }

protected:
    Float constant() const
    {
        Float denominator = 2.0f * M_PI * (std::exp(k) - std::exp(-k));
        return k / denominator;
    }

    glm::mat3 cpm(glm::vec3 const& a)
    {
        glm::mat3 mat;
        mat[0][0] = 0.0f;
        mat[0][1] = -a.z;
        mat[0][2] = a.y;

        mat[1][0] = a.z;
        mat[1][1] = 0.0f;
        mat[1][2] = -a.x;

        mat[2][0] = -a.y;
        mat[2][1] = a.x;
        mat[2][2] = 0.0f;
        return mat;
    }

    void prepareRotationMatrix()
    {
        auto float_close = [](Float a, Float b) {
            return std::abs(a - b) < 1e-3;
        };

        rot = glm::mat3x3(1);

        Float angle = std::acos(m.z);
        if (!float_close(angle, 0.0f))
        {
            glm::vec3 axis(-m.y, -m.x, 0.f);
            axis = glm::normalize(axis);

            Float sinAngle = std::sin(angle);
            Float cosAngle = m.z; // std::cos(angle)
            rot = cosAngle * glm::mat3(1.0f) + sinAngle * cpm(axis) + (1.0f - cosAngle) * glm::outerProduct(axis, axis);
        }
    }

    /// concentration parameter
    Float k;
    /// mean direction
    glm::vec3 m;
    /// rotation matrix according to mean direction
    glm::mat3x3 rot;
};

template<class TGenerator>
class ISphericalGaussian : public SphericalGaussian
{
public:
    ISphericalGaussian(Float k_, glm::vec3 const& m_) : SphericalGaussian(k_, m_), seq_id(0) { }

    virtual ~ISphericalGaussian() = default;

    /**
     * Sample Gaussian with moderated TGenerator using the ImportanceBuffer
     * @param buffer a buffer containing the skipping sequence and sampling methods
     * @param gen    deterministic generator
     * @return sample
     */
    glm::vec3 sample(AbstractImportanceBuffer<SphericalGaussian, TGenerator, glm::vec3>* buffer, TGenerator* gen)
    {
        glm::vec3 sample = buffer->sample(seq_id, gen);

        // warp according to concentration parameter _k_
        float invDetSqrt = 1.0f / std::sqrt( k * k);
        float a11 = std::sqrt(k);
        float a22 = std::sqrt(k);
        sample.x = invDetSqrt * a11 * sample.x;
        sample.y = invDetSqrt * a22 * sample.y;
        sample = glm::normalize(sample);
        return sample * rot;
    }

private:
    /// index into skipping sequence
    uint32_t seq_id;
};

template<class TGenerator>
class ISphericalGaussianMixture
{
private:
    struct IWeightedLobe
    {
        explicit IWeightedLobe(Float _val, ISphericalGaussian<TGenerator>* lobe_, TGenerator gen_)
            : weight(_val), lobe(lobe_)
        {
            gen = std::make_shared<TGenerator>(gen_);
        }

        ~IWeightedLobe() = default;

        IWeightedLobe() : weight(0), lobe(nullptr), gen(nullptr) { }

        Float weight;
        ISphericalGaussian<TGenerator>* lobe;
        std::shared_ptr<TGenerator> gen;

        // required operator overloads
        template<class T>
        IWeightedLobe& operator/=(T const& a)
        {
            weight /= a;
            return *this;
        }

        operator Float() const
        {
            return weight;
        }

        // terrible hack, required just for testing
        // so we don't have to define function with pointer template
        float operator*() const
        {
            return weight;
        }
    };

public:
    ISphericalGaussianMixture(std::vector<ISphericalGaussian<TGenerator>*> const& lobes_,
                              std::vector<Float> const& weights_,
                              AbstractImportanceBuffer<SphericalGaussian, TGenerator, glm::vec3>* buffer_,
                              TGenerator gen_)
        : lobes(nullptr), buffer(buffer_)
    {
        std::vector<IWeightedLobe> prep_lobes;
        prep_lobes.reserve(lobes_.size());
        for (size_t i = 0; i < lobes_.size(); ++i)
            prep_lobes.emplace_back(weights_[i], lobes_[i], gen_);

        lobes = new StaticVector<IWeightedLobe, Constants::defaultVectorSize>(prep_lobes);
    }

    ~ISphericalGaussianMixture()
    {
        delete lobes;
    }

    glm::vec3 sample(Float r1)
    {
        // 1. weight pick lobe
        IWeightedLobe rec = lobes->sample(r1);

        // 2. sample given lobe with buffer and generator
        return rec.lobe->sample(buffer, rec.gen.get());
    }

private:
    StaticVector<IWeightedLobe, Constants::defaultVectorSize>* lobes;
    AbstractImportanceBuffer<SphericalGaussian, TGenerator, glm::vec3>* buffer;
};

class SphericalGaussianMixture
{
public:
    SphericalGaussianMixture(std::vector<Float> weights_, std::vector<SphericalGaussian*> lobes_) :
        weights(std::move(weights_)), lobes(std::move(lobes_)) { }

    glm::vec3 sample(glm::vec2 random)
    {
        Float cdf = 0.0f, prev_cdf = 0.0f;
        uint32_t actualLobe = 0;
        uint32_t i = 0;
        while (true)
        {
            prev_cdf = cdf;
            cdf += weights[actualLobe];
            if (cdf >= random.y || i == lobes.size() - 1)
                break;
            actualLobe++;
            ++i;
        }

        random.y = std::min(1 - FLT_EPSILON, (random.y - prev_cdf) / (cdf - prev_cdf));
        auto lobe = lobes[actualLobe];
        return lobe->sample(random);
    }

private:
    std::vector<Float> weights;
    std::vector<SphericalGaussian*> lobes;
};

}

#endif // IMPORTANCE_SG_FACTORY_H__

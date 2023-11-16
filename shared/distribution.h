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

#ifndef IMPORTANCE_DISTRIBUTION_H__
#define IMPORTANCE_DISTRIBUTION_H__

#include <shared/config.h>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <is/range.h>
#include <cmath>

namespace Importance
{

class Distribution
{
public:
    Distribution() = default;

    virtual Float max() const { ASSERT_MESSAGE("", "Distribution1D::max not implemented"); return 0.0f; };
};

class Distribution1D : public Distribution
{
public:
    Distribution1D(Float x_min_, Float x_max_) : x_min(x_min_), x_max(x_max_)
    {
        x_range = x_max - x_min;
    }

    virtual Float f(Float const& /*x*/) const { ASSERT_MESSAGE("", "Distribution1D::f not implemented"); return 0.0f; }

    template<typename TGenerator>
    Float domainSample(TGenerator* gen)
    {
        return gen->next1D() * x_range + x_min;
    }

    template<typename TGenerator>
    Float domainSample(TGenerator* gen, GridRange const& range)
    {
        auto r = std::get<RangeScalar>(range);
        return gen->next1D() * (r.max - r.min) + r.min;
    }

    Float domainSample(Float u, GridRange const& range)
    {
        auto r = std::get<RangeScalar>(range);
        return u * (r.max - r.min) + r.min;
    }

    Float getXMin() const { return x_min; }
    Float getXMax() const { return x_max; }
    Float getRange() const { return x_range; }

protected:
    /// domain limits and range
    Float x_min;
    Float x_max;
    Float x_range;
};

class SphericalDistribution : public Distribution
{
public:
    SphericalDistribution() = default;

    /**
     * Spherical coordinates
     * @return
     */
    virtual Float f(glm::vec3 const& /*x*/) const { ASSERT_MESSAGE("", "SphericalDistribution::f not implemented"); return 0.0f; }

    /**
     * Cartesian coordinates
     * @return
     */
    virtual Float f(glm::vec2 const& /*x*/) const { ASSERT_MESSAGE("", "SphericalDistribution::f not implemented"); return 0.0f; }

    template<typename TGenerator>
    glm::vec3 domainSample(TGenerator* gen)
    {
        glm::vec2 u = gen->next2D();
        Float z = 1.0f - 2.0f * u[0];
        Float r = std::sqrt(std::max((Float)0, (Float)1. - z * z));
        Float phi = 2.0f * M_PI * u[1];
        return { r * std::cos(phi), r * std::sin(phi), z };
    }

    template<typename TGenerator>
    glm::vec3 domainSample(TGenerator* gen, GridRange const& range)
    {
        auto r = std::get<RangeVector>(range);
        glm::vec2 u = gen->next2D();

        // note that this formula does not follow proper conformal mapping
        Float theta = (r.max.y - r.min.y) * u[0] + r.min.y;
        ASSERT_MESSAGE(theta >= r.min.y && theta <= r.max.y, theta);
        Float cosTheta = std::cos(theta);
        Float sinTheta = std::sin(theta);
        Float phi = u[1] * (r.max.x - r.min.x) + r.min.x;

        ASSERT_MESSAGE(phi >= r.min.x && phi <= r.max.x, phi);

        return { sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta };
    }

    glm::vec3 domainSample(glm::vec2 u, GridRange const& range)
    {
        auto r = std::get<RangeVector>(range);

        // note that this formula does not follow proper conformal mapping
        Float theta = (r.max.y - r.min.y) * u[0] + r.min.y;
        ASSERT_MESSAGE(!(theta < r.min.y && theta > r.max.y), theta << " " << r.min.y << " " << r.max.y);
        Float cosTheta = std::cos(theta);
        Float sinTheta = std::sin(theta);
        Float phi = u[1] * (r.max.x - r.min.x) + r.min.x;

        ASSERT_MESSAGE(phi >= r.min.x && phi <= r.max.x, phi);

        return { sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta };
    }
};

class CircularDistribution : public Distribution
{
public:
    CircularDistribution() = default;

    virtual Float f(Float const& /*x*/) const { ASSERT_MESSAGE("", "CircularDistribution::f not implemented"); return 0.0f; }

    template<typename TGenerator>
    Float domainSample(TGenerator* gen)
    {
        return 2.0f * F_PI * gen->next1D();
    }

    template<typename TGenerator>
    Float domainSample(TGenerator* gen, GridRange const& range)
    {
        auto r = std::get<RangeScalar>(range);
        return gen->next1D() * (r.max - r.min) + r.min;
    }

    Float domainSample(Float u, GridRange const& range)
    {
        auto r = std::get<RangeScalar>(range);
        return u * (r.max - r.min) + r.min;
    }
};

class Distribution2D : public Distribution
{
public:
    Distribution2D(glm::vec2 min_, glm::vec2 max_) : min(min_), max(max_)
    {
        Float const xRange = max.x - min.x;
        Float const yRange = max.y - min.y;

        xyRange = { xRange, yRange };
    }

    virtual Float f(glm::vec2 const& /*x*/) const { ASSERT_MESSAGE("", "Distribution2D::f not implemented"); return 0.0f; }

    template<typename TGenerator>
    glm::vec2 domainSample(TGenerator* gen)
    {
        glm::vec2 u = gen->next2D();
        return { u[0] * xyRange.x + min.x, u[1] * xyRange.y + min.y };
    }

    template<typename TGenerator>
    glm::vec2 domainSample(TGenerator* gen, GridRange const& range)
    {
        auto r = std::get<RangeVector>(range);
        glm::vec2 u = gen->next2D();

        Float const x = (r.max.x - r.min.x) * u[0] + r.min.x;
        Float const y = (r.max.y - r.min.y) * u[1] + r.min.y;
        return { x, y };
    }

    glm::vec2 domainSample(glm::vec2 u, GridRange const& range)
    {
        auto r = std::get<RangeVector>(range);

        Float const x = (r.max.x - r.min.x) * u[0] + r.min.x;
        Float const y = (r.max.y - r.min.y) * u[1] + r.min.y;
        return { x, y };
    }

    glm::vec2 getMin() const { return min; }
    glm::vec2 getMax() const { return max; }
    glm::vec2 getRange() const { return xyRange; }

protected:
    glm::vec2 min;
    glm::vec2 max;
    glm::vec2 xyRange;
};

} // namespace Importance

#endif // IMPORTANCE_DISTRIBUTION_H__

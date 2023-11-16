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

#ifndef IMPORTANCE_MAPPING_H__
#define IMPORTANCE_MAPPING_H__

#include <iostream>
#include <cmath>
#include <algorithm>
#include <shared/config.h>
#include <shared/fast_float.h>
#include <glm/glm.hpp>

namespace Importance
{

inline glm::vec3 uniformHemisphereSample(glm::vec2 const& u)
{
    Float z = u[0];
    Float r = std::sqrt(std::max((Float)0, (Float)1. - z * z));
    Float phi = 2 * M_PI * u[1];
    return { r * std::cos(phi), r * std::sin(phi), z };
}

inline glm::vec3 uniformSphereSample(glm::vec2 const& u)
{
    Float z = 1.0f - 2.0f * u[0];
    Float r = std::sqrt(1.0f - z * z);
    Float phi = 2.0f * M_PI * u[1];
    Float sinPhi = std::sin(phi);
    Float cosPhi = std::cos(phi);
    return { r * cosPhi, r * sinPhi, z };
}

glm::vec2 concentricSampleDisk(glm::vec2 const& u)
{
    // Map uniform random numbers to $[-1,1]^2$
    glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return glm::vec2(0, 0);

    // Apply concentric mapping to point
    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = M_PI_4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = M_PI_2 - M_PI_4 * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(std::cos(theta), std::sin(theta));
}

inline glm::vec3 cosineSampleHemisphere(glm::vec2 const& u)
{
    glm::vec2 d = concentricSampleDisk(u);
    Float z = std::sqrt(std::max((Float)0, 1 - d.x * d.x - d.y * d.y));
    return glm::vec3(d.x, d.y, z);
}

glm::vec3 uniformHemisphereShirley(Float u1, Float u2)
{
    Float phi, r;
    const Float a = u1;//2.f*u1 - 1.f;   // (a,b) is now on [-1,1]^2
    const Float b = u2;//2.f*u2 - 1.f;
    if(a > -b) {     // region 1 or 2
        if(a > b) {  // region 1, also |a| > |b|
            r = a;
            phi = (M_PI / 4.f) * (b / a);
        } else {  // region 2, also |b| > |a|
            r = b;
            phi = (M_PI / 4.f) * (2.f - (a / b));
        }
    } else {        // region 3 or 4
        if (a < b) {  // region 3, also |a| >= |b|, a != 0
            r = -a;
            phi = (M_PI / 4.f) * (4.f + (b / a));
        }
        else {  // region 4, |b| >= |a|, but a==0 and b==0 could occur.
            r = -b;
            if (b != 0.f) {
                phi = (M_PI / 4.f) * (6.f - (a / b));
            } else {
                phi = 0.f;
            }
        }
    }
    const Float rSqr = r * r;
    const Float scaleFactor = r * std::sqrt( std::max(2 - rSqr , 0.0f));
    Float cosPhi = std::cos(phi);
    Float sinPhi = std::sin(phi);
    const glm::vec3 result( cosPhi * scaleFactor, sinPhi * scaleFactor, 1 - rSqr );
    return result;
}

glm::vec2 uniformHemisphereShirley2(Float u1, Float u2)
{
    Float phi, r;
    const Float a = 2.f * u1 - 1.f;   // (a,b) is now on [-1,1]^2
    const Float b = 2.f * u2 - 1.f;
    if(a > -b) {     // region 1 or 2
        if(a > b) {  // region 1, also |a| > |b|
            r = a;
            phi = (M_PI / 4.f) * (b / a);
        } else {  // region 2, also |b| > |a|
            r = b;
            phi = (M_PI / 4.f) * (2.f - (a / b));
        }
    } else {        // region 3 or 4
        if (a < b) {  // region 3, also |a| >= |b|, a != 0
            r = -a;
            phi = (M_PI / 4.f) * (4.f + (b / a));
        }
        else {  // region 4, |b| >= |a|, but a==0 and b==0 could occur.
            r = -b;
            if (b != 0.f) {
                phi = (M_PI / 4.f) * (6.f - (a / b));
            } else {
                phi = 0.f;
            }
        }
    }
    glm::vec2 result;
    result.x = r * std::cos(phi);
    result.y = r * std::sin(phi);
    return result;
}

bool uniformHemisphereShirley(const glm::vec2& point, glm::vec3& oLocalDir)
{
    if (point.x < 0.0f || point.x > 1.0f || point.y < 0.0f || point.y > 1.0f)
    {
        oLocalDir = glm::vec3(0.0f,0.0f,-1.0f);
        return false;
    }
    oLocalDir = uniformHemisphereShirley(point.x, point.y);
    return true;
}

}

#endif // IMPORTANCE_MAPPING_H__

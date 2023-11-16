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

#ifndef IMPORTANCE_DISTRIBUTIONS_2D_H
#define IMPORTANCE_DISTRIBUTIONS_2D_H

#define STB_IMAGE_IMPLEMENTATION
#include <shared/stb_image.h>
#include <shared/distribution.h>
#include <is/ibuffer.h>

namespace Importance
{

class BivariateNormalDistribution : public Distribution2D
{
public:
    BivariateNormalDistribution(glm::vec2 mu_, glm::vec2 si_, Float ro_, glm::vec2 min, glm::vec2 max)
        : Distribution2D(min, max), mu(mu_), si(si_), ro(ro_) { }

    [[nodiscard]] Float f(glm::vec2 const& x) const override
    {
        Float const xt = (x.x - mu.x) / si.x;
        Float const yt = (x.y - mu.y) / si.y;
        Float const rot = 1.0f - ro * ro;
        Float const frac = 2.0f * F_PI * si.x * si.y * std::sqrt(rot);
        Float const exponent = -1.0f / (2.0f * rot) * (xt * xt + yt * yt - 2.0f * ro * xt * yt);

        return 1.0f / frac * std::exp(exponent);
    }

    [[nodiscard]] Float max() const final
    {
        return f(mu);
    }

private:
    glm::vec2 mu;
    glm::vec2 si;
    Float ro;
};

template<class TGenerator>
class IBivariateNormalDistribution : public BivariateNormalDistribution
{
public:
    IBivariateNormalDistribution(glm::vec2 mu_, glm::vec2 si_, Float ro_, glm::vec2 min, glm::vec2 max)
        : BivariateNormalDistribution(mu_, si_, ro_, min, max), seq_id(0) { }

    glm::vec2 sample(AbstractImportanceBuffer<BivariateNormalDistribution, TGenerator, glm::vec2>* buffer, TGenerator* gen)
    {
        return buffer->sample2D(seq_id, gen, xyRange, min);
    }

private:
    uint32_t seq_id;
};

class HatDistribution : public Distribution2D
{
public:
    HatDistribution(glm::vec2 min, glm::vec2 max) : Distribution2D(min, max) { }

    [[nodiscard]] Float f(glm::vec2 const& x) const override
    {
        Float const x2y2 = x.x * x.x + x.y * x.y;

        Float const a = x2y2 - 1.0f;
        Float const b = x2y2 + 1.0f;
        Float const c = x2y2 - 2.0f;
        Float const d = x2y2 + 2.0f;
        Float const e = x2y2 - 3.0f;
        Float const f = x2y2 + 3.0f;
        Float nominator = (a * b * c * d * e * f) / 5.0f + 10.0f;
        Float denominator = std::pow(std::abs(x2y2), std::abs(x2y2));
        return nominator / denominator;
    }

    [[nodiscard]] Float max() const final
    {
        return f({0.865325f, 0.486358f});
    }
};

template<class TGenerator>
class IHatDistribution : public HatDistribution
{
public:
    IHatDistribution(glm::vec2 min, glm::vec2 max) : HatDistribution(min, max), seq_id(0) { }

    glm::vec2 sample(AbstractImportanceBuffer<HatDistribution, TGenerator, glm::vec2>* buffer, TGenerator* gen)
    {
        return buffer->sample2D(seq_id, gen, xyRange, min);
    }

private:
    uint32_t seq_id;
};

class HurjunDistribution : public Distribution2D
{
public:
    HurjunDistribution(glm::vec2 min, glm::vec2 max) : Distribution2D(min, max) { }

    [[nodiscard]] Float f(glm::vec2 const& x) const override
    {
        return hurjun(x);
    }

    [[nodiscard]] Float max() const final
    {
        return 1.0f;
    }

private:
    static Float sdf_line(glm::vec2 A, glm::vec2 B, glm::vec2 X)
    {
        float t = -glm::dot(A - X,B - A) / glm::dot(B - A,B - A);
        t = glm::clamp(t, 0.0f, 1.0f);
        glm::vec2 const L = A + t * (B - A);
        return glm::distance(L,X);
    }

    static Float sdEllipse(glm::vec2 p, glm::vec2 ab)
    {
        p = glm::abs(p);
        if (p.x > p.y)
        {
            Float z = p.x;
            p.x = p.y;
            p.y = z;
            z = ab.x;
            ab.x = ab.y;
            ab.y = z;
        }
        Float const l = ab.y * ab.y - ab.x * ab.x;
        Float const m = ab.x * p.x / l;
        Float const m2 = m * m;
        Float const n = ab.y * p.y / l;
        Float const n2 = n * n;
        Float const c = (m2 + n2 - 1.0f) / 3.0f;
        Float const c3 = c * c * c;
        Float const q = c3 + m2 * n2 * 2.0f;
        Float const d = c3 + m2 * n2;
        Float const g = m + m * n2;
        Float co;
        if (d < 0.0 )
        {
            Float const h = glm::acos(q / c3) / 3.0f;
            Float const s = glm::cos(h);
            Float const t = glm::sin(h) * glm::sqrt(3.0f);
            Float const rx = glm::sqrt(-c * (s + t + 2.0f) + m2);
            Float const ry = glm::sqrt(-c * (s - t + 2.0f) + m2);
            co = (ry + glm::sign(l) * rx + glm::abs(g) / (rx * ry) - m) / 2.0f;
        }
        else
        {
            Float const h = 2.0f * m * n * glm::sqrt(d);
            Float const s = glm::sign(q + h) * glm::pow(glm::abs(q + h), 1.0f / 3.0f);
            Float const u = glm::sign(q - h) * glm::pow(glm::abs(q - h), 1.0f / 3.0f);
            Float const rx = -s - u - c * 4.0f + 2.0f * m2;
            Float const ry = (s - u) * glm::sqrt(3.0f);
            Float const rm = glm::sqrt(rx * rx + ry * ry);
            co = (ry / glm::sqrt(rm - rx) + 2.0f * g / rm - m) / 2.0f;
        }
        glm::vec2 const r = ab * glm::vec2(co, glm::sqrt(1.0f - co * co));
        return glm::length(r - p) * glm::sign(p.y - r.y);
    }

    static Float hurjun(glm::vec2 uv)
    {
        Float d = 1e1f;

        d = glm::min(d,glm::abs(sdEllipse(uv-.5f, glm::vec2(.2,.15))));
        d = glm::min(d,sdf_line(glm::vec2(.2,.73), glm::vec2(.8,.73),uv));
        d = glm::min(d,sdf_line(glm::vec2(.5,.73), glm::vec2(.5,.85),uv));
        d = glm::min(d,sdf_line(glm::vec2(.95,.2), glm::vec2(.95,.85),uv));
        d = glm::min(d,sdf_line(glm::vec2(.8,.55), glm::vec2(.95,.55),uv));
        d = glm::min(d,sdf_line(glm::vec2(1.1,.85), glm::vec2(1.6,.85),uv));
        d = glm::min(d,sdf_line(glm::vec2(1.35,.85), glm::vec2(1.15,.6),uv));
        d = glm::min(d,sdf_line(glm::vec2(1.35,.85), glm::vec2(1.55,.6),uv));
        d = glm::min(d,sdf_line(glm::vec2(1.05,.5), glm::vec2(1.65,.5),uv));
        d = glm::min(d,sdf_line(glm::vec2(1.35,.5), glm::vec2(1.35,.35),uv));
        d = glm::min(d,sdf_line(glm::vec2(1.1,.4), glm::vec2(1.1,.2),uv));
        d = glm::min(d,sdf_line(glm::vec2(1.1,.2), glm::vec2(1.6,.2),uv));
        return glm::clamp(glm::pow(glm::clamp(1.4f - (glm::pow(glm::clamp(d, 0.f, 1.f), 0.2f)),
                                              0.f, 1.f), 5.2f), 0.f, 1.f);
    }
};

template<class TGenerator>
class IHurjunDistribution : public HurjunDistribution
{
public:
    IHurjunDistribution(glm::vec2 min, glm::vec2 max) : HurjunDistribution(min, max), seq_id(0) { }

    glm::vec2 sample(AbstractImportanceBuffer<HurjunDistribution, TGenerator, glm::vec2>* buffer, TGenerator* gen)
    {
        return buffer->sample2D(seq_id, gen, xyRange, min);
    }

private:
    uint32_t seq_id;
};

class EnvironmentMap : public Distribution2D
{
public:
    EnvironmentMap(std::string const& file_, glm::vec2 min, glm::vec2 max) : Distribution2D(min, max),
                                                                             width(0), height(0)
    {
        load(file_);
    }

    [[nodiscard]] Float f(glm::vec2 const& x) const override
    {
        // round to nearest pixel
        int i = std::min(std::max(0, int(x.x)), width - 1);
        int j = std::min(std::max(0, int(x.y)), height - 1);
        return pixels[i + j * width];
    }

    void store(std::string const& file, std::vector<glm::vec2> const& points, std::vector<glm::vec3> const& pointsColor)
    {
        std::vector<glm::vec3> global_buffer = pixelsRGB;

        int c = 0;
        for (auto const& p : points)
        {
            int y = p.y;
            int x = p.x;

            glm::vec3 color = pointsColor[c++];
            for (int i = -2; i < 2; i++)
                for (int j = -2; j < 2; j++)
                    setPixel(x + i, y + j, width, height, color, global_buffer);
        }

        storeBufferToPPM(file, global_buffer, 2.2f);
    }

private:
    void load(std::string const& file)
    {
        int channels = 0;
        unsigned char* data = stbi_load(file.c_str(), &width, &height, &channels, 0);
        if (!data)
            throw std::runtime_error("Failed to load image");

        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                Float r = data[(i + j * width) * 3 + 0] / 255.0f;
                Float g = data[(i + j * width) * 3 + 1] / 255.0f;
                Float b = data[(i + j * width) * 3 + 2] / 255.0f;
                Float color = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                pixels.push_back(color);
                pixelsRGB.emplace_back(r, g, b);
            }
        }

        min = glm::vec2(0, 0);
        max = glm::vec2(width, height);

        stbi_image_free(data);
    }

    static void setPixel(int x, int y, int w, int h, glm::vec3 const& color, std::vector<glm::vec3>& buffer)
    {
        if (x >= 0 && x < w && y >= 0 && y < h)
            buffer[y * w + x] = color;
    }

    void storeBufferToPPM(std::string const& output, std::vector<glm::vec3> const& buffer, Float gamma = 1.0f) const
    {
        std::cout << "Saving " << output << " ...\n";
        std::ofstream file(output);
        file << "P3\n" << width << " " << height << "\n255\n";

        // convert frame-buffer to range from 0 to 255
        for (int i = 0; i < height * width; ++i)
        {
            int r = int(std::pow(buffer[i].x, 1.0f / gamma) * 255.f);
            int g = int(std::pow(buffer[i].y, 1.0f / gamma) * 255.f);
            int b = int(std::pow(buffer[i].z, 1.0f / gamma) * 255.f);

            file << std::min(255, std::max(0, r)) << " " << std::min(255, std::max(0, g)) << " " << std::min(255, std::max(0, b)) << " ";
        }

        file.close();
    }

    std::vector<Float> pixels;
    std::vector<glm::vec3> pixelsRGB;
    int width;
    int height;
};

template<class TGenerator>
class IEnvironmentMap : public EnvironmentMap
{
public:
    IEnvironmentMap(std::string const& file_, glm::vec2 min, glm::vec2 max)
        : EnvironmentMap(file_, min, max), seq_id(0) { }

    glm::vec2 sample(AbstractImportanceBuffer<EnvironmentMap, TGenerator, glm::vec2>* buffer, TGenerator* gen)
    {
        return buffer->sample2D(seq_id, gen, xyRange, min);
    }

private:
    uint32_t seq_id;
};

} // namespace Importance

#endif // IMPORTANCE_DISTRIBUTIONS_2D_H

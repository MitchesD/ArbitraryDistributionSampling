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

#ifndef IMPORTANCE_TEST_SETUP_H__
#define IMPORTANCE_TEST_SETUP_H__

#include <is/ibuffer.h>
#include <iomanip>
#include <generators/lcg.h>
#include <generators/xorshift.h>
#include <shared/distributions_1d.h>
#include <is/grid.h>

namespace Importance
{

#define MSG_LENGTH 100

template<typename TGen, typename TBase, typename TDomain, typename TGrid = AbstractGrid<TBase, Constants::defaultVectorSize>>
class TestFactory
{
public:
    template<class... Args>
    explicit TestFactory(std::string name_, std::string output_, BufferType type,
                         Float historyPct = 0.5f, Args... args)
        : gen(new TGen()), name(std::move(name_)), output(std::move(output_))
    {
        printBegin();
        switch (type)
        {
            case BUFFER_RS:
                buffer = new RSBuffer<TBase, TGen, TDomain>();
                break;
            case BUFFER_RS_ADV:
                buffer = new RSAdvancedBuffer<TBase, TGen, TDomain>(historyPct);
                break;
            case BUFFER_RS_BOX:
                buffer = new RSWrappedAdvancedBuffer<TBase, TGen, TDomain, TGrid>(args...);
                break;
            case BUFFER_RS_BOX_PAV:
                buffer = new RSWrappedGridBuffer<TBase, TGen, TDomain, TGrid>(args...);
                break;
            case BUFFER_SET:
                buffer = new SetBuffer<TBase, TGen, TDomain>();
                break;
            default:
                break;
        }
    }

    ~TestFactory()
    {
        printEnd();
        delete gen;
        delete buffer;
    }

    void printBegin()
    {
        std::cout << std::setw(MSG_LENGTH) << std::setfill('=') << std::left << "<<<" << std::endl;
        std::cout << std::setw(MSG_LENGTH) << std::setfill('=') << std::left << name << std::endl;
        std::cout << std::setw(MSG_LENGTH) << std::setfill('=') << "" << std::endl;
    }

    void printEnd()
    {
        std::cout << std::setw(MSG_LENGTH) << std::setfill('=') << std::right << ">>>" << std::endl;
        std::cout << std::endl;
    }

    void dumpBufferSequence(std::string const& file, bool logSpace = false, bool masking = false) const
    {
        std::ofstream str(file + "-chain");
        auto chain = buffer->getChain();
        for (std::size_t i = 0; i < chain.size(); i++)
        {
            auto maskedChain = masking ? (chain[i] & bitMask) : chain[i];
            str << (logSpace ? (maskedChain == 0 ? 0 : std::log(maskedChain)) : maskedChain) << std::endl;
        }
        str.close();
    }

    TGen* gen;
    std::string name;
    std::string output;
    AbstractImportanceBuffer<TBase, TGen, TDomain>* buffer;
};

template<typename TGen, typename TBase, typename TDomain, typename TGrid = AbstractGrid<TBase, Constants::defaultVectorSize>>
class TestSetupUniversal : public TestFactory<TGen, TBase, TDomain, TGrid>
{
public:
    template<class... Args>
    TestSetupUniversal(TBase* base, std::string name_, std::string output_, uint32_t n,
                       BufferType bufferType, Float historyPct = 0.5f, Args... args)
        : TestFactory<TGen, TBase, TDomain, TGrid>(name_, output_, bufferType,
                                                   historyPct, args...)
    {
        InputData inputData = { };
        this->buffer->filterUniversal(n, base, this->gen, inputData);
        this->gen->reset();
    }
};

template<typename TGen, typename TBase, typename TDomain, typename TGrid = AbstractGrid<TBase, Constants::defaultVectorSize>>
class TestSetupSetUniversal : public TestFactory<TGen, TBase, TDomain, TGrid>
{
public:
    TestSetupSetUniversal(TBase* base, std::string name_, std::string output_, uint32_t n, uint32_t batch_size)
        : TestFactory<TGen, TBase, TDomain, TGrid>(name_, output_, BUFFER_SET)
    {
        InputData inputData;
        inputData.set.batch_size = batch_size;
        this->buffer->filterUniversal(n, base, this->gen, inputData);
        this->gen->reset();
    }
};

template <typename T, typename U, typename V>
inline T clamp(T val, U low, V high)
{
    if (val < low)
        return low;
    else if (val > high)
        return high;
    else
        return val;
}

inline void spherical_phi_theta(glm::vec3 const& d, float* phi, float* theta)
{
    *theta = std::acos(clamp(d.z, -1.0, 1.0));
    *phi = std::atan2(d.y, d.x);
}

template<typename T>
struct TypeName
{
    static const char* Get()
    {
        return typeid(T).name();
    }
};

template<>
struct TypeName<XorShiftGenerator>
{
    static const char* Get()
    {
        return "XorShift";
    }
};

template<>
struct TypeName<FastLCG>
{
    static const char* Get()
    {
        return "FastLCG";
    }
};

template<>
struct TypeName<NormalDistribution>
{
    static const char* Get()
    {
        return "NormalDistribution";
    }
};

template<>
struct TypeName<LandauDistribution>
{
    static const char* Get()
    {
        return "LandauDistribution";
    }
};

template<>
struct TypeName<HyperbolicSecantDistribution>
{
    static const char* Get()
    {
        return "HyperbolicSecantDistribution";
    }
};


} // namespace Importance

#endif // IMPORTANCE_TEST_SETUP_H__

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

#ifndef IMPORTANCE_DISTRIBUTIONS_API_H
#define IMPORTANCE_DISTRIBUTIONS_API_H

#include <string>
#include <fstream>
#include <is/ibuffer.h>
#include <is/grid.h>

namespace Importance
{

template<typename TGen, typename TBase, typename TDomain, typename TGrid = AbstractGrid<TBase, Constants::defaultVectorSize>>
class API
{
public:
    template<class... Args>
    API(BufferType type, Float errorPct = 0.5f, Args... args)
    {
        switch (type)
        {
            case BUFFER_RS:
                buffer = new RSBuffer<TBase, TGen, TDomain>();
                break;
            case BUFFER_RS_ADV:
                buffer = new RSAdvancedBuffer<TBase, TGen, TDomain>(errorPct);
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
        gen = new TGen();
    }

    ~API()
    {
        delete gen;
        delete buffer;
    }

    void prepare(uint64_t n, TBase* base)
    {
        InputData inputData = { };
        buffer->filterUniversal(n, base, gen, inputData);
        gen->reset();
    }

    void load(std::string const& filePath)
    {
        std::ifstream str(filePath, std::ios::binary);
        if (!str.is_open())
        {
            std::cerr << "Failed to open file '" << filePath << "'\n";
            throw;
        }

        str.unsetf(std::ios::skipws);
        std::streampos fileSize;
        str.seekg(0, std::ios::end);
        fileSize = str.tellg();
        str.seekg(0, std::ios::beg);

        std::vector<BufferUnit> vec;
        vec.reserve(fileSize / sizeof(BufferUnit));

        BufferUnit tmp{};
        for (uint32_t i = 0; i < fileSize / sizeof(BufferUnit); i++)
        {
            str.read(reinterpret_cast<char*>(&tmp), sizeof(BufferUnit));
            vec.push_back(tmp);
        }

        buffer->setChain(vec);
    }

    void save(std::string const& filePath)
    {
        auto data = buffer->getChain();

        std::ofstream str(filePath, std::ios::binary);
        if (!str.is_open())
        {
            std::cerr << "Failed to open file '" << filePath << "'\n";
            throw;
        }

        bool ok = data.empty() || str.write((char*) &data.front(), data.size() * sizeof(BufferUnit));
        if (!ok)
        {
            std::cerr << "Failed to write data into file\n";
            throw;
        }

        str.close();
    }

    TGen* getGenerator() { return gen; }
    AbstractImportanceBuffer<TBase, TGen, TDomain>* getBuffer() { return buffer; }

protected:
    TGen* gen;
    AbstractImportanceBuffer<TBase, TGen, TDomain>* buffer;
};

} // namespace Importance

#endif // IMPORTANCE_DISTRIBUTIONS_API_H

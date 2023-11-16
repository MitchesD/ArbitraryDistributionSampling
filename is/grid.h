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

#ifndef IMPORTANCE_DISTRIBUTIONS_GRID_H
#define IMPORTANCE_DISTRIBUTIONS_GRID_H

#include <vector>
#include <variant>
#include <shared/config.h>
#include <shared/distributions_1d.h>
#include <is/ivector.h>
#include <is/range.h>

namespace Importance
{

struct GridNode
{
    operator Float() const
    {
        return weight;
    }

    uint32_t id;
    // local function maximal value, for thresholding
    Float localMax;
    // spatial weight for combinatorics
    Float weight;
    // local function minimal value
    Float localMin;
};

template<typename TDistribution, int N = Constants::defaultVectorSize>
class AbstractGrid
{
public:
    explicit AbstractGrid(uint32_t nodeCount_) : nodeCount(nodeCount_), nodeVector(nullptr),
        heuristic(G_HEURISTIC_AREA) { }
    AbstractGrid() = default;
    virtual ~AbstractGrid() = default;

    virtual void prepare(TDistribution* /*distribution*/)
    {
        ASSERT_MESSAGE("", "AbstractGrid::prepare() not implemented");
    }

    virtual std::pair<GridRange, GridNode> sampleNode(Float const& /*r1*/)
    {
        ASSERT_MESSAGE("", "AbstractGrid::sampleNode() not implemented");
        return { GridRange{}, GridNode{} };
    }

    uint32_t getNodesCount() const
    {
        return nodeCount;
    }

    virtual GridRange getNodeGridRange(uint32_t /*nodeId*/)
    {
        ASSERT_MESSAGE("", "AbstractGrid::getNodeGridRange() not implemented");
        return {};
    }

    void setHeuristic(GridHeuristics heuristic_)
    {
        heuristic = heuristic_;
    }

    void clear()
    {
        nodes.clear();
        if (nodeVector)
        {
            nodeVector->clear();
            delete nodeVector;
            nodeVector = nullptr;
        }
    }

protected:
    uint32_t nodeCount;
    // index represents node ID
    std::vector<GridNode> nodes;
    StaticVector<GridNode, N>* nodeVector;

    GridHeuristics heuristic;
};

template<int N = Constants::defaultVectorSize>
class Grid1D : public AbstractGrid<Distribution1D, N>
{
public:
    Grid1D(uint32_t nodeCount_, Float min_, Float max_) : AbstractGrid<Distribution1D, N>(nodeCount_),
        min(min_), max(max_)
    {
        ASSERT(min < max);
        nodeRange = (max - min) / Float(this->nodeCount);
    }

    Float weightHeuristic(Float nodeMax, Float acc)
    {
        switch (this->heuristic)
        {
            case G_HEURISTIC_AREA:
                return acc / Constants::intervalSamples;
            case G_HEURISTIC_MAX:
                return nodeMax;
            default:
                return 0.0f;
        }
    }

    void prepare(Distribution1D* distribution) override
    {
        this->clear();
        Float norm = 0.0f;
        Float offset = min;

        // fill node list with sampled values over the domain intervals
        for (uint32_t i = 0; i < this->nodeCount; ++i)
        {
            GridNode node {i, 0.0f, 0.0f, std::numeric_limits<Float>::max() };
            Float acc = 0.0f;
            for (uint32_t j = 0; j < Constants::intervalSamples; ++j)
            {
                Float value = distribution->f(offset + Float(j) * nodeRange / Float(Constants::intervalSamples));
                node.localMax = std::max(node.localMax, value);
                node.localMin = std::min(node.localMin, value);
                acc += value;
            }
            node.weight = weightHeuristic(node.localMax, acc);
            offset += nodeRange;
            norm += node.weight;
            this->nodes.push_back(node);
        }

        // set node weight to be normalized value
        for (auto& node : this->nodes)
            node.weight = node.weight / norm;

#if DETAILED_DEBUG
        for (auto const& node : this->nodes)
            std::cout << node.weight << "\t\t" << node.localMax << std::endl;
#endif

        this->nodeVector = new StaticVector<GridNode, N>(this->nodes);
    }

    std::pair<GridRange, GridNode> sampleNode(Float const& r1) override
    {
        GridNode node = this->nodeVector->sample(r1);
        return { getNodeGridRange(node.id), node };
    }

    GridRange getNodeGridRange(uint32_t nodeId) override
    {
        Float const offset = min + nodeRange * Float(nodeId);
        RangeScalar range {offset, offset + nodeRange };
        return range;
    }

private:
    Float min;
    Float max;
    Float nodeRange;
};

template<int N = Constants::defaultVectorSize>
class GridCircular : public AbstractGrid<CircularDistribution, N>
{
public:
    explicit GridCircular(uint32_t nodeCount_) : AbstractGrid<CircularDistribution, N>(nodeCount_)
    {
        nodeRange = 2.0f * F_PI / Float(this->nodeCount);
    }

    Float weightHeuristic(Float nodeMax, Float acc)
    {
        switch (this->heuristic)
        {
            case G_HEURISTIC_AREA:
                return acc / Constants::intervalSamples;
            case G_HEURISTIC_MAX:
                return nodeMax;
            default:
                return 0.0f;
        }
    }

    void prepare(CircularDistribution* distribution) override
    {
        this->clear();
        Float norm = 0.0f;
        Float offset = 0.0f;

        // fill node list with sampled values over the domain intervals
        for (uint32_t i = 0; i < this->nodeCount; ++i)
        {
            GridNode node {i, 0.0f, 0.0f, std::numeric_limits<Float>::max() };
            Float acc = 0.0f;
            for (uint32_t j = 0; j < Constants::intervalSamples; ++j)
            {
                Float value = distribution->f(offset + Float(j) * nodeRange / Float(Constants::intervalSamples));
                node.localMax = std::max(node.localMax, value);
                node.localMin = std::min(node.localMin, value);
                acc += value;
            }
            node.weight = weightHeuristic(node.localMax, acc);
            offset += nodeRange;
            norm += node.weight;
            this->nodes.push_back(node);
        }

        // set node weight to be normalized value
        for (auto& node : this->nodes)
            node.weight = node.weight / norm;

#if DETAILED_DEBUG
        for (auto const& node : this->nodes)
            std::cout << node.weight << "\t\t" << node.localMax << std::endl;
#endif

        this->nodeVector = new StaticVector<GridNode, N>(this->nodes);
    }

    std::pair<GridRange, GridNode> sampleNode(Float const& r1) override
    {
        GridNode node = this->nodeVector->sample(r1);
        return { getNodeGridRange(node.id), node };
    }

    GridRange getNodeGridRange(uint32_t nodeId) override
    {
        Float const offset = nodeRange * Float(nodeId);
        RangeScalar range {offset, offset + nodeRange };
        return range;
    }

private:
    Float nodeRange;
};

template<int N = Constants::defaultVectorSize>
class GridSpherical : public AbstractGrid<SphericalDistribution, N>
{
public:
    GridSpherical(uint32_t phiNodes_, uint32_t thetaNodes_)
        : AbstractGrid<SphericalDistribution, N>(phiNodes_ * thetaNodes_),
        phiNodes(phiNodes_), thetaNodes(thetaNodes_)
    {
        Float phiRange = (2.0f * F_PI) / phiNodes;
        Float thetaRange = F_PI / thetaNodes;
        nodeRange = { phiRange, thetaRange };
    }

    Float weightHeuristic(Float nodeMax, Float acc)
    {
        switch (this->heuristic)
        {
            case G_HEURISTIC_AREA:
                return acc / (Constants::intervalSamples * Constants::intervalSamples);
            case G_HEURISTIC_MAX:
                return nodeMax;
            default:
                return 0.0f;
        }
    }

    void prepare(SphericalDistribution* distribution) override
    {
        // prepare spherical grid
        this->clear();
        Float norm = 0.0f;
        glm::vec2 offset = { 0.0f, 0.0f };

        uint32_t nodeId = 0;
        for (uint32_t j = 0; j < thetaNodes; j++)
        {
            offset.x = 0;
            for (uint32_t i = 0; i < phiNodes; i++)
            {
                GridNode node {nodeId++, 0.0f, 0.0f, std::numeric_limits<Float>::max() };
                Float acc = 0.0f;

                for (uint32_t x = 0; x < Constants::intervalSamples; x++)
                {
                    for (uint32_t y = 0; y < Constants::intervalSamples; y++)
                    {
                        Float phi = offset.x + Float(x) * nodeRange.x / Constants::intervalSamples;
                        Float theta = offset.y + Float(y) * nodeRange.y / Constants::intervalSamples;

                        Float value = distribution->f(glm::vec3(std::cos(phi) * std::sin(theta),
                                                                std::sin(phi) * std::sin(theta),
                                                                std::cos(theta)));
                        node.localMax = std::max(node.localMax, value);
                        node.localMin = std::min(node.localMin, value);
                        acc += value;
                    }
                }

                offset.x += nodeRange.x;
                node.weight = weightHeuristic(node.localMax, acc);
                norm += node.weight;
                this->nodes.push_back(node);
            }
            offset.y += nodeRange.y;
        }

        // set node weight to be normalized value
        for (auto& node : this->nodes)
            node.weight = node.weight / norm;

#if DETAILED_DEBUG
        std::cout << "norm: " << norm << std::endl;
        for (auto const& node : this->nodes)
            std::cout << node.weight << "\t\t" << node.localMax << std::endl;
#endif

        this->nodeVector = new StaticVector<GridNode, N>(this->nodes);
    }

    std::pair<GridRange, GridNode> sampleNode(Float const& r1) override
    {
        GridNode node = this->nodeVector->sample(r1);
        return { getNodeGridRange(node.id), node };
    }

    GridRange getNodeGridRange(uint32_t nodeId) override
    {
        uint32_t col_offset = nodeId % phiNodes;
        uint32_t row_offset = nodeId / phiNodes;
        glm::vec2 const offset = glm::vec2(nodeRange.x * Float(col_offset), nodeRange.y * Float(row_offset));
        RangeVector range {offset, offset + nodeRange };
        return range;
    }

private:
    glm::vec2 nodeRange;

    uint32_t phiNodes;
    uint32_t thetaNodes;
};

template<int N = Constants::defaultVectorSize>
class Grid2D : public AbstractGrid<Distribution2D, N>
{
public:
    Grid2D(uint32_t xNodes_, uint32_t yNodes_, glm::vec2 min_, glm::vec2 max_)
        : AbstractGrid<Distribution2D, N>(xNodes_ * yNodes_),
          xNodes(xNodes_), yNodes(yNodes_), min(min_), max(max_)
    {
        nodeRange = (max - min) / glm::vec2(xNodes, yNodes);
    }

    virtual ~Grid2D() = default;

    Float weightHeuristic(Float nodeMax, Float acc)
    {
        switch (this->heuristic)
        {
            case G_HEURISTIC_AREA:
                return acc / (Constants::intervalSamples * Constants::intervalSamples);
            case G_HEURISTIC_MAX:
                return nodeMax;
            default:
                return 0.0f;
        }
    }

    void prepare(Distribution2D* distribution) override
    {
        // prepare 2D grid
        this->clear();
        Float norm = 0.0f;
        glm::vec2 offset = { min.x, min.y };

        uint32_t nodeId = 0;
        for (uint32_t j = 0; j < yNodes; j++)
        {
            offset.x = min.x;
            for (uint32_t i = 0; i < xNodes; i++)
            {
                GridNode node { nodeId++, 0.0f, 0.0f, std::numeric_limits<Float>::max() };
                Float acc = 0.0f;

                for (uint32_t x = 0; x < Constants::intervalSamples; x++)
                {
                    for (uint32_t y = 0; y < Constants::intervalSamples; y++)
                    {
                        Float currentX = offset.x + Float(x) * nodeRange.x / Constants::intervalSamples;
                        Float currentY = offset.y + Float(y) * nodeRange.y / Constants::intervalSamples;

                        Float value = distribution->f(glm::vec2(currentX, currentY));
                        node.localMax = std::max(node.localMax, value);
                        node.localMin = std::min(node.localMin, value);
                        acc += value;
                    }
                }

                node.weight = weightHeuristic(node.localMax, acc);
                norm += node.weight;
                offset.x += nodeRange.x;
                this->nodes.push_back(node);
            }
            offset.y += nodeRange.y;
        }

        // set node weight to be normalized value
        for (auto& node : this->nodes)
            node.weight = node.weight / norm;

#if DETAILED_DEBUG
        std::cout << "norm: " << norm << std::endl;
        for (auto const& node : this->nodes)
            std::cout << node.weight << "\t\t" << node.localMax << std::endl;
#endif

        this->nodeVector = new StaticVector<GridNode, N>(this->nodes);
    }

    std::pair<GridRange, GridNode> sampleNode(Float const& r1) override
    {
        GridNode node = this->nodeVector->sample(r1);
        return { getNodeGridRange(node.id), node };
    }

    GridRange getNodeGridRange(uint32_t nodeId) override
    {
        uint32_t col_offset = nodeId % xNodes;
        uint32_t row_offset = nodeId / xNodes;
        glm::vec2 const offset = glm::vec2(min.x + nodeRange.x * Float(col_offset), min.y + nodeRange.y * Float(row_offset));
        RangeVector range {offset, offset + nodeRange };
        return range;
    }

private:
    glm::vec2 nodeRange;
    uint32_t xNodes;
    uint32_t yNodes;
    glm::vec2 min;
    glm::vec2 max;
};

} // namespace Importance

#endif // IMPORTANCE_DISTRIBUTIONS_GRID_H

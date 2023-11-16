# Fast and Robust Arbitrary Distribution Sampling for Rendering Applications

## Abstract
This paper proposes a novel, fast and precise sampling algorithm for arbitrary n-dimensional distribution functions. The proposed algorithm is faster than any of common methods and its precision and fitting to the desired distribution is comparable to the state-of-the-art. The proposed algorithm is not limited to subset of distributions for which the inverse cumulative distribution function is known and is faster than other general state-of-the-art methods. The main idea is based on skipping unfitting samples from a uniform pseudo-random number generator to make it proportional to the sampled probability density.
The function is split into regions where each part utilizes a domain-limited rejection sampling that filters out unimportant states and counts the number of skips and stores them into a structure called a skipping sequence. Skipping utilizes a compact precomputed acceleration structure. This structure is represented as an N-D grid. The sampling from such a moderated generator selects the cell of the grid and jumps ahead a few states, according to a skipping sequence. We propose two variants that slightly differ in the cell selection approach. The skipping sequence can be computed just once, effectively stored on a drive and re-used whenever afterwards. The algorithm has imminent usage in computer graphics applications, and some of them are shown in this manuscript.# Authors
### Michal Vlnas<sup>1</sup>
* ivlnas@fit.vutbr.cz

### Pavel Zemčík<sup>1,2</sup>
* zemcik@fit.vutbr.cz
* Pavel.Zemcik@lut.fi

### Tomáš Milet<sup>1</sup>
* imilet@fit.vutbr.cz

<sup>1</sup>Brno Unviersity of Technology,
Faculty of Information Technology, Czech Republic

<sup>2</sup>Lappeenranta-Lahti University of Technology, School of Engineering Science, Finland

# Requirements
* C++20 compiler
* CMake 3.16 or higher
* Boost 1.58 or higher
* GLM
* NTL

# Notes
* The CLRS variant proposed in the paper is implemented in the class `RSWrappedAdvancedBuffer`
* And the DLRS is implemented in the class `RSWrappedGridBuffer`
* There are 3 more variants implemented in the project:
  * `RSBuffer` - the global rejection sampling with skipping sequence, it produces samples 1:1 with native RS
  * `RSAdvancedBuffer` - modified version of the previous method with skipping sequences average step size tuning parameters
  * `SetBuffer` - an experimental buffer based on sets, it is not recommended to use it, it is just for experimental purposes
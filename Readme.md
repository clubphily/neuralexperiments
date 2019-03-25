# Some neural networks in Julia
As a repetition of my neural networks skills and as a preparation for an experiment with image processing, I decided to implement a few simple neural networks from scratch. All examples are written in [Python](https://www.python.org/) and I will port the code to [Julia](https://julialang.org/). The semantic differences between [numpy](http://www.numpy.org/) and [Julia](https://julialang.org/) are just big enough to make you understand the mathematics actually happening within the code.

## A super simple beginning
A good entry point into all that matrix and vector math again is [this very light nine liner](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1) written in python. It helps understand how matrixes and horizontal or vertical vectors are initialised and it contains enough ['.'](https://docs.julialang.org/en/v1/manual/mathematical-operations/#man-dot-operators-1) operations to get a first feeling of the power of [Julia](https://julialang.org/)'s briefness. The example also shows that [dot in numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) is not like [dot in Julia](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#LinearAlgebra.dot). There is an additional [* operator in Julia](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#Base.:*-Tuple{AbstractArray{T,2}%20where%20T,AbstractArray{T,2}%20where%20T}) for multiplying a [x, n] matrix with a [x] vector. This functionality is covered by [dot in numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html).
[The example](./supersimple.jl) was kept super-brief as it looks good. In the next example I will try out how functions work in [Julia](https://julialang.org/).

## Adding some hidden layers
Another [rather light example](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6). It was interesting to work with functions and still keep the script character by using global variables in [Julia](https://julialang.org/). The example itself is a nice introduction into back-propagation and cost functions. In respect of [Julia](https://julialang.org/) syntax, a nice feature is the usage of ['.'](https://docs.julialang.org/en/v1/manual/mathematical-operations/#man-dot-operators-1) for any function. On the other handside the usage of * vs. .* is not necessarily intuitive but I guess one gets used to it.

## More hidden layers
Ported [this code](https://medium.freecodecamp.org/building-a-3-layer-neural-network-from-scratch-99239c4af5d3). It was interesting to dive deeper into [Julia](https://julialang.org/)'s abyss. A interesting addition was how to load [DataFrames](https://juliadata.github.io/DataFrames.jl/stable/index.html) from CSV files and using [MLMetrics](https://github.com/JuliaML/MLMetrics.jl) to calculate model accuracy.

What was also astonishing but not new is how strongly accuracy is influenced by the selection of the right randomised set of initial parameters for each layer. A few plots for a different initialisation of MersenneTwister are below. MersenneTwister(1024) even converges to 100% towards the end.

![MersenneTwister(0)](./images/0_accuarcy.svg)
![MersenneTwister(16)](./images/16_accuarcy.svg)
![MersenneTwister(42)](./images/42_accuarcy.svg)
![MersenneTwister(64)](./images/64_accuarcy.svg)
![MersenneTwister(92)](./images/92_accuarcy.svg)
![MersenneTwister(512)](./images/512_accuarcy.svg)
![MersenneTwister(1024)](./images/1024_accuarcy.svg)

## Apply some real images
As a last experiment a compression algorithm will be implemented based on [this idea](https://hackernoon.com/using-ai-to-super-compress-images-5a948cf09489).

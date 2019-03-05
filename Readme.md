# Some neural networks in Julia
As a repetition of my neural networks skills and as a preparation for an experiment with image processing, I decided to implement a few simple neural networks from scratch. All examples are written in [Python](https://www.python.org/) and I will port the code to [Julia](https://julialang.org/). The semantic differences between [numpy](http://www.numpy.org/) and [Julia](https://julialang.org/) are just big enough to make you understand the mathematics actually happening within the code.

## A super simple beginning
A good entry point into all that matrix and vector math again is [this very light nine liner](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1) written in python. It helps understand how matrixes and horizontal or vertical vectors are initialised and it contains enough ['.'](https://docs.julialang.org/en/v1/manual/mathematical-operations/#man-dot-operators-1) operations to get a first feeling of the power of [Julia](https://julialang.org/)'s briefness. The example also shows that [dot in numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) is not like [dot in Julia](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#LinearAlgebra.dot). There is an additional [* operator in Julia](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#Base.:*-Tuple{AbstractArray{T,2}%20where%20T,AbstractArray{T,2}%20where%20T}) for multiplying a [x, n] matrix with a [x] vector. This functionality is covered by [dot in numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html).
[The example](./supersimple.jl) was kept super-brief as it looks good. In the next example I will try out how functions work in [Julia](https://julialang.org/).

## Adding some hidden layers
This will be a port of [this code](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6).

## More hidden layers
This will be a port of [this code](https://medium.freecodecamp.org/building-a-3-layer-neural-network-from-scratch-99239c4af5d3).

## Apply some real images
As a last experiment a compression algorithm will be implemented based on [this idea](https://hackernoon.com/using-ai-to-super-compress-images-5a948cf09489).

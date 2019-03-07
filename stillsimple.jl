using Random, LinearAlgebra

function sigmoid(t)
    return 1 / (1 + exp(-t))
end

function sigmoid_derivative(p)
    return p * (1 - p)
end

function feedforward()
    global layer_1 = sigmoid.(X * weights_1)
    global output = sigmoid.(layer_1 * weights_2)
end

function backprop()
    l_weights_2 = transpose(layer_1) * (2 .* (y .- output) .* sigmoid_derivative.(output))
    l_weights_1 = transpose(X) * (((2 * (y .- output) .* sigmoid_derivative.(output)) * transpose(weights_2)) .* sigmoid_derivative.(layer_1))

    global weights_2 .+= l_weights_2
    global weights_1 .+= l_weights_1
end

X = [0 0 1; 0 1 1; 1 0 1; 1 1 1]
y = [0, 1, 1, 0]
rnd = MersenneTwister(42)
weights_1 = rand(rnd, 3, 4) #dimension of X
weights_2 = rand(rnd, 4, 1) #dimension of output, respectively y
output = [0, 0, 0, 0] #length of y

for i = 1:1500
    if i % 100 == 0
        print("\nfor iteration ")
        print(i)
        print("\n  hidden layer has weighed in at: \n  ")
        println(layer_1)
        print("  output has weighed in at: \n  ")
        println(output)
    end
    feedforward()
    backprop()
end

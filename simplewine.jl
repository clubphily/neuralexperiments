using CSV, Random, LinearAlgebra, DataFrames, MLMetrics

function initialize_parameters(nn_input_dim, nn_hidden_dim, nn_output_dim)
    W1 = 2 * randn(rnd, Float64, (nn_input_dim, nn_hidden_dim)) .- 1
    b1 = zeros(1, nn_hidden_dim)
    W2 = 2 * randn(rnd, Float64, (nn_hidden_dim, nn_hidden_dim)) .- 1
    b2 = zeros(1, nn_hidden_dim)
    W3 = 2 * randn(rnd, Float64, (nn_hidden_dim, nn_output_dim)) .- 1
    b3 = zeros(1, nn_output_dim)
    model = Dict("W1"=>W1, "b1"=>b1, "W2"=>W2, "b2"=>b2, "W3"=>W3, "b3"=>b3)
    return model
end

function softmax_loss(y, y_hat)
    m = size(y, 1)
    loss = -1/m * sum(y .* log.(y_hat))
    return loss
end

function softmax(z)
    exp_scores = exp.(z)
    return exp_scores ./ sum(exp_scores, dims=2)
end

function loss_derivative(y, y_hat)
    return (y_hat - y)
end

function tanh_derivative(x)
    return (1 - x^2)
end

function forward_prop(model, a0)
    W1, b1, W2, b2, W3, b3 = model["W1"], model["b1"], model["W2"], model["b2"], model["W3"], model["b3"]
    z1 = (a0 * W1) .+ b1
    a1 = tanh.(z1)
    z2 = (a1 * W2) .+ b2
    a2 = tanh.(z2)
    z3 = (a2 * W3) .+ b3
    a3 = softmax(z3)
    cache = Dict("a0"=>a0, "z1"=>z1, "a1"=>a1, "z2"=>z2, "a2"=>a2, "a3"=>a3, "z3"=>z3)
    return cache
end

function backward_prop(model, cache, y)
    W1, b1, W2, b2, W3, b3 = model["W1"], model["b1"], model["W2"], model["b2"], model["W3"], model["b3"]
    a0, a1, a2, a3 = cache["a0"], cache["a1"], cache["a2"], cache["a3"]
    m = size(y, 1)
    dz3 = loss_derivative.(y, a3)
    dW3 = 1/m .* (a2' * dz3)
    db3 = 1/m .* sum(dz3, dims=1)
    dz2 = (dz3 * W3') .* tanh_derivative.(a2)
    dW2 = 1/m .* (a1' * dz2)
    db2 = 1/m .* sum(dz2, dims=1)
    dz1 = (dz2 * W2') .* tanh_derivative.(a1)
    dW1 = 1/m .* (a0' * dz1)
    db1 = 1/m .* sum(dz1, dims=1)
    grads = Dict("dW3"=>dW3, "db3"=>db3, "dW2"=>dW2,"db2"=>db2,"dW1"=>dW1,"db1"=>db1)
    return grads
end

function update_parameters(model, grads, learning_rate)
    W1, b1, W2, b2, b3, W3 = model["W1"], model["b1"], model["W2"], model["b2"],model["b3"],model["W3"]
    W1 .-= learning_rate .* grads["dW1"]
    b1 .-= learning_rate .* grads["db1"]
    W2 .-= learning_rate .* grads["dW2"]
    b2 .-= learning_rate .* grads["db2"]
    W3 .-= learning_rate .* grads["dW3"]
    b3 .-= learning_rate .* grads["db3"]
    model = Dict("W1"=>W1, "b1"=>b1, "W2"=>W2, "b2"=>b2, "W3"=>W3, "b3"=>b3)
    return model
end

function train(model, X_, y_, learning_rate, epochs, print_loss)
    for i = 1:epochs
        cache = forward_prop(model, X_)
        grads = backward_prop(model, cache, y_)
        model = update_parameters(model, grads, learning_rate)
        if print_loss && i % 40 == 0
            a3 = cache["a3"]
            println("Loss after iteration $(i): $(softmax_loss(y_, a3))")
            y_hat = argmax(a3, dims=2)
            y_true = argmax(y_, dims=2)
            println("Accuracy after iteration $(i): $(accuracy(y_hat, y_true)*100)%")
            append!(losses, accuracy(y_hat, y_true)*100)
        end
    end
end

wine_df = CSV.read("./data/W1data.csv")
y = convert(Matrix, wine_df[[:Cultivar_1, :Cultivar_2, :Cultivar_3]])
X = convert(Matrix, wine_df[:, 1:13])
losses = zeros(0)
rnd = MersenneTwister(64)
model = initialize_parameters(13, 5, 3)
model = train(model, X, y, 0.07, 4500, true)

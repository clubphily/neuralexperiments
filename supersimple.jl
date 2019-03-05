using Random, LinearAlgebra
training_set_inputs = [0 0 1; 1 1 1; 1 0 1; 0 1 1]
training_set_outputs = [0, 1, 1, 0]
synaptic_weights = 2 * rand(MersenneTwister(42), 3, 1) .- 1
for i = 1:10000
    output = 1 ./ (1 .+ exp.(-(training_set_inputs * synaptic_weights)))
    synaptic_weights .+= transpose(training_set_inputs) * ((training_set_outputs .- output) .* output .* (1 .- output))
end
println(1 ./ (1 .+ exp.(-([1 0 0] * synaptic_weights))))

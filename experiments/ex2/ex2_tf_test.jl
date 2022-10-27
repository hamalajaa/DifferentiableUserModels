using BSON
using Distributions
using Flux
using Stheno
using Tracker

include("../../NeuralProcesses.jl/src/NeuralProcesses.jl")
using .NeuralProcesses

using Printf
using Statistics

using Transformers
using Transformers.Basic

# Redundant. Required to fit the DataGenerator definition
x_context = Distributions.Uniform(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

num_context = Distributions.DiscreteUniform(10, 10)
num_target  = Distributions.DiscreteUniform(10, 10)

bias = 0.0

args = Dict("gen" => "menu_search", "n_traj" => 10, "params" => false, "p_bias" => 0.0, "perm" => false)

batch_size  = 1

x_context = Distributions.Uniform(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

num_context = Distributions.DiscreteUniform(50, 50)
num_target  = Distributions.DiscreteUniform(50, 50)

data_gen = NeuralProcesses.DataGenerator(
             SearchEnvSampler(args;),
			 batch_size=batch_size,
			 x_context=x_context,
			 x_target=x_target,
			 num_context=num_context,
			 num_target=num_target,
			 σ²=1e-8
			)

m = BSON.load("models/ex2/transformer_v1/1000.bson")[:model]

function pdf(xc, yc, xt, yt)

    c_size = size(yt)[2]

    xs = cat(xc, xt, dims=2)
    ys = cat(yc, yt, dims=2)

    probs = m(xs)[:,end-c_size+1:end,:]
    pds = maximum(yt .* probs, dims=1)
    return pds
end

n_batches = 32

# Generate evaluation data
@time data = gen_batch(data_gen, n_batches; eval=false)

# Loop over the number of trajectories
for i in 0:9

    # Init a list for processed data
    d = []

    # Loop over data batches
    for j in 1:n_batches
        
        # Manually split data into context and target sets

        xc = permutedims(data[j][1][:,:,1:1], [2,1,3]) 
        xt = permutedims(data[j][3][:,:,1:1], [2,1,3])

        yc = data[j][2][:,:,1:1] 
        yt = data[j][4][:,:,1:1]

	    start_idx = 5*(9-i)+1
	    push!(d, (xc[:,start_idx:end,:], yc[:,start_idx:end,:], xt, yt))

    end

    pds_e = []

    for j in 1:n_batches
        xc, yc, xt, yt = d[j]
	    push!(pds_e, pdf(xc, yc, xt, yt)...)
    end
    
    mean_e = Statistics.mean(pds_e)[1]							
    ci_e   = 2std(pds_e)[1] / sqrt(length(pds_e))
                                                
    @printf("Likelihood at %d context trajectories", i)
    @printf(
        "	%8.3f +- %7.3f (%d batches)\n",
        mean_e,
        ci_e,
        n_batches
    )

end






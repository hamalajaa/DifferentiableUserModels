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

args = Dict("gen" => "gridworld", "n_traj" => 10, "params" => false, "bias" => 0.0, "perm" => false)

batch_size  = 1

x_context = Distributions.Uniform(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

num_context = Distributions.DiscreteUniform(50, 50)
num_target  = Distributions.DiscreteUniform(50, 50)

data_gen = NeuralProcesses.DataGenerator(
                MCTSPlanner(args;),
                batch_size=batch_size,
                x_context=x_context,
                x_target=x_target,
                num_context=num_context,
                num_target=num_target,
                σ²=1e-8
	    )

function pdf(data, m)
    x, y = data
    probs = m(x)
    pds = maximum(y .* probs, dims=1)
    return pds
end

function eval_model(mdl, opt=Descent(0.01), updates=32)
    weights = Flux.params(mdl)
    prev_weights = deepcopy(weights)

    @time d1 = gen_batch(data_gen, 4; eval=false)
	        	      		
    tasks = construct_batch(d1)
    xc, yc, xt, yt = tasks[1]

    for i in 1:updates
        grad = Flux.gradient(weights) do
            Flux.crossentropy(m(xc), yc)
        end
        Flux.Optimise.update!(opt, weights, grad)
    end

    pds = maximum(yt .* mdl(xt), dims=1)

    Flux.loadparams!(mdl, prev_weights)

    return pds

end

n_batches = 16

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

	    start_idx = 10*(9-i)+1
	    push!(d, (xc[:,start_idx:end,:], yc[:,start_idx:end,:], xt, yt))

    end

    pds_v = []
    pds_e = []

    for j in 1:n_batches

        m = BSON.load("models/ex1/maml_v2/2000.bson")[:model]
        θ = Flux.params(m)

        θ_prev = deepcopy(θ)

        xc, yc, xt, yt = d[j]
        task1 = (xc, yc)
        task2 = (xt, yt)
        
        batchopt = Descent(0.01)

        probs = m(xt)
        pds = maximum(yt .* probs, dims=1)

	    push!(pds_v, pds)

        @time begin

            for i in 1:32

                loss1(x, y) = Flux.crossentropy(m(x), y)

                grad = Flux.gradient(θ) do
                    loss1(xc, yc)
                end
            
                for θ_i in θ
                    θ_i .-= Flux.Optimise.apply!(batchopt, θ_i, grad[θ_i])
                end

            end

            c_size = size(xc)[2]
            xs = cat(xc, xt, dims=2)

            probs = m(xs)[:,c_size:end,:]
            pds = maximum(yt .* probs, dims=1)

        end

	    push!(pds_e, pds...)
    end
    
    mean_e = Statistics.mean(pds_e)[1]							
    ci_e   = 2std(pds_e)[1] / sqrt(length(pds_e))

    mean_v = Statistics.mean(pds_v)[1]							
    ci_v   = 2std(pds_v)[1] / sqrt(length(pds_v))
                                                
    @printf("Likelihood at %d context trajectories", i)
    @printf(
        "	%8.3f +- %7.3f (%d batches)\n",
        mean_e,
        ci_e,
        n_batches
    )
    @printf(
        "	%8.3f +- %7.3f (%d batches)\n",
        mean_v,
        ci_v,
        n_batches
    )

end






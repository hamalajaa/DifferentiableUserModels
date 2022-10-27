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

using POMDPs
using POMDPModels
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using CommonRLInterface

batch_size  = 4

# Redundant. Required to fit the DataGenerator definition
x_context = Distributions.Uniform(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

num_context = Distributions.DiscreteUniform(10, 10)
num_target  = Distributions.DiscreteUniform(10, 10)

bias = 0.0

args = Dict("gen" => "h_menu_search", "n_traj" => 1, "variant" => "v4", "params" => false, "p_bias" => 0.0, "perm" => false)

data_gen = NeuralProcesses.DataGenerator(
               HierarchicalMenuSampler(args;),
               batch_size=batch_size,
               x_context=x_context,
               x_target=x_target,
               num_context=num_context,
               num_target=num_target,
               σ²=1e-8
           )

function construct_batch(d)

    tasks = []

    for data in d

        xc = permutedims(data[1][:,:,1:1], [2,1,3]) 
        xt = permutedims(data[3][:,:,1:1], [2,1,3])

        yc = data[2][:,:,1:1] 
        yt = data[4][:,:,1:1]

        push!(tasks, (xc, yc, xt, yt))

    end

    return tasks

end

dim_hidden = 2048
dim_x = 17
dim_y = 9

m = Chain(Dense(dim_x, 128, relu),
          Transformers.Transformer(128, 8, 128, dim_hidden),
	      Dense(128, 128, relu),
		  Dense(128, 128, relu),
          Dense(128, 128, relu),
          Dense(128, 128, relu),
          Dense(128, 128, relu),
          Dense(128, dim_y),
	      Flux.softmax)

θ = Flux.params(m)

function loss(data)
    x, y = data
    probs = m(x)
    l = Flux.crossentropy(probs, y)
    return l
end

function pdf(data)
    x, y = data
    probs = m(x)
    pds = maximum(y .* probs, dims=1)
    return pds
end

opt = ADAM(5e-4)

n_epochs = 2000

means_t = []
cis_t   = []

means_e = []
cis_e   = []

grads = IdDict()

for e in 0:n_epochs

    @printf("Epoch:  %-6d\n", e)
    if (e % 20) == 0
       	bson("models/ex3/maml_v1/"*string(e)*".bson", model=m)
    end

    @time d1 = gen_batch(data_gen, 4; eval=false)
	        	      		
    tasks = construct_batch(d1)
    
    θ_prev = deepcopy(θ)

    metaopt = Descent(5e-3)
	
    for (t, task) in enumerate(tasks)
        
        batchopt = Descent(5e-3)

        xc, yc, xt, yt = task
        task1 = (xc, yc)
        task2 = (xt, yt)

        loss1(x, y) = Flux.crossentropy(m(x), y)

        grad = Flux.gradient(θ) do
            loss1(xc, yc)
        end
    
        for θ_i in θ
            θ_i .-= Flux.Optimise.apply!(batchopt, θ_i, grad[θ_i])
        end

        grad = Flux.gradient(θ) do
            loss1(xt, yt)
        end

        for (θ_1, θ_2) in zip(θ, θ_prev)
            θ_1 .= θ_2
            w = get!(grads, θ_2, zero(θ_2))
            w .+= grad[θ_1]
        end

        pds_e = pdf(task2)
        pds_t = pdf(task1)
					       					        
        mean_t = Statistics.mean(pds_t)[1]					
        ci_t   = 2std(pds_t)[1] / sqrt(length(pds_t))
                                                        
        push!(means_t, mean_t)							
        push!(cis_t, ci_t)
                                                    
        mean_e = Statistics.mean(pds_e)[1]							
        ci_e   = 2std(pds_e)[1] / sqrt(length(pds_e))
                                                
        push!(means_e, mean_e)					
        push!(cis_e, ci_e)
        
        @printf("PDF train:    %8.3f +- %7.3f\n",					
		    mean_t,										
		    ci_t,
		    )
											        
	
        @printf("PDF eval:    %8.3f +- %7.3f\n",
            mean_e,				
            ci_e,
            )	
			
    end

    for (θ_1, θ_2) in zip(θ, θ_prev)
        Flux.Optimise.update!(metaopt, θ_1, grads[θ_2])
    end

end

bson("results/ex3/maml_v1.bson", means_t=means_t, cis_t=cis_t, 
     				 means_e=means_e, cis_e=cis_e)

@printf("Final results:		%8.3f +- %7.3f\n",
	Statistics.mean(means_e),
	2*Statistics.std(means_e)[1] / sqrt(length(means_e)),
	)
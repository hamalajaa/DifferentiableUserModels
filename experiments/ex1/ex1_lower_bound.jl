using BSON
using Distributions
using Flux
using Stheno
using Tracker

include("../../NeuralProcesses.jl/src/NeuralProcesses.jl")
using .NeuralProcesses

using Printf
using Statistics

batch_size  = 4

# Redundant. Required to fit the DataGenerator definition
x_context = Distributions.Uniform(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

num_context = Distributions.DiscreteUniform(10, 10)
num_target  = Distributions.DiscreteUniform(10, 10)

bias = 0.0

args1 = Dict("gen" => "gridworld", "n_traj" => 10, "params" => false, "bias" => 0.0, "perm" => false)

data_gen = NeuralProcesses.DataGenerator(MCTSPlanner(args1;),
			 batch_size=batch_size,
			 x_context=x_context,
			 x_target=x_target,
			 num_context=num_context,
			 num_target=num_target,
			 σ²=1e-8
			)

function construct_batch(data)

    xs = cat(permutedims(data[1][:,:,1:1], [2,1,3]), 
             permutedims(data[3][:,:,1:1], [2,1,3]), 
             dims=2)
    ys = cat(data[2][:,:,1:1], 
	         data[4][:,:,1:1],
	         dims=2);
	
    return (xs, ys)

end

m = Chain(Dense(2, 128, relu),
          Dense(128, 128, relu),
          Dense(128, 128, relu),
          Dense(128, 128, relu),
          Dense(128, 128, relu),
          Dense(128, 128, relu),
	      Dense(128, 5),
	      softmax)

function loss(data)
    x, y = data
	# Reset hidden state between tasks
    Flux.reset!(m)
    probs = m(x)
    l = Flux.crossentropy(probs, y)
    if isnan(l)
        l = Tracker.track(identity, 0f0)
    end
    return l
end

function pdf(data)
    x, y = data
	# Reset hidden state between tasks
    Flux.reset!(m)
    probs = m(x)
    pds = maximum(y .* probs, dims=1)
    return pds
end

ps = Flux.params(m)

opt = ADAM(5e-4)

n_epochs = 10000

means_t = []
cis_t   = []

means_e = []
cis_e   = []

for e in 1:n_epochs

    if (e % 1000) == 0
       	bson("models/ex1/mlp_v1/"*string(e)*".bson", model=m)
    end

    @time d1 = gen_batch(data_gen, 1; eval=false)[1]
	@time d2 = gen_batch(data_gen, 1; eval=false)[1]
	        	      		
    train_data = construct_batch(d1)
	eval_data = construct_batch(d2)
				        			
    grads = Flux.gradient(ps) do
		loss(train_data)
	end

    Flux.Optimise.update!(opt, ps, grads)
				        				
    pds_t = pdf(train_data)
	pds_e = pdf(eval_data)
					       					        
	mean_t = Statistics.mean(pds_t)[1]					
	ci_t   = 2std(pds_t)[1] / sqrt(length(pds_t))
							        				
	push!(means_t, mean_t)							
	push!(cis_t, ci_t)
								        		
	mean_e = Statistics.mean(pds_e)[1]							
	ci_e   = 2std(pds_e)[1] / sqrt(length(pds_e))
											
	push!(means_e, mean_e)					
	push!(cis_e, ci_e)										        
			
	@printf("Epoch:  %-6d\n", e)

	
	@printf("PDF train:    %8.3f +- %7.3f\n",					
		mean_t,										
		ci_t,
		)
											        
	
	@printf("PDF eval:    %8.3f +- %7.3f\n",
		mean_e,				
		ci_e,
		)	

end

bson("results/ex1/mlp_v1.bson", means_t=means_t, cis_t=cis_t, 
     				 means_e=means_e, cis_e=cis_e)

@printf("Final results:		%8.3f +- %7.3f\n",
	Statistics.mean(means_e),
	2*Statistics.std(means_e)[1] / sqrt(length(means_e)),
	)











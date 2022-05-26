using ArgParse
using BSON
using Distributions
using Flux
using Stheno
using Tracker
using Statistics
using Printf

include("../../NeuralProcesses.jl/src/NeuralProcesses.jl")
include("../../NeuralProcesses.jl/src/experiment/experiment.jl")

using .NeuralProcesses
using .NeuralProcesses.Experiment

parser = ArgParseSettings()
@add_arg_table! parser begin
    "--gen"
        help = "Experiment setting: gridworld, menu_search, h_menu_search"
        arg_type = String
        default = "menu_search"
    "--n_traj"
        help = "Number of context trajectories. Should be fixed to 10 for testing."
        arg_type = Int
        default = 10
    "--params"
        help = "Return params?"
        arg_type = Bool
        default = false
    "--p_bias"
        help = "Probability of generating a sample with biased model"
        arg_type = Float64
        default = 0.0
    "--batch_size"
        help = "Batch size."
        arg_type = Int
        default = 8
    "--bson"
        help = "Directly specify the file to save the model to and load it from."
        arg_type = String
end
args = parse_args(parser)

model = NeuralProcesses.Experiment.recent_model("models/ex2/anp_v1/109.bson") |> gpu

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

likelihoods(xs...) = NeuralProcesses.likelihood(
                xs...,
                target=true,
                num_samples=5,
                fixed_σ_epochs=3
           )

n_batches = 64

# Generate evaluation data
@time data = gen_batch(data_gen, n_batches)

# Loop over the number of trajectories
for i in 0:9

    # Init a list for processed data
    d = []

    # Loop over data batches
    for j in 1:n_batches
						            
        # Manually split data into context and target sets
	    xc, yc, xt, yt = data[j]
    
        start_idx = 5*(9-i)+1

        push!(d, [xc[start_idx:end,:,:], yc[:,start_idx:end,:], xt, yt])

    end

    tuples = map(x -> likelihoods(model, 0, gpu.(x)...), d)
										        
    _mean_error(xs) = (Statistics.mean(xs), 2std(xs) / sqrt(length(xs)))

    values = map(x -> x[1], tuples)
    sizes  = map(x -> x[2], tuples)

    lik_value, lik_error = _mean_error(values)
    @printf("Likelihood at %d context trajectories", i)
    @printf(
	"	%8.3f +- %7.3f (%d batches)\n",
	lik_value,
	lik_error,
        n_batches
    )

end







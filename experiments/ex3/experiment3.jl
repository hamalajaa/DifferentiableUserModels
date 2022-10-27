using ArgParse
using BSON
using Distributions
using Statistics
using Flux
using Stheno
using Tracker
using CUDA

include("../../NeuralProcesses.jl/src/NeuralProcesses.jl")
using .NeuralProcesses

parser = ArgParseSettings()
@add_arg_table! parser begin
    "--gen"
        help = "Experiment setting: gridworld, menu_search, h_menu_search"
        arg_type = String
        default = "h_menu_search"
    "--n_traj"
        help = "Number of context trajectories. Setting to 0 randomizes between 1 and 8."
        arg_type = Int
        default = 0
    "--p_bias"
        help = "Probability of generating a sample with biased model"
        arg_type = Float64
        default = 0.0
    "--params"
        help = "Return params?"
        arg_type = Bool
        default = false
end
args = parse_args(parser)

model = anp_ex2(
    dim_embedding=128,
    num_encoder_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    args=args
) |> gpu

println("Initializing loss...")

loss(xs...) = np_elbo(
    xs...,
    num_samples=5,
    fixed_σ_epochs=3
)

batch_size  = 4
num_batches = 5000

# Redundant. Required to fit the DataGenerator definition
x_context = Distributions.Uniform(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

num_context = Distributions.DiscreteUniform(10, 10)
num_target  = Distributions.DiscreteUniform(10, 10)

data_gen = NeuralProcesses.DataGenerator(
               HierarchicalMenuSampler(args;),
               batch_size=batch_size,
               x_context=x_context,
               x_target=x_target,
               num_context=num_context,
               num_target=num_target,
               σ²=1e-8
           )

println("Proceeding to training loop...")
           
train_model!(
        model,
        loss,
        data_gen,
        ADAM(5e-4),
        bson="ex3/anp",
        starting_epoch=0,
        tasks_per_epoch=2^7,
        epochs=300,
    )











































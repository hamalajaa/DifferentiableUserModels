module NeuralProcesses

using CUDA
using Distributions
using Flux
using LinearAlgebra
using NNlib
using Printf
using Random
using Statistics
using StatsBase
using Tracker

if CUDA.functional()
    include("gpu.jl")

    randn_gpu = CUDA.randn
    zeros_gpu = CUDA.zeros
    ones_gpu = CUDA.ones
else
    const CuOrVector = Vector
    const CuOrMatrix = Matrix
    const CuOrArray = Array

    randn_gpu = randn
    zeros_gpu = zeros
    ones_gpu = ones
end

const AV = AbstractVector
const AM = AbstractMatrix
const AA = AbstractArray

const MaybeAV = Union{Nothing, AbstractVector}
const MaybeAM = Union{Nothing, AbstractMatrix}
const MaybeAA = Union{Nothing, AbstractArray}

export Chain

include("util.jl")
include("data.jl")
include("conv.jl")
include("discretisation.jl")
include("parallel.jl")
include("nn.jl")
include("distribution.jl")

include("model/setconv.jl")
include("model/attention.jl")
include("model/coding.jl")
include("model/coder.jl")
include("model/noise.jl")
include("model/model.jl")

include("model/architectures/cnp.jl")
include("model/architectures/np.jl")
include("model/architectures/acnp.jl")
include("model/architectures/anp.jl")
include("model/architectures/convcnp.jl")
include("model/architectures/corconvcnp.jl")
include("model/architectures/convnp.jl")

####################################################################
# Additional files and imports for differentiable user models here #
####################################################################
using Combinatorics

using POMDPs
using POMDPModels
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using CommonRLInterface

using DeepQLearning
using MCTS

include("../../src/util.jl")
include("../../src/train.jl")
include("../../src/data_gen.jl")

include("../../src/environments/gridworld.jl")
include("../../src/environments/menumodel.jl")
include("../../src/environments/menumodel_h.jl")
include("../../src/environments/menu_assistant.jl")

include("../../src/architectures/np_ex1.jl")
include("../../src/architectures/anp_ex1.jl")
include("../../src/architectures/anp_ex2.jl")
####################################################################
# Additional files and imports for differentiable user models here #
####################################################################

include("experiment/experiment.jl")

end

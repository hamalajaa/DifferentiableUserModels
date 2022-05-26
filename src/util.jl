export Categorical, UniformGrid, ConditionalUniformGrid, np_elbo, predict_action, 
            likelihood, sample_latent, SoftmaxLikelihood, build_categorical_noise

using .NeuralProcesses

"""
Several additional definitions for working with NP-based models.
"""

function np_elbo(
    model::Model,
    epoch::Integer,
    xc::AA,
    yc::AA,
    xt::AA,
    yt::AA;
    num_samples::Integer,
    fixed_σ::Float32=1f-2,
    fixed_σ_epochs::Integer=0,
    kws...
)
    yc = permutedims(yc, [2,1,3])
    yt = permutedims(yt, [2,1,3])

    x_all = cat(xc, xt, dims=1)
    y_all = cat(yc, yt, dims=1)

    # Handle empty set case.
    size(xc, 1) == 0 && (xc = yc = nothing)

    # Perform deterministic and latent encoding.
    xz, pz, h = code_track(model.encoder, xc, yc, x_all; kws...)

    # Construct posterior over latent variable.
    qz = recode_stochastic(model.encoder, pz, x_all, y_all, h; kws...)

    # Sample latent variable and perform decoding.
    z = sample(qz, num_samples=num_samples)
    _, d = code(model.decoder, xz, z, x_all)
   
    logp = mean(sum(logpdf(d.p, y_all), dims=(1, 2)), dims=4)

    elbos = logp .- sum(kl(qz, pz), dims=(1, 2))

    return -mean(elbos), size(x_all, 1)
end

# Get likelihood of observations
function likelihood(
      model::Any,
      epoch::Integer,
      xc::AA,
      yc::AA,
      xt::AA,
      yt::AA;
      target::Bool,
      num_samples::Integer,
      fixed_σ::Float32=1f-2,
      fixed_σ_epochs::Integer=0,
      kws...
)

    yc = permutedims(yc, [2,1,3])
    yt = permutedims(yt, [2,1,3])

    size(xc, 1) == 0 && (xc = yc = nothing)

    xz, pz = code(model.encoder, xc, yc, xt; kws...)

    z = sample(pz, num_samples=num_samples)
    _, d = code(model.decoder, xz, z, xt)

    p = mean(maximum(yt .* d.p, dims=2))

    # Return average over batches and the "size" of the loss.
    return p, size(xt, 1)
end


# Return latent samples
function sample_latent(
    model::Model,
	epoch::Integer,
	xc::AA,
	yc::AA,
	xt::AA;
	num_samples::Integer,
	kws...
)

    yc = permutedims(yc, [2,1,3])

    size(xc, 1) == 0 && (xc = yc = nothing)

    xz, pz = code(model.encoder, xc, yc, xt; kws...)
    
    z = sample(pz, num_samples=num_samples)

    return pz, z
end


function predict_action(
    model::Model,
      epoch::Integer,
      xc::AA,
      yc::AA,
      xt::AA;
      target::Bool,
      num_samples::Integer,
      fixed_σ::Float32=1f-2,
      fixed_σ_epochs::Integer=0,
      kws...
)

    yc = permutedims(yc, [2,1,3])

    size(xc, 1) == 0 && (xc = yc = nothing)

    xz, pz = code(model.encoder, xc, yc, xt; kws...)

    z = sample(pz, num_samples=num_samples)
    _, d = code(model.decoder, xz, z, xt)

    return d.p, size(xt, 1)
end


"""
    struct SoftmaxLikelihood <: Noise

# Fields
- `k`: Number of available actions
"""

struct SoftmaxLikelihood <: Noise
    k
end

SoftmaxLikelihood() = SoftmaxLikelihood(4)

@Flux.functor SoftmaxLikelihood

function (noise::SoftmaxLikelihood)(x::AA)
    p = softmax(x, dims=[2])
    return Categorical(p)
end



function build_categorical_noise(
    build_local_transform=n -> identity;
    dim_y::Integer=1,
)
    num_noise_channels = dim_y
    noise = Chain(
        build_local_transform(dim_y),
        SoftmaxLikelihood(4)
    )

    return num_noise_channels, noise
end


"""
Several additional distribution definitions.
"""

"""
    struct Categorical

# Fields
- `p`: Event probabilities
"""
struct Categorical
    p
end

function sample(d::Categorical; num_samples::Integer)
    c = accumulate(+, d.p)
    r = rand_gpu(data_eltype(d.p), num_samples)
    samples = map(x->count(y->x>y,c),r) + ones_gpu(num_samples)

    return samples
end


function logpdf(d, x) 
    return _cuda_log.(maximum(x .* d, dims=2))
end

Base.map(f, d::Categorical) = Categorical(f(d.p))

probabilities(d::Categorical) = d.p


"""
    struct UniformGrid

# Fields
- `w`: Grid width
- `h`: Grid height 
"""

struct UniformGrid
    w
    h
end


# Returns a set of random states without replacement
sample(d::UniformGrid, num_samples::Integer) =
    StatsBase.sample([(x,y) for x=1:d.w for y=1:d.h], num_samples, replace=false)


"""
    struct ConditionalUniformGrid

# Fields
- `w`: Grid width
- `h`: Grid height
"""

struct ConditionalUniformGrid
    w
    h
end

# Returns a set of random states without replacement that are not already in the conditioning set c
# (Can be used to sample unique input states while avoiding reward states specified in c)
sample(d::ConditionalUniformGrid, num_samples::Integer, c) =
    StatsBase.sample([[x,y] for x=1:d.w for y=1:d.h if (x,y) ∉ c], num_samples, replace=false)



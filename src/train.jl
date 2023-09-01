export eval_model!, train_model!

"""
Essentially the same file as src/expeirment/experiment.jl in NeuralProcesses.jl with slight modifications.
"""

using .NeuralProcesses

using BSON
using CUDA
using Flux
#using PyPlot
using Printf
using ProgressMeter
using Stheno
using Tracker
using Statistics

import StatsBase: std

include("../NeuralProcesses.jl/src/experiment/checkpoint.jl")

function eval_model!(model, loss, data_gen, epoch::Integer; num_batches::Integer=8)
    model = NeuralProcesses.untrack(model)
    
    # Generate evaluation data
    @time data = gen_batch(data_gen, num_batches)

    tuples = map(x -> loss(model, epoch, gpu.(x)...), data)
    values = map(x -> x[1], tuples)
    sizes = map(x -> x[2], tuples)

    # Compute and print loss.
    loss_value, loss_error = _mean_error(values)
    println("Losses:")
    @printf(
        "    %8.3f +- %7.3f (%d batches)\n",
        loss_value,
        loss_error,
        num_batches
    )

    # Normalise by average size of target set.
    @printf(
        "    %8.3f +- %7.3f (%d batches; normalised)\n",
        _mean_error(values ./ mean(sizes))...,
        num_batches
    )

    # Normalise by the target set size.
    @printf(
        "    %8.3f +- %7.3f (%d batches; global mean)\n",
        _mean_error(values ./ sizes)...,
        num_batches
    )

    likelihoods(xs...) = likelihood(
        xs...,
        target=true,
        num_samples=5,
        fixed_σ_epochs=3
    )
    
    tuples = map(x -> likelihoods(model, epoch, gpu.(x)...), data)
    values = map(x -> x[1], tuples)
    sizes  = map(x -> x[2], tuples)

    # Compute and print loss
    lik_value, lik_error = _mean_error(values)
    println("Likelihoods of observations:")
    @printf(
        "    %8.3f +- %7.3f (%d batches)\n",
        lik_value,
        lik_error,
        num_batches
    )

    # Normalise by average size of target set.
    @printf(
        "    %8.3f +- %7.3f (%d batches; normalised)\n",
        _mean_error(values ./ mean(sizes))...,
        num_batches
    )

    # Normalise by the target set size.
    @printf(
        "    %8.3f +- %7.3f (%d batches; global mean)\n",
        _mean_error(values ./ sizes)...,
        num_batches
    )

    return loss_value, loss_error, lik_value, lik_error
end


_mean_error(xs) = (Statistics.mean(xs), 2std(xs) / sqrt(length(xs)))

_nanreport = Flux.throttle(() -> println("Encountered NaN loss! Returning zero."), 1)

function _nansafe(loss, xs...)
    value, value_size = loss(xs...)
    if isnan(value)
        _nanreport()
        return Tracker.track(identity, 0f0), value_size
    else
        return value, value_size
    end
end

function train_model!(
    model,
    loss,
    data_gen,
    opt;
    bson=nothing,
    starting_epoch::Integer=1,
    epochs::Integer=100,
    tasks_per_epoch::Integer=1000
)
    CUDA.GPUArrays.allowscalar(false)
    
    # Divide out batch size to get the number of batches per epoch.
    batches_per_epoch = div(tasks_per_epoch, data_gen.batch_size)

    # Display the settings of the training run.
    @printf("Epochs:               %-6d\n", epochs)
    @printf("Starting epoch:       %-6d\n", starting_epoch)
    @printf("Tasks per epoch:      %-6d\n", batches_per_epoch * data_gen.batch_size)
    @printf("Batch size:           %-6d\n", data_gen.batch_size)
    @printf("Batches per epoch:    %-6d\n", batches_per_epoch)

    # Track the parameters of the model for training.
    model = NeuralProcesses.track(model)

    loss_means  = []
    loss_errors = []
    lik_means   = []
    lik_errors  = []

    for epoch in starting_epoch:(starting_epoch + epochs - 1)
        # Perform epoch.
        CUDA.reclaim()
        @time begin
            ps = Flux.Params(Flux.params(model))
	    # Warmup epoch
	    if epoch == starting_epoch
	    	n_batches = 1
	    else
	    	n_batches = batches_per_epoch
	    end
            @showprogress "Epoch $epoch: " for d in gen_batch(data_gen, n_batches; eval=false)
                gs = Tracker.gradient(ps) do
                    first(_nansafe(loss, model, epoch, gpu.(d)...))
                end
                for p in ps
                    Tracker.update!(p, -Flux.Optimise.apply!(opt, Tracker.data(p), Tracker.data(gs[p])))
                end
            end
        end

        # Evalute model.
        CUDA.reclaim()
        loss_value, loss_error, lik_value, lik_error = eval_model!(
            NeuralProcesses.untrack(model),
            loss,
            data_gen,
            epoch
        )

	    push!(loss_means,  loss_value)
	    push!(loss_errors, loss_error)
	    push!(lik_means,   lik_value)
	    push!(lik_errors,  lik_error)

        CUDA.reclaim()

        # Save result.
        if !isnothing(bson)
            checkpoint!(
                "models/"*bson*"/"*string(epoch)*".bson",
                NeuralProcesses.untrack(model),
                epoch,
                loss_value,
                loss_error
            )
        end
    end

    BSON.bson("results/"*bson*".bson", loss_means=loss_means, loss_stds=loss_errors,
	                             lik_means=lik_means, lik_stds=lik_errors)

end


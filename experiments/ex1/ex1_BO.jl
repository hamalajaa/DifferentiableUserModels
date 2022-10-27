using BSON
using Flux
using Stheno
using Tracker
using StatsBase
using Combinatorics

include("../../NeuralProcesses.jl/src/NeuralProcesses.jl")
using .NeuralProcesses

using Printf
using Statistics

using BayesianOptimization, GaussianProcesses, Distributions

batch_size  = 4

# Redundant. Required to fit the DataGenerator definition
x_context = Distributions.Uniform(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

num_context = Distributions.DiscreteUniform(10, 10)
num_target  = Distributions.DiscreteUniform(10, 10)

bias = 0.0

args = Dict("gen" => "gridworld", "n_traj" => 10, "params" => false, "bias" => 0.0, "perm" => false)

generator = NeuralProcesses.DataGenerator(MCTSPlanner(args;),
			 batch_size=batch_size,
			 x_context=x_context,
			 x_target=x_target,
			 num_context=num_context,
			 num_target=num_target,
			 σ²=1e-8
			)

function simulator(fix_params::Any; fix_init_state::Any=nothing)

    tasks = []

    n_reward_states = 2
    n_features = 3n_reward_states + 5
    
    n_traj = args["n_traj"]
    
    num_trajectories = n_traj

    len_c_traj = Base.rand(1:9)

    context_length = num_trajectories*10 + len_c_traj

    if fix_params == nothing
        # Sample n reward states and their corresponding values 
        r_states = StatsBase.sample([(x,y) for x=1:10 for y=1:10], 2, replace=false)

        # Flatten the array of reward states
        r_states = vcat([collect(t) for t in r_states]...)

        # Sample remaining features
        td         = Base.rand(5:10)
        reuse_tree = Base.rand(0:1)    # Used for v4
        horizon    = Base.rand(0:1)    # Used for v4
    else
        r_states   = fix_params[1]
        td         = fix_params[2]
        reuse_tree = fix_params[3]
        horizon    = fix_params[4]
    end

    r_values   = [-1,1]

    trajectories_s = []
    trajectories_a = []

    in_states = []

    for i in 1:num_trajectories

        if fix_init_state == nothing
            # Sample initial state for the sequence
            in_state = StatsBase.sample([[x,y] for x=1:10 for y=1:10 if (x,y) ∉ r_states], 1, replace=false)
        else
            in_state = fix_init_state[i]
        end
        x = vcat(in_state..., r_states..., r_values, reuse_tree, horizon, td)

        # Get trajectory
        x, y = Base.rand(generator.process(x, generator.σ²))

        push!(in_states, in_state)
        push!(trajectories_s, x)
        push!(trajectories_a, y)

    end

    target_s = Float32.(trajectories_s[num_trajectories])
    target_a = Float32.(trajectories_a[num_trajectories])

    # Past contexts
    p = 1:num_trajectories-1

    context_s = Float32.(cat(trajectories_s[p]..., dims=1))
    context_a = Float32.(cat(trajectories_a[p]..., dims=2))

    push!(tasks, (
            context_s,
            context_a,
            target_s,
            target_a
        ))

    if fix_params == nothing
        params = (r_states, td, reuse_tree, horizon)
    else
        params = fix_params
    end

    return map(x -> cat(x...; dims=3), zip(tasks...)), params, in_states

end
    

# Calculate discrepancy between observed and simulated trajectories
function discrepancy(obs::Any, sim::Any)
    
    # Unpack trajectories
    s_obs = obs
    s_sim = sim

    score = 0

    for i in 1:size(s_obs)[1]
        # l1 distance between trajectory states
        score += sum(abs.(s_obs[i,:].-s_sim[i,:]))^2
    end

    # Normalize
    score = 2. * (score / size(s_obs)[1]) / (18. ^2) - 1.

    return score

end



function accuracy(proposal::Any, init_state::Any, xc::Any, yc::Any, xt::Any, yt::Any)

    # Round proposal to integers
    rx1, ry1, rx2, ry2, td, re, ho = proposal
    p_in = (Int(round(rx1)), Int(round(ry1)), Int(round(rx2)), Int(round(ry2)), Int(round(td)), Int(round(re)), Int(round(ho)))

    t_prob     = 1.0
    ex_const   = 5.0

    reward_states = [GridWorldState(p_in[1], p_in[2]), GridWorldState(p_in[3], p_in[4])]
    reward_values = Float64.([-1., 1.])

    tree_depth = Int64(p_in[5])
    reuse_tree = Bool(p_in[6])
    horizon    = Bool(p_in[7])

    in_state = GridWorldState(init_state[1], init_state[2])

    # Init MDP
    mdp = GridWorld(sx=10, sy=10, rs=reward_states, rv=reward_values, tp=t_prob)

    # Set initial state distribution (deterministic as already sampled)
    initialstate(mdp) = Deterministic(in_state)
    
    if !horizon
        # Init MCTS solver
        solver = NeuralProcesses.MCTSSolver(n_iterations=1000,
                            depth=tree_depth,
                            exploration_constant=ex_const,
                            enable_tree_vis=false,
                            reuse_tree=reuse_tree)
    # Limit tree horizon (i.e., use fixed estimates instead of rollout)
    else
        # Init MCTS solver
        solver = NeuralProcesses.MCTSSolver(n_iterations=1000,
                            depth=tree_depth,
                            exploration_constant=ex_const,
                            enable_tree_vis=false,
                            reuse_tree=reuse_tree,
                            estimate_value=0)
    end

    # Init planner
    planner = NeuralProcesses.solve(solver, mdp)

    # Record predictions
    scores = []

    """
    for i in 1:size(xc)[1]
        s = GridWorldState(xc[i,:]...)
            
        # Plan action
        a = NeuralProcesses.actionindex(mdp, NeuralProcesses.action(planner, s))
        a = Int.(collect(Flux.onehot(a, 1:5)))

        # Update score
        if a == yc[:,i] || yc[:,i] == [0,0,0,0,1]
            push!(scores, 1.)
        else
            push!(scores, 0.)
        end
    end
    """

    # Run through current trajectory
    for i in 1:size(yt)[2]

        s = GridWorldState(xt[i,:]...)

        # Plan action
        a = NeuralProcesses.actionindex(mdp, NeuralProcesses.action(planner, s))
        a = Int.(collect(Flux.onehot(a, 1:5)))

        # Update score
        if a == yt[:,i] || yt[:,i] == [0,0,0,0,1]
            push!(scores, 1.)
        else
            push!(scores, 0.)
        end
    end

    return scores

end



# Individual optimization task
function optimization_task()

    # Generate observed trajectory
    obs, params, in_states = simulator(nothing)
    xc, yc, xt, yt = obs
    
    xc = xc[:,:,1]
    yc = yc[:,:,1]
    xt = xt[:,:,1]
    yt = yt[:,:,1]

    obs = xc

    println(params)
    println(in_states)

    # Function to optimize
    function f(proposal::Any)

        # Round proposal to integers
        rx1, ry1, rx2, ry2, td, re, ho = proposal
        p_in = ([Int(round(rx1)), Int(round(ry1)), Int(round(rx2)), Int(round(ry2))], Int(round(td)), Int(round(re)), Int(round(ho)))

        # Simulate on proposal
        sim, _, _ = simulator(p_in, fix_init_state=in_states)
        xc, _, xt, _ = sim
        
        sim = xc
        
        # Return discrepancy
        return discrepancy(obs, sim)

    end

    # Choose as a model an elastic GP with input dimensions 7.
    model = ElasticGPE(7,                            # 7 input dimensions
                    mean = MeanConst(0.),         
                    kernel = Mat12Ard([0., 0., 0., 0., 0., 0., 0.], 4.),
                    logNoise = 0.5,
                    capacity = 3000)              # the initial capacity of the GP is 3000 samples.
    set_priors!(model.mean, [Normal(0, 1)])

    # Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
    modeloptimizer = MAPGPOptimizer(every = 5, noisebounds = [-5, 5],       # bounds of the logNoise
                                    kernbounds = [[-20, -20, -20, -20, -20, -20, -20, 0], [20, 20, 20, 20, 20, 20, 20, 20]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                    maxeval = 50)
    opt = BOpt(f,
            model,
            UpperConfidenceBound(),                                                     # type of acquisition
            modeloptimizer,
            [1., 1., 1., 1., 5., 0., 0.], [10., 10., 10., 10., 10., 1., 1.],            # lowerbounds, upperbounds         
            repetitions = 1,                                                            # evaluate the function for each input 5 times
            maxiterations = 400,                                                        # evaluate at 100 input positions
            sense = Min,                                                                # minimize the function
            acquisitionoptions = (method = :LD_LBFGS,   # run optimization of acquisition function with NLopts :LD_LBFGS method
                                    restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                    maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                    maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
                verbosity = Progress)

    result = boptimize!(opt)

    println(result)

    param_map = result[4]

    # Calculate accuracy
    scores = accuracy(param_map, in_states[end][1], Int64.(xc), Int64.(yc), Int64.(xt), Int64.(yt))
    
    println(scores)
    println(Statistics.mean(scores))

    return scores

end

s = []

for e in 1:10
    @printf("Epoch:  %-6d\n", e)
    score = optimization_task()

    push!(s, score)
end

bson("results/ex1/bo_avg10.bson", scores=s)



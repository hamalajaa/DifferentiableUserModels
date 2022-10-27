export MCTSPlanner, SearchEnvSampler, HierarchicalMenuSampler, restore_model, gen_batch

using .NeuralProcesses

include("util.jl")

"""
Implementations for the data generators used in the experiments.
"""


# Generate batch of data
function gen_batch(generator::DataGenerator, num_batches::Integer; eval=true)
    gen_type = generator.process.args["gen"]

    if gen_type == "menu_search"
        return [_make_search_env_batch(
            generator,
            eval
        ) for i in 1:num_batches]
    end
    if gen_type == "h_menu_search"
        return [make_menu_batch(
	        generator,
            eval
	    ) for i in 1:num_batches]
    end
    if gen_type == "gridworld"
        return [_make_mcts_batch(
            generator,
            eval=false
        ) for i in 1:num_batches]
    end
end

_float32(x) = Float32.(x)

struct MCTSPlanner
    args::Any
    n_reward_states::Integer
    n_iterations::Integer
    iss::ConditionalUniformGrid  # Input state sampler
    rss::UniformGrid             # Reward state sampler
    rvs::Distribution            # Reward value sampler
    tps::Distribution            # Transition prob sampler
    ecs::Distribution            # Exploration constant sampler
    tds::Distribution            # Tree depth sampler

    function MCTSPlanner(
    args::Any;
        n_reward_states::Integer    = 2,
        n_iterations::Integer       = 50,
        iss::ConditionalUniformGrid = ConditionalUniformGrid(10, 10),
        rss::UniformGrid            = UniformGrid(10, 10),
        rvs::Distribution           = DiscreteUniform(-1, 1),
        tps::Distribution           = Distributions.Uniform(0.75, 1.0),
        ecs::Distribution           = Distributions.Uniform(1.0, 10.0),
        tds::Distribution           = DiscreteUniform(5, 10)
    )
        new(args,
        n_reward_states,
            n_iterations,
            iss,
            rss,
            rvs,
            tps,
            ecs,
            tds
            )
    end
end

(mctsp::MCTSPlanner)(x, noise) = NeuralProcesses.FDD(x, noise, mctsp)


function Base.rand(mctsp::NeuralProcesses.FDD{MCTSPlanner}, num_samples::Integer=10)
    
    # Get n of reward states
    n_reward_states = mctsp.process.n_reward_states

    n_features = 5 + 3n_reward_states

    # Extract individual features
    mdp_features  = Int64.(mctsp.x[1:n_features-3])
    mcts_features = Float64.(mctsp.x[n_features-2:n_features])

    # Initial state
    init_state    = GridWorldState(mdp_features[1:2]...)

    # MDP parametrizations
    reward_states = [GridWorldState(mdp_features[2i-1:2i]...) for i in 2:n_reward_states+1]

    reward_values = Float64.(mdp_features[end-n_reward_states+1:end])

    # MCTS parametrizations
    t_prob     = 1.0 #mcts_features[1]
    ex_const   = 5.0 #mcts_features[2]
    reuse_tree = Bool(mcts_features[1]) # v4
    horizon    = Bool(mcts_features[2]) # v4

    t_depth  = Int64(mcts_features[3])

    # Init MDP
    mdp = GridWorld(sx=10, sy=10, rs=reward_states, rv=reward_values, tp=t_prob)

    # Set initial state distribution (deterministic as already sampled)
    initialstate(mdp) = Deterministic(init_state)
    
    if !horizon
        # Init MCTS solver
        solver = MCTSSolver(n_iterations=1000,
                            depth=t_depth,
                            exploration_constant=ex_const,
                            enable_tree_vis=false,
                            reuse_tree=reuse_tree)
    # Limit tree horizon (i.e., use fixed estimates instead of rollout)
    else
        # Init MCTS solver
        solver = MCTSSolver(n_iterations=1000,
                            depth=t_depth,
                            exploration_constant=ex_const,
                            enable_tree_vis=false,
                            reuse_tree=reuse_tree,
                            estimate_value=0)
    end

    # Init planner
    planner = solve(solver, mdp)

    # Collect trajectories into vectors
    xs = zeros(Float64, 0, 2)
    ys = zeros(Float64, 0)

    # Set s as initial state
    s = init_state

    # Loop MCTS iterations to construct a trajectory
    for j in 1:num_samples
        if s in reward_states
            # Separately specify "stay in place" action when reaching a reward state
            a = 5
            # Collect transition
            xs = vcat(xs, [s.x s.y])
            ys = cat(ys, a, dims=1)
        else
            # Get action
            a = action(planner, s)
            # Collect transition
            xs = vcat(xs, [s.x s.y])
            ys = cat(ys, actionindex(mdp, a), dims=1)
            # Update current state
            s = Base.rand(transition(mdp, s, a))
        end
    end

    # One-hot encode ys
    ys = Int.(collect(Flux.onehotbatch(ys, 1:5)))
    ys_e = ones(size(ys)) .* ys
    
    return xs, ys
end



function _make_mcts_batch(generator::NeuralProcesses.DataGenerator; eval::Bool=false)
    # Sample tasks.
    tasks = []

    n_reward_states = generator.process.n_reward_states
    n_features = 3n_reward_states + 5
    
    n_traj = generator.process.args["n_traj"]
    
    # Sample number and length of trajectories for each batch
    if n_traj == 0
        num_trajectories = Base.rand(1:8)
    else
        num_trajectories = n_traj
    end

    len_c_traj = Base.rand(1:9)

    context_length = num_trajectories*10 + len_c_traj

    for i in 1:generator.batch_size

        # Sample n reward states and their corresponding values 
        r_states = sample(generator.process.rss, n_reward_states)
        #r_values = rand(generator.process.rvs, n_reward_states)
        r_values = [-1,1]

        # Sample remaining features
        t_prob     = Base.rand(generator.process.tps, 1)
        ec         = Base.rand(generator.process.ecs, 1)
        td         = Base.rand(generator.process.tds, 1)
        reuse_tree = Base.rand(0:1)    # Used for v4
        horizon    = Base.rand(0:1)    # Used for v4
    
        # Flatten the array of reward states
        r_states = vcat([collect(t) for t in r_states]...)

        trajectories_s = []
        trajectories_a = []
        trajectories_e = []

        for i in 1:num_trajectories

            # Sample initial state for the sequence
            in_state = sample(generator.process.iss, 1, r_states)
            x = vcat(in_state..., r_states..., r_values, reuse_tree, horizon, td)

            # Get trajectory
            x, y = Base.rand(generator.process(x, generator.σ²))

            push!(trajectories_s, x)
            push!(trajectories_a, y)

        end

        # Each trajectory is selected once as the target
        for i in 1:num_trajectories
            target_s = trajectories_s[i][len_c_traj+1:len_c_traj+1, :]
            target_a = trajectories_a[i][:, len_c_traj + 1]

            current_context_s = trajectories_s[i][1:len_c_traj, :]
            current_context_a = trajectories_a[i][:, 1:len_c_traj]

            past_context_ids = (1:num_trajectories)[1:num_trajectories .!= i]

            # Permute past contexts
            for p in Base.rand(collect(permutations(past_context_ids)), 5)
                past_context_s = trajectories_s[p]
                past_context_a = trajectories_a[p]

                context_s = cat(past_context_s..., current_context_s, dims=1)
                context_a = cat(past_context_a..., current_context_a, dims=2)

                push!(tasks, _float32.((
                    context_s,
                    context_a,
                    target_s,
                    target_a
                )))

            end
        end
    end

    # Collect as a batch and return.
    return map(x -> cat(x...; dims=3), zip(tasks...))
end



struct SearchEnvSampler

    args::Any
    menu_recall_probability::Distribution
    focus_duration_100ms::Distribution
    selection_delay_s::Distribution

    function SearchEnvSampler(args::Any;
        menu_recall_probability::Distribution=Distributions.Beta(3.0, 1.35),
        focus_duration_100ms::Distribution=Distributions.TruncatedNormal(3.0, 1.0, 0.0, 5.0),
        selection_delay_s::Distribution=Distributions.TruncatedNormal(0.3, 0.3, 0.0, 1.0)
    )
        new(args, menu_recall_probability, focus_duration_100ms, selection_delay_s)
    end
end


(s_env_sampler::SearchEnvSampler)(x, noise) = NeuralProcesses.FDD(x, noise, s_env_sampler)

# From DeepQLearning.jl
function restore_model(solver::DeepQLearningSolver, problem::MDP, file_name::String)
    env = convert(AbstractEnv, problem)
    restore_model(solver, env, file_name)
end

# From DeepQLearning.jl
function restore_model(solver::DeepQLearningSolver, env::CommonRLInterface.AbstractEnv, file_name::String)
    if solver.dueling
        active_q = DeepQLearning.create_dueling_network(solver.qnetwork)
    else
        active_q = solver.qnetwork
    end
    policy = NNPolicy(env, active_q, collect(CommonRLInterface.actions(env)), length(DeepQLearning.obs_dimensions(env)))
    weights = BSON.load(file_name)[:qnetwork]
    Flux.loadparams!(getnetwork(policy), weights)
    Flux.testmode!(getnetwork(policy))
    return policy
end

function Base.rand(s_env_sampler::NeuralProcesses.FDD{SearchEnvSampler}, n_traj::Integer, p_bias, params; num_samples::Integer=5)

    mdp, policy = s_env_sampler.x

    p_rec = mdp.menu_recall_probability
    f_dur = mdp.focus_duration_100ms
    d_sel = mdp.selection_delay_s

    num_trajectories = n_traj

    x_all = []
    y_all = []

    for i in 1:num_trajectories

        reset(mdp)

        xs = zeros(Float32, 0, 17)
        ys = zeros(Float32, 0)

        target = mdp.target_idx

        if mdp.target_idx == nothing
            target = 0
        end

        # Loop iterations to construct a trajectory
        for j in 1:num_samples

            if !isFinished(mdp)

                items = [Float64.([i.item_relevance, i.item_length]) for i in mdp.state.obs_items] |> vec
	       
		        duration = mdp.action_duration == nothing ? 0.0 : mdp.action_duration
		
                s = cat(items..., mdp.state.focus, dims=1)
                
                # Random saccades for biased model
		        if Base.rand() < 0.4 * p_bias
		            a = Base.rand(POMDPs.actions(mdp))
		        else
                    a = action(policy, mdp.state)
		        end

                xs = vcat(xs, transpose(s))
                ys = cat(ys, a, dims=1)

                gen(mdp, mdp.state, a, 0)

            else

                # Return vectors of -1 for finished mdps
                s = zeros(1,17) .- 1
                a = 8

                xs = vcat(xs, s)
                ys = cat(ys, a, dims=1)

            end
        end

	    if params
            xs = hcat(xs, ones(num_samples) * p_rec,
                          ones(num_samples) * f_dur,
                          ones(num_samples) * d_sel,
                          ones(num_samples) * target)
        end

        # One-hot encode ys
        ys = Flux.onehotbatch(ys, 0:8)

        push!(x_all, xs)
        push!(y_all, Int32.(ys))

    end

    return x_all, y_all

end



function _make_search_env_batch(generator::NeuralProcesses.DataGenerator, eval::Bool)

    n_samples = 5

    tasks = []

    # Number of trajectories
    n_traj = generator.process.args["n_traj"]

    # Probability of a biased model
    if eval
        p_bias = 0.0
    else
        p_bias = generator.process.args["p_bias"]
    end

    params = generator.process.args["params"]

    # Sample number and length of trajectories for each batch
    if n_traj == 0
        num_trajectories = Base.rand(1:8)
    else
        num_trajectories = n_traj
    end

    len_c_traj = 1

    for i in 1:generator.batch_size

        # Sample user parameters
        p_rec = Base.rand(generator.process.menu_recall_probability)
        f_dur = Base.rand(generator.process.focus_duration_100ms)
        d_sel = Base.rand(generator.process.selection_delay_s)

        # Init search env
        mdp = SearchEnvironment()

        mdp.menu_recall_probability = p_rec
        mdp.focus_duration_100ms    = f_dur
        mdp.selection_delay_s       = d_sel

        # Reset environment
        reset(mdp)

        # Model for deepQ
        model = Chain(Dense(18, 32), Dense(32, length(POMDPs.actions(mdp))))

        # Exploration policy
        exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=0.10, stop=0.01, steps=10000/2))

        # Init solver
        solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
                                    exploration_policy = exploration,
                                    learning_rate=0.0020,log_freq=50000000,
                                    recurrence=false,double_q=true, dueling=true, prioritized_replay=true,
                                    max_episode_length=5, save_freq=1000000, verbose=false, logdir=nothing)

        env = MDPCommonRLEnv{AbstractArray{Float32}}(mdp)

        action_map = collect(CommonRLInterface.actions(env))
        action_indices = Dict(a=>i for (i, a) in enumerate(action_map))

        # check reccurence
        if isrecurrent(solver.qnetwork) && !solver.recurrence
            throw("DeepQLearningError: you passed in a recurrent model but recurrence is set to false")
        end
        replay = DeepQLearning.initialize_replay_buffer(solver, env, action_indices)
        if solver.dueling
            active_q = DeepQLearning.create_dueling_network(solver.qnetwork)
        else
            active_q = solver.qnetwork
        end

        base_file = "log/qnetwork.bson"

        # Load pre-trained model
        policy = restore_model(solver, env, base_file)

        # Update model on current env
        policy = DeepQLearning.dqn_train!(solver, env, policy, replay)

        # Get data
        xs, ys = Base.rand(generator.process((mdp, policy), generator.σ²), num_trajectories, p_bias, params)

        # Divide data into context and target sets
        for j in 1:num_trajectories
        
            target_s = xs[j][len_c_traj+1:end,:]
            target_a = ys[j][:,len_c_traj+1:end]

            current_context_s = xs[j][1:len_c_traj,:]
            current_context_a = ys[j][:,1:len_c_traj]

            past_context_ids = (1:num_trajectories)[1:num_trajectories .!= j]

            # Permute past contexts
            for p in Base.rand(collect(permutations(past_context_ids)), 5)
                    
                past_context_s = xs[p]
                past_context_a = ys[p]

                context_s = cat(past_context_s..., current_context_s, dims=1)
                context_a = cat(past_context_a..., current_context_a, dims=2)

                push!(tasks, _float32.((
                    context_s,
                    context_a,
                    target_s,
                    target_a
                )))

            end
        end

    end

    # Collect as a batch and return.
    return map(x -> cat(x...; dims=3), zip(tasks...))

end


struct HierarchicalMenuSampler

    args::Any
    menu_recall_probability::Distribution
    focus_duration_100ms::Distribution
    selection_delay_s::Distribution

    function HierarchicalMenuSampler(args::Any;
        menu_recall_probability::Distribution=Distributions.Beta(3.0, 1.35),
        focus_duration_100ms::Distribution=Distributions.TruncatedNormal(3.0, 1.0, 0.0, 5.0),
        selection_delay_s::Distribution=Distributions.TruncatedNormal(0.3, 0.3, 0.0, 1.0)
    )
        new(args, menu_recall_probability, focus_duration_100ms, selection_delay_s)
    end
end

(s_env_sampler::HierarchicalMenuSampler)(x, noise) = NeuralProcesses.FDD(x, noise, s_env_sampler)


function Base.rand(s_env_sampler::NeuralProcesses.FDD{HierarchicalMenuSampler}, n_traj::Integer, p_bias, params; num_samples::Integer=20)

    mdp, policy1, policy2 = s_env_sampler.x

    p_rec = mdp.menu_recall_probability
    f_dur = mdp.focus_duration_100ms
    d_sel = mdp.selection_delay_s

    num_trajectories = n_traj

    x_all = []
    y_all = []

    target_groups = []

    for i in 1:num_trajectories

        reset(mdp)

        target_group = div(sum((mdp.rel_labels .== 2) .* mdp.log_labels), 2)

        xs = zeros(Float32, 0, 17)
        ys = zeros(Float32, 0)

        target_menu = mdp.sub_menus[mdp.target_menu]
        target = target_menu.target_idx

        # Loop iterations to construct a trajectory
        for j in 1:num_samples

            if !isFinished(mdp)

                # Map user observations to assistant observations
                if mdp.current_menu == 0
                    current_menu = mdp.main_menu
                    log_groups1 = mdp.log_labels[:,1]
                    log_groups2 = mdp.log_labels[:,2]

                    items = []

                    for (j,i) in enumerate(current_menu.state.obs_items)
                        masked_log_item1 = log_groups1[j] * Int64(i.item_relevance > 0)
                        masked_log_item2 = log_groups2[j] * Int64(i.item_relevance > 0)
                        push!(items, Float64.([masked_log_item1, masked_log_item2]))
                    end
                else
                    current_menu = mdp.sub_menus[mdp.current_menu]
                    log_menu = mdp.log_labels[mdp.current_menu,:]
                    log_menu = vcat(ones(Int64, 4) .* log_menu[1], ones(Int64, 4) .* log_menu[2])

                    items = []

                    for (j,i) in enumerate(current_menu.state.obs_items)
                        # Mask unobserved items and map relevancies to logical groups
                        # (assume unobservability can be inferred based on eye fixations)
                        masked_log_item = log_menu[j] * Int64(i.item_relevance > 0)
                        # Mask length relation to target
                        length = i.item_length
                        if mdp.length_permutation
                            length = replace([length], 1. => 2., 2. => 1.)[1]
                        end
                        push!(items, Float64.([masked_log_item, length]))
                    end
                    
                end
                
		        duration = mdp.action_duration == nothing ? 0.0 : mdp.action_duration
		
                s = cat(items..., current_menu.state.focus, dims=1)
                
                if Base.rand() < 0.4 * p_bias
                    a = Base.rand(POMDPs.actions(mdp))
                else
                    # Handle main menu and submenus with appropriate policies
                    if mdp.current_menu == 0
                        a = action(policy1, current_menu.state)
                    else
                        a = action(policy2, current_menu.state)
                    end
                end

                xs = vcat(xs, transpose(s))
                ys = cat(ys, a, dims=1)

                gen(mdp, current_menu.state, a, 0)

            else

                # Return vectors of -1 for finished mdps
                s = zeros(1,17) .- 1
                a = 8

                xs = vcat(xs, s)
                ys = cat(ys, a, dims=1)

            end
        end

	    if params
            xs = hcat(xs, ones(num_samples) * target_group)
        end

        # One-hot encode ys
        ys = Flux.onehotbatch(ys, 0:8)

        push!(x_all, xs)
        push!(y_all, Int32.(ys))

    end

    return x_all, y_all

end


function make_menu_batch(generator::NeuralProcesses.DataGenerator, eval::Bool)

    tasks = []

    # Number of trajectories
    n_traj = generator.process.args["n_traj"]

    # Probability of a biased model
    if eval
        p_bias = 0.0
    else
        p_bias = generator.process.args["p_bias"]
    end

    params = generator.process.args["params"]

    # Sample number and length of trajectories for each batch
    if n_traj == 0
        num_trajectories = 1
    else
        num_trajectories = n_traj
    end

    len_c_traj = Base.rand(2:10)

    for i in 1:generator.batch_size

        # Sample user parameters
        p_rec = Base.rand(generator.process.menu_recall_probability)
        f_dur = Base.rand(generator.process.focus_duration_100ms)
        d_sel = Base.rand(generator.process.selection_delay_s)

        # Train separate policies for sub menus and main menus to reduce training time

        # Init training main menu
        mdp1 = HierarchicalSearchEnvironment()
        mdp1.training = true
        mdp1.init_menu = true
        mdp1.max_number_of_actions_per_session = 20
        mdp1.menu_recall_probability = p_rec
        mdp1.focus_duration_100ms    = f_dur
        mdp1.selection_delay_s       = d_sel

        # Init training sub menus
        mdp2 = HierarchicalSearchEnvironment()
        mdp2.training = true
        mdp2.init_menu = false
        mdp2.max_number_of_actions_per_session = 20
        mdp2.menu_recall_probability = p_rec
        mdp2.focus_duration_100ms    = f_dur
        mdp2.selection_delay_s       = d_sel

        reset(mdp1)
        reset(mdp2)

        # Model for deepQ
        model1 = Chain(Dense(18, 32, elu), Dense(32, 64, elu), Dense(64, length(POMDPs.actions(mdp1))))
        model2 = Chain(Dense(18, 32, elu), Dense(32, 64, elu), Dense(64, length(POMDPs.actions(mdp2))))

        # Exploration policy
        exploration1 = EpsGreedyPolicy(mdp1, LinearDecaySchedule(start=0.10, stop=0.01, steps=10000/2))
        exploration2 = EpsGreedyPolicy(mdp2, LinearDecaySchedule(start=0.10, stop=0.01, steps=10000/2))

        # Init solver
        solver1 = DeepQLearningSolver(qnetwork = model1, max_steps=10000, 
                             exploration_policy = exploration1,
                             learning_rate=0.0020,log_freq=50000000,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true,
                             max_episode_length=20, save_freq=1000000, verbose=false, logdir=nothing)
        solver2 = DeepQLearningSolver(qnetwork = model1, max_steps=10000, 
                             exploration_policy = exploration2,
                             learning_rate=0.0020,log_freq=50000000,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true,
                             max_episode_length=20, save_freq=1000000, verbose=false, logdir=nothing)

        env1 = MDPCommonRLEnv{AbstractArray{Float32}}(mdp1)
        env2 = MDPCommonRLEnv{AbstractArray{Float32}}(mdp2)

        action_map1 = collect(CommonRLInterface.actions(env1))
        action_indices1 = Dict(a=>i for (i, a) in enumerate(action_map1))

        action_map2 = collect(CommonRLInterface.actions(env2))
        action_indices2 = Dict(a=>i for (i, a) in enumerate(action_map2))

        replay1 = DeepQLearning.initialize_replay_buffer(solver1, env1, action_indices1)
        replay2 = DeepQLearning.initialize_replay_buffer(solver2, env2, action_indices2)
        active_q1 = DeepQLearning.create_dueling_network(solver1.qnetwork)
        active_q2 = DeepQLearning.create_dueling_network(solver2.qnetwork)

        base_file1 = "models/bases/mainmenu.bson"
        base_file2 = "models/bases/submenu.bson"

        # Load pre-trained model
        policy1 = restore_model(solver1, env1, base_file1)
        policy2 = restore_model(solver2, env2, base_file2)

        # Update model on current env
        policy1 = DeepQLearning.dqn_train!(solver1, env1, policy1, replay1)
        policy2 = DeepQLearning.dqn_train!(solver2, env2, policy2, replay2)

        # Init full search env
        mdp = HierarchicalSearchEnvironment()
        mdp.training = false
        mdp.init_menu = true
        mdp.max_number_of_actions_per_session = 20
        mdp.menu_recall_probability = p_rec
        mdp.focus_duration_100ms    = f_dur
        mdp.selection_delay_s       = d_sel

        reset(mdp)

        # Get data
        xs, ys = Base.rand(generator.process((mdp, policy1, policy2), generator.σ²), num_trajectories, p_bias, params)

        # Divide data into context and target sets
        for j in 1:num_trajectories
        
            target_s = xs[j][len_c_traj+1:end,:]
            target_a = ys[j][:,len_c_traj+1:end]

            current_context_s = xs[j][1:len_c_traj,:]
            current_context_a = ys[j][:,1:len_c_traj]

            past_context_ids = (1:num_trajectories)[1:num_trajectories .!= j]

            # Permute past contexts
            for p in Base.rand(collect(permutations(past_context_ids)), 5)
                    
                past_context_s = xs[p]
                past_context_a = ys[p]

                context_s = cat(past_context_s..., current_context_s, dims=1)
                context_a = cat(past_context_a..., current_context_a, dims=2)

                push!(tasks, _float32.((
                    context_s,
                    context_a,
                    target_s,
                    target_a
                )))

            end
        end

    end

    # Collect as a batch and return.
    return map(x -> cat(x...; dims=3), zip(tasks...))

end









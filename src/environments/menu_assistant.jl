export MenuAssistant, reset, gen

# MDP definition
mutable struct MenuAssistant <: MDP{AbstractArray{Float32}, AbstractArray{Int64}}
	
	env::Union{HierarchicalSearchEnvironment, Nothing}
	env_main::Union{HierarchicalSearchEnvironment, Nothing}
	env_sub::Union{HierarchicalSearchEnvironment, Nothing}
	user_model::Union{Any, Nothing}
	data_gen::Union{HierarchicalMenuSampler, Nothing}
	finished::Union{Bool, Nothing}
	accumulated_time::Union{Float64, Nothing}
	user_policy1::Union{Any, Nothing}
	user_policy2::Union{Any, Nothing}
	xc::Union{Any, Nothing}
	yc::Union{Any, Nothing}
	f_dur::Union{Float64, Nothing}
	d_sel::Union{Float64, Nothing}
	action::Union{Vector{Int64}, Nothing}
	groups_seen::Union{Any, Nothing}
	opened_sub_menu::Bool
	n_traj::Int64
	enable_assistant::Bool
    observe_target::Bool
    anp_assistant::Bool
	maml_assistant::Bool
    accuracy::Vector{Int64}
	predictions::Vector{Any}
	prediction_times::Vector{Float64}
	
end

function MenuAssistant()
	MenuAssistant(nothing, nothing, nothing, nothing, nothing, false, 0., nothing, nothing, nothing, nothing, nothing, nothing, nothing, [], false, 0, true, false, true, false, [], [], [])
end

function reset(mdp::MenuAssistant)

	mdp.finished = false

	if mdp.n_traj % 20 == 0

		p_rec = Base.rand(mdp.data_gen.menu_recall_probability)
		f_dur = Base.rand(mdp.data_gen.focus_duration_100ms)
		d_sel = Base.rand(mdp.data_gen.selection_delay_s)

		mdp.env.menu_recall_probability = p_rec
		mdp.env.focus_duration_100ms    = f_dur
		mdp.env.selection_delay_s       = d_sel

		mdp.env_main.menu_recall_probability = p_rec
		mdp.env_main.focus_duration_100ms    = f_dur
		mdp.env_main.selection_delay_s       = d_sel

		mdp.env_sub.menu_recall_probability = p_rec
		mdp.env_sub.focus_duration_100ms    = f_dur
		mdp.env_sub.selection_delay_s       = d_sel

		mdp.f_dur = f_dur
		mdp.d_sel = d_sel

		reset(mdp.env)
		reset(mdp.env_main)
		reset(mdp.env_sub)

		# Model for deepQ
		model1 = Chain(Dense(18, 32, elu), Dense(32, 64, elu), Dense(64, length(POMDPs.actions(mdp.env_main))))
		model2 = Chain(Dense(18, 32, elu), Dense(32, 64, elu), Dense(64, length(POMDPs.actions(mdp.env_sub))))

		# Exploration policy
		exploration1 = EpsGreedyPolicy(mdp.env_main, LinearDecaySchedule(start=0.10, stop=0.01, steps=10000/2))
		exploration2 = EpsGreedyPolicy(mdp.env_sub, LinearDecaySchedule(start=0.10, stop=0.01, steps=10000/2))

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

		env1 = MDPCommonRLEnv{AbstractArray{Float32}}(mdp.env_main)
		env2 = MDPCommonRLEnv{AbstractArray{Float32}}(mdp.env_sub)

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
		mdp.user_policy1 = DeepQLearning.dqn_train!(solver1, env1, policy1, replay1)
		mdp.user_policy2 = DeepQLearning.dqn_train!(solver2, env2, policy2, replay2)

		mdp.n_traj = 1

	else

		reset(mdp.env)
		reset(mdp.env_main)
		reset(mdp.env_sub)

		mdp.n_traj += 1

	end
	
end

struct DetState
	s::AbstractArray{Float32}
end

Base.rand(d::DetState) = d.s

# Action space
POMDPs.actions(::MenuAssistant) = vcat(collect(with_replacement_combinations(1:8,2)), [[0,0]])

# Discount
POMDPs.discount(::MenuAssistant) = 0.9


POMDPs.isterminal(mdp::MenuAssistant, state::Any) = mdp.finished

predict(xs...) = NeuralProcesses.predict_action(
		xs...,
		target=true,
		num_samples=5,
		fixed_σ_epochs=3
	  )


# Simulates one user action
function user_step(mdp::MenuAssistant)

	opened_sub_menu = false

	# Map user observations to assistant observations
	if mdp.env.current_menu == 0
		current_menu = mdp.env.main_menu
		log_groups1 = mdp.env.log_labels[:,1]
		log_groups2 = mdp.env.log_labels[:,2]

		items = []

		for (j,i) in enumerate(current_menu.state.obs_items)
			masked_log_item1 = log_groups1[j] * Int64(i.item_relevance > 0)
			masked_log_item2 = log_groups2[j] * Int64(i.item_relevance > 0)
			push!(items, Float64.([masked_log_item1, masked_log_item2]))
		end
	else

		current_menu = mdp.env.sub_menus[mdp.env.current_menu]

		# If in synth menu
		if mdp.env.current_menu == 9
			# Construct assistant observation of synth menu
			log_menu = vcat(ones(Int64, 4) .* mdp.action[1], ones(Int64, 4) .* mdp.action[2])
		else
			log_menu = mdp.env.log_labels[mdp.env.current_menu,:]
			log_menu = vcat(ones(Int64, 4) .* log_menu[1], ones(Int64, 4) .* log_menu[2])
			push!(mdp.groups_seen, (mdp.env.current_menu, 1))
			push!(mdp.groups_seen, (mdp.env.current_menu, 2))
		end

		items = []

		for (j,i) in enumerate(current_menu.state.obs_items)
			# Mask unobserved items and map relevancies to logical groups
			# (assume unobservability can be inferred based on eye fixations)
			masked_log_item = log_menu[j] * Int64(i.item_relevance > 0)
			# Mask length relation to target
			length = i.item_length
			if mdp.env.length_permutation
				length = replace([length], 1. => 2., 2. => 1.)[1]
			end
			push!(items, Float64.([masked_log_item, length]))
		end
		opened_sub_menu = true
	end

	duration = mdp.env.action_duration == nothing ? 0.0 : mdp.env.action_duration

	s = cat(items..., current_menu.state.focus, dims=1)

	# Handle main menu and submenus with appropriate policies
	if mdp.env.current_menu == 0
		a = action(mdp.user_policy1, current_menu.state)
	else
		a = action(mdp.user_policy2, current_menu.state)
	end

	x = transpose(s)
	y = a

	if opened_sub_menu
	    mdp.opened_sub_menu = true
	end
	gen(mdp.env, current_menu.state, a, 0)

	return x, y, duration

end


# Simulate user until back in main menu
function observe_user(mdp::MenuAssistant)

	accumulated_time = 0.

	xc = zeros(Float32, 0, 17)
    yc = zeros(Float32, 0)

	# Simulate user (max 20 steps)
	for i in 1:20

		x, y, duration = user_step(mdp)

		accumulated_time += duration
    	
		xc = vcat(xc, x)
        yc = cat(yc, y, dims=1)

		if isFinished(mdp.env)
			mdp.finished = true
			break
		end

		# Assistant to act if user in main menu and visited a submenu at least once
		if mdp.env.current_menu == 0 && mdp.opened_sub_menu
			break
		end

	end

	yc = Flux.onehotbatch(yc, 0:8)

	xc = ones(1,1,1) .* xc
	yc = ones(1,1,1) .* yc

	return xc, yc, accumulated_time

end
	


# Initial state distribution
function POMDPs.initialstate(mdp::MenuAssistant)

	reset(mdp)

	# Observe user until returned to main menu
	mdp.xc, mdp.yc, mdp.accumulated_time = observe_user(mdp)

	return DetState([1.])

end


function synthetize_menu(mdp::MenuAssistant, a::Vector{Int64})

	synth_menu = SubMenu()
	group1, group2 = a

	idx1 = nothing
	idx2 = nothing

	for i in 1:100
		# Select submenu groups
		idx1 = Base.rand(findall(x->x==group1, mdp.env.log_labels))
		idx2 = Base.rand(findall(x->x==group2, mdp.env.log_labels))
		if !((idx1[1], idx1[2]) in mdp.groups_seen) && !((idx2[1], idx2[2]) in mdp.groups_seen)
			break
		end
	end

	push!(mdp.groups_seen, (idx1[1], idx1[2]))
	push!(mdp.groups_seen, (idx2[1], idx2[2]))

	group1_items = mdp.env.sub_menus[idx1[1]].items[(idx1[2]-1)*4+1:idx1[2]*4]
	group2_items = mdp.env.sub_menus[idx2[1]].items[(idx2[2]-1)*4+1:idx2[2]*4]

	synth_menu.menu_recall_probability = 0.0
	synth_menu.focus_duration_100ms    = mdp.env.focus_duration_100ms
    synth_menu.selection_delay_s       = mdp.env.selection_delay_s

	synth_menu.fixed_elements = cat(group1_items, group2_items, dims=1)

	reset(synth_menu)

	r = 0

	if mdp.env.rel_labels[idx1[1], idx1[2]] == 2 && idx1[1] == mdp.env.target_menu
		r += 5000
	end
	if mdp.env.rel_labels[idx2[1], idx2[2]] == 2 && idx2[1] == mdp.env.target_menu
		r += 5000
	end

	return synth_menu, idx1, idx2, r
	
end


function predict_menu(mdp::MenuAssistant)

	t = 0.
    s2 = cat(vcat([mdp.env.log_labels[i,:] for i in 1:8]...)..., mdp.env.main_menu.state.focus, dims=1)
    xt2 = ones(1,1,1) .* transpose(s2)

    if mdp.anp_assistant
        d2 = [mdp.xc, mdp.yc, xt2]
        @time p2, _ = map(x -> predict(mdp.user_model, 0, gpu.(x)...), [d2])[1] |> cpu
	elseif mdp.maml_assistant
		θ = Flux.params(mdp.user_model)
		θ_prev = deepcopy(θ)
		batchopt = Descent(0.01)
		t = @elapsed begin
			@time begin	
				for i in 1:32
					loss1(x, y) = Flux.crossentropy(mdp.user_model(x), y)
					grad = Flux.gradient(θ) do
						loss1(permutedims(mdp.xc, [2,1,3]), mdp.yc)
					end
					for θ_i in θ
						θ_i .-= Flux.Optimise.apply!(batchopt, θ_i, grad[θ_i])
					end
				end
				p2 = mdp.user_model(permutedims(cat(mdp.xc, xt2, dims=1), [2,1,3]))[:,end]
			end
		end
	else
        @time p2 = mdp.user_model(permutedims(cat(mdp.xc, xt2, dims=1), [2,1,3]))[:,end]
    end	

    prediction = findmax(p2[1:8])[2]

	return prediction, t

end



# POMDP definition with generative interface
function POMDPs.gen(mdp::MenuAssistant, state, act, rng)

	if isFinished(mdp.env)
		mdp.finished = true
	end

	t = 0.

	if !mdp.finished
		
        if mdp.enable_assistant
            if !mdp.observe_target
		        prediction, t = predict_menu(mdp)
				push!(mdp.predictions, prediction)
				
				if t == 0.
					gen(mdp.env, mdp.env.main_menu.state, prediction, 0)
					mdp.env.main_menu.state.obs_items[prediction].item_relevance -= 1
					mdp.env.main_menu.items[prediction].item_relevance -= 1

					if prediction == mdp.env.target_menu
						push!(mdp.accuracy, 1)
					else
						push!(mdp.accuracy, 0)
					end

				else
					push!(mdp.prediction_times, mdp.accumulated_time + 1000 * t)
					i = findlast(mdp.accumulated_time .> mdp.prediction_times)
					if i != nothing
						gen(mdp.env, mdp.env.main_menu.state, mdp.predictions[i], 0)
						mdp.env.main_menu.state.obs_items[prediction].item_relevance -= 1
						mdp.env.main_menu.items[prediction].item_relevance -= 1

						if prediction == mdp.env.target_menu
							push!(mdp.accuracy, 1)
						else
							push!(mdp.accuracy, 0)
						end
					end
				end

            else
                gen(mdp.env, mdp.env.main_menu.state, mdp.env.target_menu, 0)
                mdp.env.main_menu.state.obs_items[mdp.env.target_menu].item_relevance -= 1
                mdp.env.main_menu.items[mdp.env.target_menu].item_relevance -= 1
            end
            mdp.accumulated_time += mdp.env.action_duration
		end
		
		xs, ys, time = observe_user(mdp)

		mdp.xc = cat(mdp.xc, xs, dims=1)
		mdp.yc = cat(mdp.yc, ys, dims=2)

		mdp.accumulated_time += time
	end

	s = Float32.([mdp.finished])
	r = mdp.accumulated_time

	return (sp=s, o=mdp.accuracy, r=r)

end

POMDPs.isterminal(mdp::MenuAssistant, s::AbstractArray{Float32}) = mdp.finished

function POMDPs.convert_s(T::Type{A1}, v::A2, problem::MenuAssistant) where {A1<:AbstractArray, A2<:AbstractArray}

	return Float32.(v)

end















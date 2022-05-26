export SubMenu, reset, gen, isFinished, HierarchicalSearchEnvironment, make_menu_batch

mutable struct SubMenu <: MDP{State, Int64}

    menu_type::String
    menu_groups::Int64
    menu_items_per_group::Int64
    semantic_levels::Int64
    gap_between_items::Float64
    prop_target_absent::Float64
    length_observations::Bool
    p_obs_len_cur::Float64
    p_obs_len_adj::Float64
    n_item_lengths::Int64
    semantic_type::Int64
    fix_target_present::Bool
    fixed_elements::Union{Vector{MenuItem}, Nothing}

    n_items::Int64

    discrete_states::Bool
    outdim::Int64
    indim::Int64
    discrete_actions::Bool
    num_actions::Int64

    menu_recall_probability::Float64
    focus_duration_100ms::Float64
    p_obs_adjacent::Float64
    selection_delay_s::Float64

    # Conditionals
    items::Union{Vector{MenuItem}, Nothing}
    target_present::Union{Bool, Nothing}
    target_idx::Union{Int64, Nothing}
    state::Union{State, Nothing}
    prev_state::Union{State, Nothing}
    action_duration::Union{Float64, Nothing}
    duration_focus_ms::Union{Float64, Nothing}
    duration_saccade_ms::Union{Float64, Nothing}
    action::Union{Int64, Nothing}
    gaze_location::Union{Int64, Nothing}
    n_actions::Union{Int64, Nothing}
    item_locations::Union{Vector{Float64}, Nothing}
    

    function SubMenu(;menu_type::String="semantic",
                                menu_groups::Int64=2,
                                menu_items_per_group::Int64=4,
                                semantic_levels::Int64=3,
                                gap_between_items::Float64=0.75,
                                prop_target_absent::Float64=0.1,
                                length_observations::Bool=true,
                                p_obs_len_cur::Float64=0.95,
                                p_obs_len_adj::Float64=0.89,
                                n_item_lengths::Int64=3,
                                semantic_type::Int64=4,
                                fix_target_present::Bool=false,
                                discrete_states::Bool=true,
                                outdim::Int64=1,
                                indim::Int64=1,
                                discrete_actions::Bool=true,
                                menu_recall_probability::Distribution=Distributions.Beta(3.0, 1.35),
                                focus_duration_100ms::Distribution=Distributions.TruncatedNormal(3.0, 1.0, 0.0, 5.0),
                                p_obs_adjacent::Float64=0.93,
                                selection_delay_s::Distribution=Distributions.TruncatedNormal(0.3, 0.3, 0.0, 1.0))

        new(menu_type, 
            menu_groups, 
            menu_items_per_group, 
            semantic_levels, 
            gap_between_items, 
            prop_target_absent, 
            length_observations,
            p_obs_len_cur, 
            p_obs_len_adj,
            n_item_lengths,
            semantic_type,
            fix_target_present,
            nothing,
            menu_groups*menu_items_per_group,
            discrete_states,
            outdim,
            indim,
            discrete_actions,
            menu_groups*menu_items_per_group+1,
            Base.rand(menu_recall_probability),
            Base.rand(focus_duration_100ms),
            p_obs_adjacent,
            Base.rand(selection_delay_s),
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing
            )
    end
end


function _get_menu(mdp::SubMenu, semantic_type::Int64, target_present::Bool)

    if mdp.fixed_elements == nothing

        # generate menu item semantic relevances and lengths
        items = Vector()
        if mdp.menu_type == "semantic"
            items, target_idx = _get_semantic_menu(mdp, mdp.menu_groups,
                                                mdp.menu_items_per_group, 
                                                mdp.semantic_levels, 
                                                semantic_type)
        elseif mdp.menu_type == "unordered"
            items, target_idx = _get_unordered_menu(mdp, mdp.menu_groups,
                                                    mdp.menu_items_per_group, 
                                                    mdp.semantic_levels, 
                                                    mdp.prop_target_absent)
        else
            error("Unknown menu type")
        end

        lengths = Base.rand(0:mdp.n_item_lengths-1, length(items))

        if target_present
            items[target_idx].item_relevance = ItemRelevance().TARGET_RELEVANCE
            target_len = lengths[target_idx]
        else
            target_len = Base.rand(0:mdp.n_item_lengths)
        end

        for (i, length) in enumerate(lengths)
            if length == target_len
                items[i].item_length = ItemLength().TARGET_LENGTH
            else
                items[i].item_length = ItemLength().NOT_TARGET_LENGTH
            end
        end
        
        menu = Menu(items, target_present, target_idx)

    else

        items = mdp.fixed_elements
        lengths = zeros(length(items))
        target_present = false
        target_idx = 0

        menu = Menu(items, target_present, target_idx)

    end
        
    return menu
end


function reset(mdp::SubMenu)

    menu = _get_menu(mdp, mdp.semantic_type, mdp.fix_target_present)

    mdp.items = menu.items
    mdp.target_present = menu.target_present
    mdp.target_idx = menu.target_idx

    obs_items = [MenuItem(ItemRelevance().NOT_OBSERVED, ItemLength().NOT_OBSERVED) for i in 1:mdp.n_items]

    focus = Focus().ABOVE_MENU
    quit = false
    mdp.state = State(obs_items, focus, quit)
    mdp.prev_state = deepcopy(mdp.state)

    mdp.action_duration     = nothing
    mdp.duration_focus_ms   = nothing
    mdp.duration_saccade_ms = nothing
    mdp.action              = nothing
    mdp.gaze_location       = nothing
    mdp.n_actions           = 0

    mdp.item_locations = collect(mdp.gap_between_items:
                                 mdp.gap_between_items:
                                 mdp.gap_between_items*(mdp.n_items+2))

end


function perform_action(mdp::SubMenu, action::Int64)

    mdp.action = action
    mdp.prev_state = deepcopy(mdp.state)
    mdp.state, mdp.duration_focus_ms, mdp.duration_saccade_ms = do_transition(mdp, mdp.state, mdp.action)
    mdp.action_duration = mdp.duration_focus_ms + mdp.duration_saccade_ms

    mdp.gaze_location = mdp.state.focus
    mdp.n_actions += 1

end


function _observe_relevance_at(mdp::SubMenu, state::State, focus::Int64)
    state.obs_items[focus+1].item_relevance = mdp.items[focus+1].item_relevance
    return state
end


function _observe_length_at(mdp::SubMenu, state::State, focus::Int64)
    state.obs_items[focus+1].item_length = mdp.items[focus+1].item_length
    return state
end


function do_transition(mdp::SubMenu, state::State, action::Int64)
    state = deepcopy(state)

    if mdp.n_actions == 0
        if Base.rand() < Float64(mdp.menu_recall_probability)
            state.obs_items = [deepcopy(item) for item in mdp.items]
        end
    end

    if action == Action().QUIT
        state.quit = true
        focus_duration = 0
        saccade_duration = 0
    else
        # saccade
        # item_locations are off-by-one to other lists
        if state.focus != Focus().ABOVE_MENU
            amplitude = abs(mdp.item_locations[state.focus+2] - mdp.item_locations[action+2])
        else
            amplitude = abs(mdp.item_locations[1] - mdp.item_locations[action+2])
        end

        saccade_duration = Int64(floor(37 + 2.7 * amplitude))
        state.focus = action

        # fixation
        focus_duration = Int64(floor(mdp.focus_duration_100ms * 100))

        # semantic observation at focus
        state = _observe_relevance_at(mdp, state, state.focus)

        # possible length observations
        if mdp.length_observations
            if state.focus > 0 && Base.rand() < mdp.p_obs_len_adj
                state = _observe_length_at(mdp, state, state.focus-1)
            end

            if Base.rand() < mdp.p_obs_len_cur
                state = _observe_length_at(mdp, state, state.focus)
            end

            if state.focus < mdp.n_items-1 && Base.rand() < mdp.p_obs_len_adj
                state = _observe_length_at(mdp, state, state.focus+1)
            end
        end

        # possible semantic peripheral observations
        if state.focus > 0 && Base.rand() < Float64(mdp.p_obs_adjacent)
            state = _observe_relevance_at(mdp, state, state.focus-1)
        end

        if state.focus < mdp.n_items-1 && Base.rand() < Float64(mdp.p_obs_adjacent)
            state = _observe_relevance_at(mdp, state, state.focus+1)
        end

        # found target -> will click
        if state.focus != Focus().ABOVE_MENU && state.obs_items[state.focus+1].item_relevance == ItemRelevance().TARGET_RELEVANCE
            focus_duration += Int64(floor(mdp.selection_delay_s * 1000))
        end
    end

    return state, focus_duration, saccade_duration

end


function has_found_item(mdp::SubMenu)
    return mdp.state.focus != Focus().ABOVE_MENU && mdp.state.obs_items[mdp.state.focus+1].item_relevance == ItemRelevance().TARGET_RELEVANCE
end


function has_quit(mdp::SubMenu)
    return mdp.state.quit
end


function getSensors(mdp::SubMenu)
    return mdp.state.obs_items
end


function _semantic(mdp::SubMenu, n_groups::Int64, n_each_group::Int64, semantic_type::Int64)

    n_items = n_groups * n_each_group
    target_value = 1

    absent_menu_parameters      = [2.1422, 13.4426]
    non_target_group_parameters = [5.3665, 18.8826]
    target_group_parameters     = [3.1625,  1.2766]

    semantic_menu = zeros(1, n_items)

    target_location = Base.rand(1:n_items)

    # Only irrelevant elements
    if semantic_type == 4
        menu1 = Base.rand(Distributions.Beta(absent_menu_parameters...), (1,n_items))
        target_location = nothing
    # Relevant and irrelevant elements
    elseif semantic_type == 3
        target_group_samples = Base.rand(Distributions.Beta(target_group_parameters...), (n_each_group,))
        distractor_group_samples = Base.rand(Distributions.Beta(non_target_group_parameters...), (n_items,))

        menu1 = distractor_group_samples
        target_in_group = Int64(ceil((target_location) / Float64(n_each_group)))

        b = (target_in_group - 1) * n_each_group + 1
        e = (target_in_group - 1) * n_each_group + n_each_group

        menu1[b:e] = target_group_samples
        menu1[target_location] = target_value

    # Only relevant elements
    elseif semantic_type == 2
        menu1 = Base.rand(Distributions.Beta(target_group_parameters...), (n_items,))
        menu1[target_location] = target_value
    end

    semantic_menu = menu1

    return semantic_menu, target_location

end


function _get_unordered_menu(mdp::SubMenu, n_groups, n_each_group, n_grids, p_absent)
    return nothing
end


function _griding(mdp::SubMenu, menu, target, n_levels)
    start = 1 / Float64(2 * n_levels)
    stop  = 1
    step  = 1 / Float64(n_levels)

    np_menu = reshape(menu, (1, size(menu)...))
    griding_semantic_levels = collect(start:step:stop)

    if target != nothing
        temp_levels = permutedims(abs.(np_menu .- repeat(griding_semantic_levels, 1, 8)), [2,1])
        _, id = findmin(temp_levels,dims=2)
        min_index = map(x -> x[2], Tuple.(id[:,1,:]))
        gridded_menu = griding_semantic_levels[min_index]
        gridded_menu[target] = 1
    else
        temp_levels = permutedims(abs.(np_menu .- repeat(griding_semantic_levels, 1, 1, 8)), [3,2,1])
        _, id = findmin(temp_levels,dims=3)
        min_index = map(x -> x[3], Tuple.(id[:,1,:]))
        gridded_menu = griding_semantic_levels[min_index]
    end

    return transpose(gridded_menu)

end


function _get_semantic_menu(mdp::SubMenu, n_groups::Int64, n_each_group::Int64, n_grids::Int64, semantic_type::Int64)

    menu, target = _semantic(mdp, n_groups, n_each_group, semantic_type)
    gridded_menu = _griding(mdp, menu, target, n_grids)

    menu_length = n_each_group * n_groups
    coded_menu = [MenuItem(ItemRelevance().LOW_RELEVANCE, ItemLength().NOT_OBSERVED) for i in 1:menu_length]

    start = 1 / Float64(2 * n_grids)
    stop  = 1
    step  = 1 / Float64(n_grids)

    grids = collect(start:step:stop)

    count = 1

    for item in gridded_menu

        if 0 == (item .- grids[1])
            coded_menu[count] = MenuItem(ItemRelevance().LOW_RELEVANCE, ItemLength().NOT_OBSERVED)
        elseif 0 == (item .- grids[2])
            coded_menu[count] = MenuItem(ItemRelevance().MED_RELEVANCE, ItemLength().NOT_OBSERVED)
        elseif 0 == (item .- grids[3])
            coded_menu[count] = MenuItem(ItemRelevance().HIGH_RELEVANCE, ItemLength().NOT_OBSERVED)
        end

        count += 1
    end

    return coded_menu, target

end
                

function getReward(mdp::SubMenu)

    reward_success = 10000
    reward_failure = -10000

    if has_found_item(mdp)
        return reward_success
    elseif has_quit(mdp)
        if mdp.target_present
            return reward_failure
        else
            return reward_success
        end
    end

    return Int64(-1 * mdp.action_duration)

end


function isFinished(mdp::SubMenu)

    max_number_of_actions_per_session = 500

    if mdp.n_actions >= max_number_of_actions_per_session
        return true
    elseif has_found_item(mdp)
        return true
    elseif has_quit(mdp)
        return true
    end

    return false

end



# POMDP definition with generative interface
function POMDPs.gen(mdp::SubMenu, state, action, rng)

    if isFinished(mdp)
        r = 0
    else
        perform_action(mdp, action)
        r = getReward(mdp)
    end

    sp = mdp.state
    o = sp

    return (sp=sp, o=o, r=r)

end

# Initial state distribution
function POMDPs.initialstate(mdp::SubMenu)

    # Reset mdp on init
    reset(mdp)

    return DeterministicState(State([MenuItem(ItemRelevance().NOT_OBSERVED, ItemLength().NOT_OBSERVED) for i in 1:mdp.n_items],
                                    Focus().ABOVE_MENU,
                                    false))

end

# Action space
POMDPs.actions(::SubMenu) = [0,1,2,3,4,5,6,7,8]

# Discount
POMDPs.discount(::SubMenu) = 0.9


POMDPs.isterminal(mdp::SubMenu, s::State) = isFinished(mdp)


mutable struct HState
    obs_items::Vector{MenuItem}
    focus::Int64
    quit::Bool
    current_menu::Int64

    function HState(obs_items::Vector{MenuItem}, focus::Int64, quit::Bool, current_menu::Int64)
        new(obs_items, focus, quit, current_menu)
    end
end

mutable struct HierarchicalSearchEnvironment <: MDP{State, Int64}

    main_menu::Union{SubMenu, Nothing}
    sub_menus::Union{Vector{SubMenu}, Nothing}
    log_labels::Union{Matrix{Int64}, Nothing}
    rel_labels::Union{Matrix{Int64}, Nothing}
    target_menu::Union{Int64, Nothing}
    menu_recall_probability::Union{Float64, Nothing}
    focus_duration_100ms::Union{Float64, Nothing}                    
    selection_delay_s::Union{Float64, Nothing}
    current_menu::Union{Int64, Nothing}
    length_permutation::Union{Bool, Nothing}

    action_duration::Union{Float64, Nothing}
    has_quit::Union{Bool, Nothing}
    has_found_item::Union{Bool, Nothing}
    n_actions::Union{Int64, Nothing}
    training::Union{Bool, Nothing}
    init_menu::Union{Bool, Nothing}
    max_number_of_actions_per_session::Int64

    full_reward::Union{Int64, Nothing}

    function HierarchicalSearchEnvironment()
        new(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, 150, nothing)
    end

end

function reset(mdp::HierarchicalSearchEnvironment)

    logical_groups = collect(1:8)
    
    # Sample logical groups for menus
    group1 = shuffle(logical_groups)
    group2 = shuffle(logical_groups)

    # Menu compositions (logical groups)
    mdp.log_labels = hcat(group1, group2)

    # Assign semantic levels to logical groups
    # (Random assignment with structural correlation)
    relevancies = [4,4,4,3,3,2,3,3]
    relevancies = circshift(relevancies, Base.rand(1:8))

    # Menu relevancy groups
    rel1 = map(x -> relevancies[x], group1)
    rel2 = map(x -> relevancies[x], group2)

    # Menu compositions (semantic groups)
    mdp.rel_labels = hcat(rel1, rel2)

    mdp.sub_menus = []
    main_menu_items = []

    # Select target group

    weights = (sum(mdp.rel_labels .== 2, dims=2) ./ 2)[:,1]
    mdp.target_menu = StatsBase.sample(collect(1:8), ProbabilityWeights(weights), 1)[1]

    for i in 1:8
        groups = mdp.rel_labels[i,:]
        log_groups = mdp.log_labels[i,:]

        semantic_type = mean(groups)
        semantic_type = Int64(Base.rand([floor(semantic_type), ceil(semantic_type)]))

        # Init new submenu
        submenu = SubMenu()

        submenu.menu_recall_probability = mdp.menu_recall_probability
        submenu.focus_duration_100ms    = mdp.focus_duration_100ms
        submenu.selection_delay_s       = mdp.selection_delay_s

        submenu.semantic_type = semantic_type

        if i == mdp.target_menu
            submenu.fix_target_present = true
            # During training, randomly mark target menus as assistant suggestions
            if mdp.training && Base.rand(0:2) == 0
                semantic_type = 1
            end
        end

        reset(submenu)
        
        push!(main_menu_items, MenuItem(semantic_type, 1))
        push!(mdp.sub_menus, submenu)

    end

    mdp.main_menu = SubMenu()
    mdp.main_menu.menu_recall_probability = 0.0
    mdp.main_menu.focus_duration_100ms    = mdp.focus_duration_100ms
    mdp.main_menu.selection_delay_s       = mdp.selection_delay_s

    mdp.main_menu.fixed_elements = main_menu_items

    reset(mdp.main_menu)

    # Init at main menu
    mdp.current_menu = 0
    mdp.action_duration = 0
    mdp.has_quit = false
    mdp.has_found_item = false
    mdp.n_actions = 0
    mdp.length_permutation = Bool(Base.rand(0:1))

    mdp.full_reward = 100000

    if mdp.training
        if mdp.init_menu
            mdp.current_menu = 0
        else
            if Base.rand(0:2) == 2
                mdp.current_menu = Base.rand(1:8)
            else
                mdp.current_menu = mdp.target_menu
            end
        end
        mdp.full_reward = 10000
    end

end

function perform_action(mdp::HierarchicalSearchEnvironment, action::Int64)

    # If in submenu
    if mdp.current_menu != 0
        # Check if submenu is not finished
        if !isFinished(mdp.sub_menus[mdp.current_menu])
            # Pass action to submenu
            perform_action(mdp.sub_menus[mdp.current_menu], action)
            # Get action duration
            mdp.action_duration = mdp.sub_menus[mdp.current_menu].action_duration
            # If item is found, complete the full mdp
            if has_found_item(mdp.sub_menus[mdp.current_menu])
                mdp.has_found_item = true
            end
            # Check if user has quit
            if action == Action().QUIT
                # If training (i.e. only one submenu), reward quit success accordingly
                if mdp.training
                    if mdp.current_menu == mdp.target_menu
                        # If false quit
                        mdp.has_quit = true
                    else
                        # I.e. successful quit (just to emit the reward)
                        mdp.has_found_item = true
                    end
                # If full setting, proceed to main menu and mark the submenu as visited
                else
                    mdp.action_duration = mdp.main_menu.selection_delay_s * 1000

                    mdp.current_menu = 0
                    mdp.main_menu.state.obs_items[mdp.main_menu.state.focus+1].item_relevance = 4
                    mdp.main_menu.items[mdp.main_menu.state.focus+1].item_relevance = 4
                    mdp.main_menu.state.obs_items[mdp.main_menu.state.focus+1].item_length = 2
                    mdp.main_menu.items[mdp.main_menu.state.focus+1].item_length = 2
                end
            end
        # If submenu is already finished
        else
            # Pass action to submenu (should have no effect)
            perform_action(mdp.sub_menus[mdp.current_menu], action)
            mdp.action_duration = 0
            # If training menu is quit, end the scenario
            if mdp.training && has_quit(mdp.sub_menus[mdp.current_menu])
                mdp.has_quit = true
            end
        end

    else
        
        if action == Action().QUIT
            mdp.action_duration = 0
            mdp.has_quit = true
        else
            if action == mdp.main_menu.state.focus
                mdp.action_duration = mdp.main_menu.selection_delay_s * 1000
                if mdp.training
                    if mdp.main_menu.state.focus+1 == mdp.target_menu
                        mdp.has_found_item = true
                    else
                        # Reward sensible menu choices accordingly (to speedup training)
                        if mdp.main_menu.state.obs_items[mdp.main_menu.state.focus+1].item_relevance == 2
                            mdp.full_reward = mdp.full_reward / 2
                            mdp.action_duration = - mdp.full_reward
                        end
                        if mdp.main_menu.state.obs_items[mdp.main_menu.state.focus+1].item_relevance == 3
                            mdp.full_reward = 2 * Int64(floor(mdp.full_reward / 3))
                            mdp.action_duration = - mdp.full_reward / 2
                        end
                        mdp.main_menu.state.obs_items[mdp.main_menu.state.focus+1].item_relevance = 4
                        mdp.main_menu.items[mdp.main_menu.state.focus+1].item_relevance = 4
                        mdp.main_menu.state.obs_items[mdp.main_menu.state.focus+1].item_length = 2
                        mdp.main_menu.items[mdp.main_menu.state.focus+1].item_length = 2
                    end
                else
                    mdp.current_menu = mdp.main_menu.state.focus+1
                end
            else
                mdp.action_duration = 0
                if !isFinished(mdp.main_menu)
                    perform_action(mdp.main_menu, action)
                    mdp.action_duration = mdp.main_menu.action_duration
                elseif has_quit(mdp.main_menu)
                    mdp.has_quit = true
                end
            end
        end
    end

    mdp.n_actions += 1

end




function getReward(mdp::HierarchicalSearchEnvironment)

    if mdp.training
        reward_success = mdp.full_reward
        reward_failure = -10000
    else
        reward_success = mdp.full_reward
        reward_failure = -100000
    end

    if mdp.has_found_item
        return reward_success + Int64(floor(-1 * mdp.action_duration))
    end

    if mdp.has_quit
        return reward_failure * 10 + Int64(floor(-1 * mdp.action_duration))
    end

    return Int64(floor(-1 * mdp.action_duration))

end


function isFinished(mdp::HierarchicalSearchEnvironment)

    if mdp.n_actions >= mdp.max_number_of_actions_per_session
        return true
    elseif mdp.has_found_item
        return true
    elseif mdp.has_quit
        return true
    end

    return false

end


# POMDP definition with generative interface
function POMDPs.gen(mdp::HierarchicalSearchEnvironment, state, action, rng)

    if isFinished(mdp)
        r = 0
    else
        perform_action(mdp, action)
        r = getReward(mdp)
    end

    if mdp.current_menu == 0
        sp = mdp.main_menu.state
    else
        sp = mdp.sub_menus[mdp.current_menu].state
    end

    o = sp

    return (sp=sp, o=o, r=r)

end

function perform_action(mdp::HierarchicalSearchEnvironment, action::Vector{Int64})
    perform_action(mdp, action[1])
end


struct DeterministicHState
    s::HState
end

Base.rand(d::DeterministicHState) = d.s

# Initial state distribution
function POMDPs.initialstate(mdp::HierarchicalSearchEnvironment)

    # Reset mdp on init
    reset(mdp)

    return DeterministicState(State([MenuItem(ItemRelevance().NOT_OBSERVED, ItemLength().NOT_OBSERVED) for i in 1:8],
                                    Focus().ABOVE_MENU,
                                    false))

end

CommonRLInterface.valid_action_mask(x::MDPCommonRLEnv{AbstractArray{Float32, N} where N, HierarchicalSearchEnvironment, State}) = nothing

# Action space
POMDPs.actions(::HierarchicalSearchEnvironment) = [0,1,2,3,4,5,6,7,8]

# Discount
POMDPs.discount(::HierarchicalSearchEnvironment) = 1.0


POMDPs.isterminal(mdp::HierarchicalSearchEnvironment, s::State) = isFinished(mdp)


function POMDPs.convert_s(T::Type{A1}, s::A2, problem::Union{MDP, POMDP}) where {A1<:AbstractArray, A2<:HState}
    
    items = [Float64.([i.item_relevance, i.item_length]) for i in s.obs_items] |> vec
    focus = Float64(s.focus)
    quit  = Float64(s.quit)
    current_menu = Float64(s.current_menu)

    return convert(T, cat(items..., focus, quit, current_menu, dims=1))

end


function POMDPs.convert_s(T::Type{A1}, v::A2, problem::Union{MDP, POMDP}) where {A1<:HState, A2<:AbstractArray}
    
    return HState([MenuItem(Int64.(v[2i-1:2i])...) for i in 1:8], Int64(v[end-2]), Bool(v[end-1]), Int64(v[end]))

end


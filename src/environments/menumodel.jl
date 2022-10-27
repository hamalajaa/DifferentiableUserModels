export State, MenuItem, SearchEnvironment, reset, gen, isFinished


struct ItemRelevance
    NOT_OBSERVED::Int64
    TARGET_RELEVANCE::Int64
    HIGH_RELEVANCE::Int64
    MED_RELEVANCE::Int64
    LOW_RELEVANCE::Int64
    
    function ItemRelevance()
        new(0,1,2,3,4)
    end
    
end


struct ItemLength
    NOT_OBSERVED::Int64
    TARGET_LENGTH::Int64
    NOT_TARGET_LENGTH::Int64

    function ItemLength()
        new(0,1,2)
    end
end


mutable struct Focus
    ITEM_1::Int64
    ITEM_2::Int64
    ITEM_3::Int64
    ITEM_4::Int64
    ITEM_5::Int64
    ITEM_6::Int64
    ITEM_7::Int64
    ITEM_8::Int64
    ABOVE_MENU::Int64

    function Focus()
        new(0,1,2,3,4,5,6,7,8)
    end
end


struct Action
    LOOK_1::Int64
    LOOK_2::Int64
    LOOK_3::Int64
    LOOK_4::Int64
    LOOK_5::Int64
    LOOK_6::Int64
    LOOK_7::Int64
    LOOK_8::Int64
    QUIT::Int64

    function Action()
        new(0,1,2,3,4,5,6,7,8)
    end
end


mutable struct MenuItem
    item_relevance::Int64
    item_length::Int64

    function MenuItem(item_relevance::Int64, item_length::Int64)
        new(item_relevance, item_length)
    end
end

#https://hal.sorbonne-universite.fr/hal-02063155/document
#https://github.com/JuliaPOMDP/TabularTDLearning.jl



mutable struct State
    obs_items::Vector{MenuItem}
    focus::Int64
    quit::Bool

    function State(obs_items::Vector{MenuItem}, focus::Int64, quit::Bool)
        new(obs_items, focus, quit)
    end
end


struct Menu
    items::Vector{MenuItem}
    target_present::Bool
    target_idx::Union{Int64, Nothing}

    function Menu(items::Vector{MenuItem}, target_present, target_idx)
        new(items, target_present, target_idx)
    end
end



mutable struct SearchEnvironment <: MDP{State, Int64}

    menu_type::String
    menu_groups::Int64
    menu_items_per_group::Int64
    semantic_levels::Int64
    gap_between_items::Float64
    prop_target_absent::Float64
    length_observations::Bool
    p_obs_len_cur::Float64
    p_obs_len_adj::Float64
    n_training_menus::Int64
    training_menus::Vector{Menu}
    training::Bool
    n_item_lengths::Int64

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
    

    function SearchEnvironment(;menu_type::String="semantic",
                                menu_groups::Int64=2,
                                menu_items_per_group::Int64=4,
                                semantic_levels::Int64=3,
                                gap_between_items::Float64=0.75,
                                prop_target_absent::Float64=0.1,
                                length_observations::Bool=true,
                                p_obs_len_cur::Float64=0.95,
                                p_obs_len_adj::Float64=0.89,
                                n_training_menus::Int64=10000,
                                training_menus::Vector{Menu}=Vector{Menu}(),
                                training::Bool=true,
                                n_item_lengths::Int64=3,
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
            n_training_menus,
            training_menus,
            training,
            n_item_lengths,
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


function _get_menu(mdp::SearchEnvironment)

    if mdp.training && length(mdp.training_menus) >= mdp.n_training_menus
        idx = Base.rand(1:mdp.n_training_menus)
        return mdp.training_menus[idx]
    end
    # generate menu item semantic relevances and lengths
    items = Vector()
    if mdp.menu_type == "semantic"
        items, target_idx = _get_semantic_menu(mdp, mdp.menu_groups,
                                               mdp.menu_items_per_group, 
                                               mdp.semantic_levels, 
                                               mdp.prop_target_absent)
    elseif mdp.menu_type == "unordered"
        items, target_idx = _get_unordered_menu(mdp, mdp.menu_groups,
                                                mdp.menu_items_per_group, 
                                                mdp.semantic_levels, 
                                                mdp.prop_target_absent)
    else
        error("Unknown menu type")
    end

    lengths = Base.rand(0:mdp.n_item_lengths-1, length(items))
    target_present = (target_idx != nothing)

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

    if mdp.training
        append!(mdp.training_menus, [menu])
    end
        
    return menu
end


function reset(mdp::SearchEnvironment)

    menu = _get_menu(mdp)

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


function perform_action(mdp::SearchEnvironment, action::Int64)

    # ???
    mdp.action = action
    mdp.prev_state = deepcopy(mdp.state)
    mdp.state, mdp.duration_focus_ms, mdp.duration_saccade_ms = do_transition(mdp, mdp.state, mdp.action)
    mdp.action_duration = mdp.duration_focus_ms + mdp.duration_saccade_ms

    mdp.gaze_location = mdp.state.focus
    mdp.n_actions += 1

end


function _observe_relevance_at(mdp::SearchEnvironment, state::State, focus::Int64)
    state.obs_items[focus+1].item_relevance = mdp.items[focus+1].item_relevance
    return state
end


function _observe_length_at(mdp::SearchEnvironment, state::State, focus::Int64)
    state.obs_items[focus+1].item_length = mdp.items[focus+1].item_length
    return state
end


function do_transition(mdp::SearchEnvironment, state::State, action::Int64)
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


function has_found_item(mdp::SearchEnvironment)
    return mdp.state.focus != Focus().ABOVE_MENU && mdp.state.obs_items[mdp.state.focus+1].item_relevance == ItemRelevance().TARGET_RELEVANCE
end


function has_quit(mdp::SearchEnvironment)
    return mdp.state.quit
end


# May require editing
function getSensors(mdp::SearchEnvironment)
    return mdp.state.obs_items
end


function _semantic(mdp::SearchEnvironment, n_groups::Int64, n_each_group::Int64, p_absent::Float64)

    n_items = n_groups * n_each_group
    target_value = 1

    absent_menu_parameters      = [2.1422, 13.4426]
    non_target_group_parameters = [5.3665, 18.8826]
    target_group_parameters     = [3.1625,  1.2766]

    semantic_menu = zeros(1, n_items)
    
    target_type = Base.rand()
    target_location = Base.rand(1:n_items)

    if target_type > p_absent
        target_group_samples = Base.rand(Distributions.Beta(target_group_parameters...), (n_each_group,))
        distractor_group_samples = Base.rand(Distributions.Beta(non_target_group_parameters...), (n_items,))

        menu1 = distractor_group_samples
        target_in_group = Int64(ceil((target_location) / Float64(n_each_group)))

        b = (target_in_group - 1) * n_each_group + 1
        e = (target_in_group - 1) * n_each_group + n_each_group

        menu1[b:e] = target_group_samples
        menu1[target_location] = target_value
        
    else
        target_location = nothing
        menu1 = Base.rand(Distributions.Beta(absent_menu_parameters...), (1,n_items))
    end

    semantic_menu = menu1

    return semantic_menu, target_location

end


function _get_unordered_menu(mdp::SearchEnvironment, n_groups, n_each_group, n_grids, p_absent)
    return nothing
end


function _griding(mdp::SearchEnvironment, menu, target, n_levels)
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


function _get_semantic_menu(mdp::SearchEnvironment, n_groups::Int64, n_each_group::Int64, n_grids::Int64, p_absent::Float64)

    menu, target = _semantic(mdp, n_groups, n_each_group, p_absent)
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
                

function getReward(mdp::SearchEnvironment)

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


function isFinished(mdp::SearchEnvironment)

    max_number_of_actions_per_session = 20

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
function POMDPs.gen(mdp::SearchEnvironment, state, action, rng)

    # Assert state == mdp.state

    if isFinished(mdp)
        r = 0
    else
        perform_action(mdp, action)
        r = getReward(mdp)
    end

    #println("States: ")
    #println(state)
    #println(mdp.state)

    #println("Action: ")
    #println(action)
    
    #println("Reward: ")
    #println(r)

    #println("")

    sp = mdp.state
    o = sp

    return (sp=sp, o=o, r=r)

end


struct DeterministicState
    s::State
end

Base.rand(d::DeterministicState) = d.s


# Initial state distribution
function POMDPs.initialstate(mdp::SearchEnvironment)

    # Reset mdp on init
    reset(mdp)

    return DeterministicState(State([MenuItem(ItemRelevance().NOT_OBSERVED, ItemLength().NOT_OBSERVED) for i in 1:mdp.n_items],
                                    Focus().ABOVE_MENU,
                                    false))

end

# Action space
POMDPs.actions(::SearchEnvironment) = [0,1,2,3,4,5,6,7,8]

# Discount
POMDPs.discount(::SearchEnvironment) = 0.9


function POMDPs.convert_s(T::Type{A1}, s::A2, problem::Union{MDP, POMDP}) where {A1<:AbstractArray, A2<:State}
    
    items = [Float64.([i.item_relevance, i.item_length]) for i in s.obs_items] |> vec
    focus = Float64(s.focus)
    quit  = Float64(s.quit)

    #ret = convert(T, cat(items..., focus, quit, dims=1))

    #println("To vector")
    #println(s)
    #println(ret)

    return convert(T, cat(items..., focus, quit, dims=1))

end


function POMDPs.convert_s(T::Type{A1}, v::A2, problem::Union{MDP, POMDP}) where {A1<:State, A2<:AbstractArray}

    #ret = State([MenuItem(Int64.(v[2i-1:2i])...) for i in 1:8], Int64(v[end-1]), Bool(v[end]))

    #println("To state")
    #println(v)
    #println(ret)

    
    return State([MenuItem(Int64.(v[2i-1:2i])...) for i in 1:8], Int64(v[end-1]), Bool(v[end]))

end


POMDPs.isterminal(mdp::SearchEnvironment, s::State) = isFinished(mdp)





































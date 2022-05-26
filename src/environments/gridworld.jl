export GridWorld, GridWorldState, GridWorldStates

# Based on the following tutorial:
#"https://github.com/JuliaPOMDP/POMDPExamples.jl/blob/master/notebooks/GridWorld.ipynb"

struct GridWorldState 
    x::Int64 # x position
    y::Int64 # y position
    done::Bool # are we in a terminal state?

    # initial state constructor
    function GridWorldState(x::Int64, y::Int64, done::Bool=false)
        new(x,y,done)
    end

end

# initial state constructor
#GridWorldState(x::Int64, y::Int64) = GridWorldState(x,y,false)
# checks if the position of two states are the same
posequal(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y



# Grid world MDP
mutable struct GridWorld <: MDP{GridWorldState, Symbol} # Note that our MDP is parametarized by the state and the action
    size_x::Int64 # x size of the grid
    size_y::Int64 # y size of the grid
    reward_states::Vector{GridWorldState} # the states in which agent recieves reward
    reward_values::Vector{Float64} # reward values for those states
    tprob::Float64 # probability of transitioning to the desired state
    discount_factor::Float64 # disocunt factor
end



#we use key worded arguments so we can change any of the values we pass in 
function GridWorld(;sx::Int64=10, # size_x
    sy::Int64=10, # size_y
    rs::Vector{GridWorldState}=[GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)], # reward states
    rv::Vector{Float64}=rv = [-10.,-5,10,3], # reward values
    tp::Float64=0.7, # tprob
    discount_factor::Float64=0.9)
return GridWorld(sx, sy, rs, rv, tp, discount_factor)
end

# we can now create a GridWorld mdp instance like this:
mdp = GridWorld()
mdp.reward_states # mdp contains all the defualt values from the constructor


# MDP state space

function POMDPs.states(mdp::GridWorld)
    s = GridWorldState[] # initialize an array of GridWorldStates
    # loop over all our states, remeber there are two binary variables:
    # done (d)
    for d = 0:1, y = 1:mdp.size_y, x = 1:mdp.size_x
        push!(s, GridWorldState(x,y,Bool(d)))
    end
    return s
end

mdp = GridWorld()
state_space = states(mdp);
state_space[1]


# Action space

POMDPs.actions(mdp::GridWorld) = [:up, :down, :left, :right];


####################
# Transition model #
####################

# transition helpers
function inbounds(mdp::GridWorld,x::Int64,y::Int64)
    if 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y
        return true
    else
        return false
    end
end

inbounds(mdp::GridWorld, state::GridWorldState) = inbounds(mdp, state.x, state.y);


function POMDPs.transition(mdp::GridWorld, state::GridWorldState, action::Symbol)
    a = action
    x = state.x
    y = state.y
    
    if state.done
        return SparseCat([GridWorldState(x, y, true)], [1.0])
    elseif state in mdp.reward_states
        return SparseCat([GridWorldState(x, y, true)], [1.0])
    end

    neighbors = [
        GridWorldState(x+1, y, false), # right
        GridWorldState(x-1, y, false), # left
        GridWorldState(x, y-1, false), # down
        GridWorldState(x, y+1, false), # up
        ] # See Performance Note below
    
    targets = Dict(:right=>1, :left=>2, :down=>3, :up=>4) # See Performance Note below
    target = targets[a]
    
    probability = fill(0.0, 4)

    if !inbounds(mdp, neighbors[target])
        # If would transition out of bounds, stay in
        # same cell with probability 1
        return SparseCat([GridWorldState(x, y)], [1.0])
    else
        probability[target] = mdp.tprob

        oob_count = sum(!inbounds(mdp, n) for n in neighbors) # number of out of bounds neighbors

        new_probability = (1.0 - mdp.tprob)/(3-oob_count)

        for i = 1:4 # do not include neighbor 5
            if inbounds(mdp, neighbors[i]) && i != target
                probability[i] = new_probability
            end
        end
    end

    return SparseCat(neighbors, probability)
end



################
# Reward model #
################

function POMDPs.reward(mdp::GridWorld, state::GridWorldState, action::Symbol, statep::GridWorldState) #deleted action
    if state.done
        return 0.0
    end
    r = 0.0
    n = length(mdp.reward_states)
    for i = 1:n
        if posequal(state, mdp.reward_states[i])
            r += mdp.reward_values[i]
        end
    end
    return r
end


# Misc

POMDPs.discount(mdp::GridWorld) = mdp.discount_factor


function POMDPs.stateindex(mdp::GridWorld, state::GridWorldState)
    sd = Int(state.done + 1)
    ci = CartesianIndices((mdp.size_x, mdp.size_y, 2))
    return LinearIndices(ci)[state.x, state.y, sd]
end

function POMDPs.actionindex(mdp::GridWorld, act::Symbol)
    if act==:up
        return 1
    elseif act==:down
        return 2
    elseif act==:left
        return 3
    elseif act==:right
        return 4
    end
    error("Invalid GridWorld action: $act")
end


POMDPs.isterminal(mdp::GridWorld, s::GridWorldState) = s.done

POMDPs.initialstate(pomdp::GridWorld) = Deterministic(GridWorldState(1,1)) 




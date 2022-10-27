using ArgParse
using BSON
using Distributions
using Statistics
using Flux
using Stheno
using Tracker
using CUDA

# Addirional imports
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators

using MCTS
using DeepQLearning
using POMDPModels

include("../../NeuralProcesses.jl/src/NeuralProcesses.jl")
include("../../NeuralProcesses.jl/src/experiment/experiment.jl")

using .NeuralProcesses
using .NeuralProcesses.Experiment

args = Dict("variant" => "v4", "n_traj"=>0, "plot"=>false, "p_bias"=>0.0, "perm"=>false, "params"=>false);


# MAML assistant

using Transformers
using Transformers.Basic

model = BSON.load("models/ex3/maml_v1/1000.bson")[:model]
#model = NeuralProcesses.Experiment.recent_model("models/ex3/anp/499.bson")

# Init full env
mdp1 = HierarchicalSearchEnvironment()
mdp1.training = false
mdp1.init_menu = true
mdp1.max_number_of_actions_per_session = 50

# Init train envs
mdp2 = HierarchicalSearchEnvironment()
mdp2.training = true
mdp2.init_menu = true
mdp2.max_number_of_actions_per_session = 20

mdp3 = HierarchicalSearchEnvironment()
mdp3.training = true
mdp3.init_menu = false
mdp3.max_number_of_actions_per_session = 20

mdp = MenuAssistant()
mdp.env        = mdp1
mdp.env_main   = mdp2
mdp.env_sub    = mdp3
mdp.user_model = model
mdp.data_gen   = NeuralProcesses.HierarchicalMenuSampler(args;)

mdp.anp_assistant = false
mdp.maml_assistant = true

sp = 0.0
times_maml = []

for i in 1:100
    
    POMDPs.initialstate(mdp)
    sp = 0.0
    
    for j in 1:30

        if mdp.env.current_menu == 0
            s = mdp.env.main_menu.state
        else
            s = mdp.env.sub_menus[mdp.env.current_menu].state
        end

        sp, o, r = gen(mdp, s, 0, 0)
        
        if sp[1] == 1.0
            push!(times_maml, r)
            break
        end
    end
end


println("MAML complete...")
println(times_maml)


# ANP assistant

model = NeuralProcesses.Experiment.recent_model("models/ex3/anp/499.bson")

# Init full env
mdp1 = HierarchicalSearchEnvironment()
mdp1.training = false
mdp1.init_menu = true
mdp1.max_number_of_actions_per_session = 50

# Init train envs
mdp2 = HierarchicalSearchEnvironment()
mdp2.training = true
mdp2.init_menu = true
mdp2.max_number_of_actions_per_session = 20

mdp3 = HierarchicalSearchEnvironment()
mdp3.training = true
mdp3.init_menu = false
mdp3.max_number_of_actions_per_session = 20

mdp = MenuAssistant()
mdp.env        = mdp1
mdp.env_main   = mdp2
mdp.env_sub    = mdp3
mdp.user_model = model
mdp.data_gen   = NeuralProcesses.HierarchicalMenuSampler(args;)

sp = 0.0
times_anp = []

for i in 1:100
    
    POMDPs.initialstate(mdp)
    sp = 0.0
    
    for j in 1:30

        if mdp.env.current_menu == 0
            s = mdp.env.main_menu.state
        else
            s = mdp.env.sub_menus[mdp.env.current_menu].state
        end

        sp, o, r = gen(mdp, s, 0, 0)
        
        if sp[1] == 1.0
            push!(times_anp, r)
            break
        end
    end
end

println("ANP complete...")
println(times_anp)

# No assistance

model = NeuralProcesses.Experiment.recent_model("models/ex3/anp/499.bson")

# Init full env
mdp1 = HierarchicalSearchEnvironment()
mdp1.training = false
mdp1.init_menu = true
mdp1.max_number_of_actions_per_session = 50

# Init train envs
mdp2 = HierarchicalSearchEnvironment()
mdp2.training = true
mdp2.init_menu = true
mdp2.max_number_of_actions_per_session = 20

mdp3 = HierarchicalSearchEnvironment()
mdp3.training = true
mdp3.init_menu = false
mdp3.max_number_of_actions_per_session = 20

mdp = MenuAssistant()
mdp.env        = mdp1
mdp.env_main   = mdp2
mdp.env_sub    = mdp3
mdp.user_model = model
mdp.data_gen   = NeuralProcesses.HierarchicalMenuSampler(args;)

mdp.enable_assistant = false

sp = 0.0
times_ = []

for i in 1:100
    
    POMDPs.initialstate(mdp)
    sp = 0.0
    
    for j in 1:30

        if mdp.env.current_menu == 0
            s = mdp.env.main_menu.state
        else
            s = mdp.env.sub_menus[mdp.env.current_menu].state
        end

        sp, o, r = gen(mdp, s, 0, 0)
        
        if sp[1] == 1.0
            push!(times_, r)
            break
        end
    end
end

println("No assistance complete...")
println(times_)

# Full observability

model = NeuralProcesses.Experiment.recent_model("models/ex3/anp/499.bson")

# Init full env
mdp1 = HierarchicalSearchEnvironment()
mdp1.training = false
mdp1.init_menu = true
mdp1.max_number_of_actions_per_session = 50

# Init train envs
mdp2 = HierarchicalSearchEnvironment()
mdp2.training = true
mdp2.init_menu = true
mdp2.max_number_of_actions_per_session = 20

mdp3 = HierarchicalSearchEnvironment()
mdp3.training = true
mdp3.init_menu = false
mdp3.max_number_of_actions_per_session = 20

mdp = MenuAssistant()
mdp.env        = mdp1
mdp.env_main   = mdp2
mdp.env_sub    = mdp3
mdp.user_model = model
mdp.data_gen   = NeuralProcesses.HierarchicalMenuSampler(args;)

mdp.observe_target = true

sp = 0.0
times_iis = []

for i in 1:100
    
    POMDPs.initialstate(mdp)
    sp = 0.0
    
    for j in 1:30

        if mdp.env.current_menu == 0
            s = mdp.env.main_menu.state
        else
            s = mdp.env.sub_menus[mdp.env.current_menu].state
        end

        sp, o, r = gen(mdp, s, 0, 0)
        
        if sp[1] == 1.0
            push!(times_iis, r)
            break
        end
    end
end

println("Full observability complete...")
println(times_iis)

# Transformer assistant

using Transformers
using Transformers.Basic

model = BSON.load("models/ex3/transformer/10000.bson")[:model]

# Init full env
mdp1 = HierarchicalSearchEnvironment()
mdp1.training = false
mdp1.init_menu = true
mdp1.max_number_of_actions_per_session = 50

# Init train envs
mdp2 = HierarchicalSearchEnvironment()
mdp2.training = true
mdp2.init_menu = true
mdp2.max_number_of_actions_per_session = 20

mdp3 = HierarchicalSearchEnvironment()
mdp3.training = true
mdp3.init_menu = false
mdp3.max_number_of_actions_per_session = 20

mdp = MenuAssistant()
mdp.env        = mdp1
mdp.env_main   = mdp2
mdp.env_sub    = mdp3
mdp.user_model = model
mdp.data_gen   = NeuralProcesses.HierarchicalMenuSampler(args;)

mdp.anp_assistant = false

sp = 0.0
times_tf = []

for i in 1:100
    
    POMDPs.initialstate(mdp)
    sp = 0.0
    
    for j in 1:30

        if mdp.env.current_menu == 0
            s = mdp.env.main_menu.state
        else
            s = mdp.env.sub_menus[mdp.env.current_menu].state
        end

        sp, o, r = gen(mdp, s, 0, 0)
        
        if sp[1] == 1.0
            push!(times_tf, r)
            break
        end
    end
end


println("Transformer complete...")
println(times_tf)
    

bson("results/ex3/assistants_v1.bson", times_anp=times_anp, times_=times_, times_iis=times_iis, times_tf=times_tf, times_maml=times_maml)
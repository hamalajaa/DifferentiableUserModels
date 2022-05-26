# Differentiable user models

This repository contains the code used to produce the results of the paper "Differentiable user models". 

This code is built on top of the NeuralProcesses.jl library (https://github.com/wesselb/NeuralProcesses.jl). Respectively, **we emphasize that the code included in the NeuralProcesses.jl folder in this project does not represent our contribution** and is only slightly modified for the purposes of this work.

### Running the experiments

The ANP model can be trained for the experiment settings introduced in the paper with the following commands:
```
$ julia --project=Project.toml experiments/ex[n]/experiment[n].jl
```
where `[n]` corresponds to the number of the experiment.

In the first two experiments, the trained models can be straightforwardly evaluated with:
```
$ julia --project=Project.toml experiments/ex[n]/ex[n]_test.jl
```
More detailed documentation on the remaining experiments will be added shortly.

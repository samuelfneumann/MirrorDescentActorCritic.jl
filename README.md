# MirrorDescentActorCritic.jl

Tabular, Linear, and Deep Actor-Critic algorithms implemented with
[Lux.jl](https://github.com/LuxDL/Lux.jl).

## Installation

First, you'll need to get the
[ExtendedDistributions.jl](https://github.com/samuelfneumann/ExtendedDistributions.jl)
package:
```bash
git clone git@github.com:samuelfneumann/ExtendedDistributions.jl.git
cd ExtendedDistributions
julia --project
```
then:
```juliaREPL
julia> ]instantiate
```

Then, clone this repo and jump into a Julia project
```bash
git clone git@github.com:samuelfneumann/MirrorDescentActorCritic.jl.git
cd MirrorDescentActorCritic.jl
julia --project
```
then
```juliaREPL
julia> ]instantiate
```

## Algorithms

This codebase separates algorithms into actor and critic updates. Each
algorithm is composed of each of these components, allowing different
actor and critic updates to be composed easily to create new algorithms/agents.
We provide implementations for the following standard actor updates, as well as
their functional mirror descent counterparts:

    - `CCEM`: Conditional Cross-Entropy Optimization [Link](https://arxiv.org/abs/1810.09103)
    - `RKL`: Reverse KL to the Boltzmann (SAC-like) [Link](https://www.jmlr.org/papers/volume23/21-054/21-054.pdf)
    - `FKL`: Forward KL to the Boltzmann [Link](https://www.jmlr.org/papers/volume23/21-054/21-054.pdf)
    - `MPO`: Maximum A-Posteriori (MPO-like) [Link](https://www.jmlr.org/papers/volume23/21-054/21-054.pdf)
    - `DualAveragingMPI`: Dual Averaging MPI [Link](https://arxiv.org/abs/2003.14089)
    - `REINFORCE`: REINFORCE-like update
    - `PPO`: PPO-like update

The following critic updates are implemented:
    - `Sarsa`

Critic updates can take a `BellmanRegulariser`, which alters the update to use
a regularised Bellman equation. The following regularisers are implemented:
    - `KLBellmanRegulariser`: KL regularization to the previous policy
    - `EntropyBellmanRegulariser`: Entropy regularizer/soft value functions

## Policies

This package has three kinds of continuous-action policies: `BoundedPolicy`,
`UnBoundedPolicy`, and `TruncatedPolicy`.

To create a policy, you simply provide a policy struct with a distribution from
the `ExtendedDistributions.jl` package. You should ensure that you provide
bounded distributions to `BoundedPolicy` types and unbounded or half-bounded
distributions to `UnBoundedPolicy` types.

Convenience constructors are provided for the following policies:

| Distribution  | Policy Type       | RSample Supported     |
|---------------|-------------------|-----------------------|
| Kumaraswamy   | `BoundedPolicy`   | yes (quantile method) |
| Beta          | `BoundedPolicy`   | no                    |
| ArctanhNormal | `BoundedPolicy`   | yes                   |
| LogitNormal   | `BoundedPolicy`   | yes                   |
| Normal        | `UnBoundedPolicy` | yes                   |
| Laplace       | `UnBoundedPolicy` | yes                   |

## Scheduling Experiments with SLURM

This codebase has been set up to easily schedule experiments with slurm.

There are two main directories for this, `config` and `parallel`. The `config`
directory hold subdirectories of `toml` config files. The `parallel` directory
holds subdirectories of `sh` jobscript files, which are scheduled with slurm
through `sbatch`.

Each of these directories should be mirror images of each other. That is, if a
file exists at `config/X/Y/Z.toml`, then a corresponding file should exists in
`parallel/X/Y/Z.sh`. Here, `config/X/Y/Z.toml` outlines the experiment to run,
including hyperparameter sweeps. Hyperparameters are swept using
`Reproduce.jl`. The `paralell/X/Y/Z.sh` talks to slurm and tells it what to
in order to run the experiment outlined by the corresponding config file.

To run an experiment, first create the `config/X/Y/Z.toml` file. A template is
provided at `config/template.toml`. Then, create the corresponding
`parallel/X/Y/Z.sh` file. A template is provided at `parallel/template.toml`.
You'll need to fill in the preamble to tell slurm what resources you need.
Then, fill in the lines marked with `TODO`. Finally, schedule the experiment
using `sbatch parallel/X/Y/Z.sh`. Alternatively, you can use the
`scripts/seqbatch.sh` file to automatically schedule sequential slurm jobs with
dependencies. Note that if your experiment does not finish and times out, then
rescheduling the experiment will continue from where the experiment left off.

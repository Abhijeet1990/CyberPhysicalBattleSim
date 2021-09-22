#!/usr/bin/python3.8

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
"""CLI to run the baseline Deep Q-learning and Random agents
   on a sample CyberBattle gym environment and plot the respective
   cummulative rewards in the terminal.

Example usage:

    python3.8 -m run --training_episode_count 50  --iteration_count 9000 --rewardplot_with 80  --chain_size=20 --ownership_goal 1.0

"""
import torch
import gym
import logging
import sys
import asciichartpy
import argparse
import cyberbattle._env.cyberphysicalbattle_env as cyberphysicalbattle_env
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.learner as learner
from cyberbattle._env.cp_defender import ScanAndReimageCompromisedMachines
from cyberbattle._env.cyberphysicalbattle_env import DefenderConstraint
import cyberbattle.agents.baseline.agent_randomcredlookup as rca

parser = argparse.ArgumentParser(description='Run simulation with DQL baseline agent.')

parser.add_argument('--training_episode_count', default=50, type=int,
                    help='number of training epochs')

parser.add_argument('--eval_episode_count', default=10, type=int,
                    help='number of evaluation epochs')
# default was previously 9000
parser.add_argument('--iteration_count', default=90, type=int,
                    help='number of simulation iterations for each epoch')

parser.add_argument('--reward_goal', default=2180, type=int,
                    help='minimum target rewards to reach for the attacker to reach its goal')

parser.add_argument('--ownership_goal', default=1.0, type=float,
                    help='percentage of network nodes to own for the attacker to reach its goal')

parser.add_argument('--rewardplot_with', default=80, type=int,
                    help='width of the reward plot (values are averaged across iterations to fit in the desired width)')

parser.add_argument('--chain_size', default=4, type=int,
                    help='size of the chain of the CyberBattleChain sample environment')
parser.add_argument('--with_defender',default= True, type=bool, help='Indicates, if Defender is considered in the environment')

parser.add_argument('--random_agent', dest='run_random_agent', action='store_true', help='run the random agent as a baseline for comparison')
parser.add_argument('--no-random_agent', dest='run_random_agent', action='store_false', help='do not run the random agent as a baseline for comparison')
parser.set_defaults(run_random_agent=True)

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

print(f"torch cuda available={torch.cuda.is_available()}")

cyberphysicalbattle = gym.make('CyberPhysicalBattle-v0',
                            attacker_goal=cyberphysicalbattle_env.AttackerGoal(
                                own_atleast_percent=args.ownership_goal,
                                reward=args.reward_goal))

if args.with_defender:
    cyberphysicalbattle = gym.make('CyberPhysicalBattle-v0',
                                attacker_goal=cyberphysicalbattle_env.AttackerGoal(
                                    own_atleast_percent=args.ownership_goal,
                                    reward=args.reward_goal),
                                     defender_constraint=DefenderConstraint(
                                         maintain_sla=0.80
                                     ),
                                     defender_agent=ScanAndReimageCompromisedMachines(
                                         probability=0.6,
                                         scan_capacity=2,
                                         scan_frequency=5)
                                   )

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=22,
    maximum_node_count=22,
    identifiers=cyberphysicalbattle.identifiers
)

all_runs = []

cyberphysicalbattle.seed(34)


if args.with_defender:
    # %%
    dqn_with_defender = learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberphysicalbattle,
        environment_properties=ep,
        learner=dqla.DeepQLearnerPolicy(
            ep=ep,
            gamma=0.15,
            replay_memory_size=10000,
            target_update=5,
            batch_size=256,
            learning_rate=0.01),
        episode_count=args.training_episode_count,
        iteration_count=args.iteration_count,
        epsilon=0.90,
        render=False,
        epsilon_exponential_decay=5000,
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="DQL"
    )

    # %%
    dql_exploit_run = learner.epsilon_greedy_search(
        cyberphysicalbattle,
        ep,
        learner=dqn_with_defender['learner'],
        episode_count=args.training_episode_count,
        iteration_count=args.iteration_count,
        epsilon=0.0,  # 0.35,
        render=False,
        # render_last_episode_rewards_to='images/chain10',
        verbosity=Verbosity.Quiet,
        title="Exploiting DQL"
    )

    # %%
    credlookup_run = learner.epsilon_greedy_search(
        cyberphysicalbattle,
        ep,
        learner=rca.CredentialCacheExploiter(),
        episode_count=10,
        iteration_count=args.iteration_count,
        epsilon=0.90,
        render=False,
        epsilon_exponential_decay=10000,
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="Credential lookups (ϵ-greedy)"
    )

    # %%
    # Plots
    all_runs = [
        credlookup_run,
        dqn_with_defender,
        dql_exploit_run
    ]
    p.plot_averaged_cummulative_rewards(
        all_runs=all_runs,
        title=f'Attacker agents vs Basic Defender -- rewards\n env={cyberphysicalbattle.name}, episodes={args.training_episode_count}'
    )

    # p.plot_episodes_length(all_runs)
    p.plot_averaged_availability(
        title=f"Attacker agents vs Basic Defender -- availability\n env={cyberphysicalbattle.name}, episodes={args.training_episode_count}",
        all_runs=all_runs)

else:

    # Run Deep Q-learning
    dqn_learning_run = learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberphysicalbattle,
        environment_properties=ep,
        learner=dqla.DeepQLearnerPolicy(
            ep=ep,
            gamma=0.015,
            replay_memory_size=10000,
            target_update=10,
            batch_size=512,
            learning_rate=0.01),  # torch default is 1e-2
        episode_count=args.training_episode_count,
        iteration_count=args.iteration_count,
        epsilon=0.90,
        render=False,
        # epsilon_multdecay=0.75,  # 0.999,
        epsilon_exponential_decay=5000,  # 10000
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="DQL"
    )

    all_runs.append(dqn_learning_run)

    if args.run_random_agent:
        random_run = learner.epsilon_greedy_search(
            cyberphysicalbattle,
            ep,
            learner=learner.RandomPolicy(),
            episode_count=args.eval_episode_count,
            iteration_count=args.iteration_count,
            epsilon=1.0,  # purely random
            render=False,
            verbosity=Verbosity.Quiet,
            title="Random search"
        )
        all_runs.append(random_run)

    colors = [asciichartpy.red, asciichartpy.green, asciichartpy.yellow, asciichartpy.blue]

    print("Episode duration -- DQN=Red, Random=Green")
    print(asciichartpy.plot(p.episodes_lengths_for_all_runs(all_runs), {'height': 30, 'colors': colors}))

    print("Cumulative rewards -- DQN=Red, Random=Green")
    c = p.averaged_cummulative_rewards(all_runs, args.rewardplot_with)
    print(asciichartpy.plot(c, {'height': 10, 'colors': colors}))

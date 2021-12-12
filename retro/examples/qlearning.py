"""
Implementation of the Brute from "Revisiting the Arcade Learning Environment:
Evaluation Protocols and Open Problems for General Agents" by Machado et al.
https://arxiv.org/abs/1709.06009

This is an agent that uses the determinism of the environment in order to do
pretty well at a number of retro games.  It does not save emulator state but
does rely on the same sequence of actions producing the same result when played
back.
"""

import random
import argparse
import gym.spaces
import numpy as np
import retro
import gym


EXPLORATION_PARAM = 0.005



class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

def evaluate_q(env, q_table, best_reward):
    state = env.reset()
    done = False
    total_reward = 0
    acts = []

   
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        total_reward += reward
        acts.append(action)

    if total_reward > best_reward:
            print("new best reward {} => {}".format(best_reward, total_reward))
            best_reward = total_reward
            env.unwrapped.record_movie("best.bk2")
            env.reset()
            for act in acts:
                env.step(act)
            env.unwrapped.stop_record()

def q_learning(
    game,
    max_episode_steps=4500,
    timestep_limit=1e8,
    state=retro.State.DEFAULT,
    scenario=None,
):
    env = retro.make(game, state, use_restricted_actions=retro.Actions.DISCRETE, scenario=scenario)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    alpha = 0.1
    gamma = 0.5
    epsilon = 0.1
    best_reward = float('-inf')
    env.reset()
    next_state, reward, done, info = env.step(env.action_space.sample()) 
    print(env.action_space.n)
    q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])
    
    for i in range(1, 100001):
        env.reset()
    
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action) 

        old_q = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_q = old_q + alpha * (reward + (gamma * next_max) - old_q)
        q_table[state, action] = new_q

    if i%100 == 0:
        evaluate_q(env, q_table, best_reward)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Airstriker-Genesis')
    parser.add_argument('--state', default=retro.State.DEFAULT)
    parser.add_argument('--scenario', default=None)
    args = parser.parse_args()

    q_learning(game=args.game, state=args.state, scenario=args.scenario)
    

if __name__ == "__main__":
    main()

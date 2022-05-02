import sys
import random
from numpy.core.fromnumeric import argmax, cumprod
from environment import MountainCar
import numpy as np


def main(args):
    global epsilon, mode, gamma, learning_rate, actions, states
    global max_iterations, episodes, weights_out, returns_out
    mode = args[1]
    weights_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])

def select_mode(mode):
    return mc.state_space, mc.action_space 

def q_learning(mc, states, actions):
    theta = np.zeros([states,actions])
    bias = 0
    list_of_rewards = []
    for i in range(episodes):
        state_dict = mc.reset()
        total_reward = 0
        for j in range(max_iterations):
            if random.random() < epsilon:
                action = np.random.randint(0,2)
            else:
                current_q= calculate_q_vector(state_dict, theta, bias)
                action = np.argmax(current_q)
            next_state_dict, reward, done = mc.step(action)
            next_q = calculate_q_vector(next_state_dict, theta, bias)
            optimal_q = learning_rate*(current_q[action] - (reward + (gamma*np.max(next_q))))
            for index, state in state_dict.items():
                theta[index,action] -= (optimal_q*state)
            bias -= optimal_q
            total_reward += reward
            state_dict = next_state_dict
            if(done == True):
                break
        list_of_rewards.append(total_reward)
    return bias, theta, list_of_rewards

def write_returns_file(filename, file_contents):
        with open(filename, 'w') as filehandle:
            filehandle.writelines("{0:.1f}\n".format(theta) for theta in file_contents)

def write_weights_file(filename, file_contents):
        with open(filename, 'w') as filehandle:
            for theta in file_contents:
                if theta != 0.0:
                    filehandle.writelines("{0:.16f}\n".format(theta))
                else:
                    filehandle.writelines("{0:.1f}\n".format(theta))

def calculate_q_vector(state_dict, theta, bias):
    q = np.zeros([actions])
    for index, state in state_dict.items():
        q += theta[index, :]*state
    q += bias
    return q


if __name__ == "__main__":
    main(sys.argv)
    mc = MountainCar(mode)
    states, actions = select_mode(mc)
    bias, theta, returns = q_learning(mc, states, actions)
    theta_list = []
    theta_list.append(bias)
    returns_list = list(returns)
    for row in range(theta.shape[0]):
        for column in range(theta.shape[1]):
            theta_list.append(theta[row,column])
    write_weights_file(weights_out, theta_list)
    write_returns_file(returns_out, returns_list)



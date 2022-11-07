import time
from collections import deque, namedtuple # using deque data structure for memory buffer
import utils

import gym
import numpy as np


# Hyperparameters
MEMORY_SIZE = 100_000     # size of memory buffer.
GAMMA = 0.995             # discount factor.
ALPHA = 1e-3              # learning rate.
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps.



if __name__ == "__main__":
    environment_name = 'LunarLander-v2'
    env = gym.make(environment_name)

    optimizer = utils.get_optimizer(learning_rate=ALPHA)
    q_network, target_q_network = utils.create_q_and_target_networks(env)

    # Store experiences as named tuples.
    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    # START LEARNING ALGORITHM
    start = time.time() 

    num_episodes = 2000
    max_num_timesteps = 1000

    total_point_history = []

    num_p_av = 100    # number of total points to use for averaging.
    epsilon = 1.0     # initial ε value for ε-greedy policy.

    # Create a memory buffer D with capacity N.
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights.
    target_q_network.set_weights(q_network.get_weights())

    for episode in range(1, num_episodes+1):
        
        # Reset the environment to the initial state and get the initial state.
        state = env.reset()
        total_points = 0
        
        for t in range(max_num_timesteps):
            
            # From the current state S choose an action A using an ε-greedy policy.
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network.
            q_values = q_network(state_qn)
            action = utils.get_action(q_values, epsilon)
            
            # Take action A and receive reward R and the next state S'.
            next_state, reward, done, _ = env.step(action)
            
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))
            
            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
            
            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D.
                experiences = utils.get_experiences(memory_buffer)
                
                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                utils.agent_learn(experiences, GAMMA, optimizer, q_network, target_q_network)
            
            state = next_state.copy()
            total_points += reward
            
            if done:
                break
                
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        
        # Update the ε value.
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {episode} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

        if episode % num_p_av == 0:
            print(f"\rEpisode {episode} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        # We will consider that the environment is solved if we get an
        # average of 200 points in the last num_p_av episodes (num_p_av=100 in this case).
        if av_latest_points >= 200.0:
            print(f"\n\nEnvironment solved in {episode} episodes!")
            print('Saving model... ', end='')
            q_network.save('./models/lunar_lander_model.h5')
            print('Done!')
            break
            
    tot_time = time.time() - start # end of learning algorithm

    print(f"\nTotal Training Time: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

    # Plot the total point history along with the moving average.
    utils.save_plot_history(total_point_history, filepath='./images/moving_average.png')

    # Create a video of our agent interacting with the environment using the trained Q-Network
    filename = "./videos/lunar_lander.mp4"
    utils.create_video(filename, env, q_network)
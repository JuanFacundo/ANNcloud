#from collection import defaultdict          #allows access to keys do not access?

import matplotlib.pyplot as plt             #drawing plots
import matplotlib.patches                   #draw shaped
import numpy as np
import seaborn as sns

from collections import defaultdict

from tqdm import tqdm                       #progress bars
import gymnasium as gym



env = gym.make('Blackjack-v1',sab=True,render_mode="rgb_array")

#reset the environment to get the first observation
done=False
observation,info = env.reset()

#observation = (16,9,False)

#Note that our observation is a tuple consisting of 3 values:
    #1. The players current sum
    #2. Values of the dealers face-up card
    #3. Boolean whether the player holds a usable ace (is is usable if is counts as 11 without busting)


#Sample a random action from all valid actions
action = env.action_space.sample()
#action 1

#execute the action in our environment and receive info after taking the step
observation,reward,terminated,truncated,info = env.step(action)
#observation=(24,10,False)
#reward=-1.0
#terminated=True
#truncated=False
#info={}

class BlackjackAgent:
    def __init__(
        self,
        learning_rate:float,
        initial_epsilon:float,
        epsilon_decay:float,
        final_epsilon:float,
        discount_factor:float = 0.95
    ):
        """Initialize a RL agent with empty dictionary of state-action value (q_values), a learning rate and an epsilon.
        """
        self.q_values = defaultdict(lambda:np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_facor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, obs: tuple[int,int,bool]) -> int:
        """
        Returns the best action with probability (1-epsilon) otherwise
        a random action with probability epsilon to ensure exploration
        """

        #with probability epsilon return a random action to explotr the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        
        #with probability (1-epsilon) act greedy-ly (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
        self,
        obs:tuple[int,int,bool],
        action:int,
        reward:float,
        terminated:bool,
        next_obs: tuple[int,int,bool]
    ):
        """
            Updates the Q-value of an action
        """

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_facor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,self.epsilon-epsilon_decay)
    

#hyperparameters
learning_rate=0.01
n_episodes = 100_000    ###################
start_epsilon=1.0
epsilon_decay = start_epsilon / (n_episodes/2) #reduce the explorarion over time
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate = learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
)

from IPython.display import clear_output
env = gym.wrappers.RecordEpisodeStatistics(env,deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    clear_output()

    #play one episode:
    while not done:
        action = agent.get_action(obs)
        next_obs,reward,terminated,truncated,info = env.step(action)

        #
        agent.update(obs,action,reward,terminated,next_obs)
        #frame=env.render()
        #plt.imshow(frame)
        #plt.show()

        #
        done = terminated or truncated
        obs = next_obs
    
    agent.decay_epsilon()

rolling_length = 500
fig , axs = plt.subplots(ncols=3,figsize=(12,5))
axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(),np.ones(rolling_length),mode="valid"
    )/rolling_length
)
axs[0].plot(range(len(reward_moving_average)),reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(),np.ones(rolling_length),mode="same"
    )/rolling_length
)
axs[1].plot(range(len(length_moving_average)),length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(
        np.array(agent.training_error),np.ones(rolling_length), mode="same"
    )/rolling_length
)
axs[2].plot(range(len(training_error_moving_average)),training_error_moving_average)
plt.tight_layout()
plt.show()

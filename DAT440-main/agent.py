import random
import numpy as np


class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, state_space, action_space, learner = "q-learning", initialization = "zero"):
        """
        learner could be: "q-learning", "double-q-learning", "sarsa", "expected-sarsa"

        """
        
        
        self.action_space = action_space
        self.state_space = state_space

        self.gamma = 0.95
        self.epsilon = 0.05
        self.alpha = 0.2

        c = 0.3 # The q-value used in heuristic initilization, 
                # and scalar for random initilization (so could function as optimistic initialization).
        
        self.algorithm = learner # q-learning, double-q-learning, sarsa or expected-sarsa

        match self.algorithm:
            case "q-learning":
                self.q_table = self.initializeQtable(initialization, c)
            case "double-q-learning":
                self.q1_table = self.initializeQtable(initialization, c)
                self.q2_table = self.initializeQtable(initialization, c)
            case "sarsa":
                self.q_table = self.initializeQtable(initialization, c)
            case "expected-sarsa":
                self.q_table = self.initializeQtable(initialization, c)
            case _:
                print("WARNING: Invalid algorithm. Please specify learning algorithm to be used used")
                print("For this run, defaulting to q-learning")
                self.algorithm = "q-learning"
                self.q_table = self.initializeQtable(initialization, c)

    def initializeQtable(self, strategy = "zero", c=0.3):
        """
        Initialize strategies include:
        'zero' initialization:      Initialize every state-action pair to 0. Unbiased
        'random' initialization:    Initialize every state-action pair to a random value. Biases 
                                    exploraiton in a random direction which may result in more
                                    exploration.
        'heuristic' initialization: Initializes based on prior knowledge about the environment.
                                    In our case, we set 'right' and 'down' = 1, and 'up', 'left' = 0
                                    as we know that the goal is down and to the right of the start.
                                    Can promote faster learning in expense of exploration.
        """
        match strategy:
            case "zero":
                return np.zeros((self.state_space, self.action_space))
            case "random":
                return np.random.rand(self.state_space, self.action_space)
            case "heuristic": #[left, down, right, up]
                return np.asarray([[-c, c, c, -c] for state in range(self.state_space)])
            case "optimistic":
                return c*np.zeros((self.state_space, self.action_space))
            case _:
                print("WARNING: Invalid initilization strategy. Please choose between: 'zero', 'random', 'heuristic','optimistic'")
                print("For this run, defaulting to zero initialization")
                return np.zeros((self.state_space, self.action_space))
    def observe(self, observation, reward, done):
        # Add your code here
        if (self.algorithm == "q-learning"):
            delta = reward + self.gamma * \
                np.max(self.q_table[observation, :]) - \
                self.q_table[self.previous_state, self.previous_action]
            self.q_table[self.previous_state, self.previous_action] = self.q_table[self.previous_state,
                                                                                   self.previous_action] + self.alpha * delta

        elif (self.algorithm == "double-q-learning"):
            if np.random.rand() < 0.5:
                max_action = self.q1_table[observation, :].tolist().index(
                    np.max(self.q1_table[observation, :]))
                delta = reward + self.gamma * \
                    self.q2_table[observation, max_action] - \
                    self.q1_table[self.previous_state, self.previous_action]
                self.q1_table[self.previous_state, self.previous_action] = self.q1_table[self.previous_state,
                                                                                         self.previous_action] + self.alpha * delta
            else:
                max_action = self.q2_table[observation, :].tolist().index(
                    np.max(self.q2_table[observation, :]))
                delta = reward + self.gamma * \
                    self.q1_table[observation, max_action] - \
                    self.q2_table[self.previous_state, self.previous_action]
                self.q2_table[self.previous_state, self.previous_action] = self.q2_table[self.previous_state,
                                                                                         self.previous_action] + self.alpha * delta

        elif (self.algorithm == "sarsa"):
            next_action = self.q_table[observation, :].tolist().index(
                np.max(self.q_table[observation, :]))
            delta = reward+self.gamma * self.q_table[observation, next_action] - self.q_table[self.previous_state, self.previous_action] 
            self.q_table[self.previous_state, self.previous_action] += self.alpha*delta

        elif (self.algorithm == "expected-sarsa"):
            next_action = self.q_table[observation, :].tolist().index(
                np.max(self.q_table[observation, :]))
            expected_value = np.sum(self.q_table[observation, :] * self.epsilon / self.action_space) + (
                1 - self.epsilon) * np.max(self.q_table[observation, :])
            delta = reward + self.gamma * expected_value - \
                self.q_table[self.previous_state, self.previous_action]
            self.q_table[self.previous_state,
                         self.previous_action] += self.alpha * delta
        else:
            print("did not know what algorithm to use so I learnt nothing")

    def act(self, observation):
        # Add your code here
        if isinstance(observation, tuple):
            observation = observation[0]

        if self.algorithm == "double-q-learning":
            self.q_table = self.q1_table + self.q2_table
            

        if np.random.rand() < self.epsilon or np.all(list(map(lambda x: x == self.q_table[observation, 0], self.q_table[observation, :]))):
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.q_table[observation, :])

        self.previous_action = action
        self.previous_state = observation
        return action

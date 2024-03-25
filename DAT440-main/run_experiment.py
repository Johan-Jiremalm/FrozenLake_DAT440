import argparse
import gymnasium as gym
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import riverswim

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str,
                    help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment",
                    default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)


try:
    env = gym.make(args.env, is_slippery=True, render_mode="rgb_array", map_name="4x4")
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)

def qtable_directions_map(qtable, map_size):  # Taken from gymnasium documentation
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


# Taken from gymnasium documentation
def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(img_title, bbox_inches="tight")
    plt.show()

action_dim = env.action_space.n
state_dim = env.observation_space.n



learners = ["q-learning", "double-q-learning", "sarsa", "expected-sarsa"]
for learner in learners:
    nrOfExperiments = 5
    ys = []
    q_tables = []
    for experiment in range(nrOfExperiments):
        agent = agentfile.Agent(state_dim, action_dim, learner=learner, initialization="random")
        rewards = []
        observation = env.reset()
        step = 0
        while step < 10000:
            #if i > 100000+30:
            #    plt.imshow(env.render())
            #    plt.show()
            action = agent.act(observation) # your agent here (this currently takes random actions)
            observation, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            agent.observe(observation, reward, done)
            
            if done:
                observation, info = env.reset()
            step += 1 

        y = []  # time to reward
        s = 0
        movingWindow = 100 #100 is the max episode length
        for i in range(len(rewards)):
            low = max(0, i-movingWindow)
            s += sum(rewards[low:i+1])/(i-low+1)
            y.append(s/(i+1))
        ys.append(y) #record experiment
        q_tables.append(copy.deepcopy(agent.q_table))

    avgReward = np.asarray([sum([ys[experiment][i] for experiment in range(nrOfExperiments)])/nrOfExperiments\
                for i in range(len(ys[0]))])
    confRadius = np.asarray([1.96*np.std([ys[experiment][i] for experiment in range(nrOfExperiments)])/np.sqrt(nrOfExperiments)\
                for i in range(len(ys[0]))])

    #plot_q_values_map(q_tables[0], env, 4)

    fig, ax = plt.subplots()
    ax.plot(avgReward, label="Reward, moving average over 100 steps")
    ax.fill_between(list(range(len(avgReward))), avgReward-confRadius, avgReward+confRadius, alpha = 0.8, color = "red", label="95% confidence radius")
    ax.legend()
    ax.set_title(learner)
    #plt.show()
    fig.savefig("Results/FrozenLake/"+learner+"Random", bbox_inches="tight")
#print(variance)
env.close()

import sys
from cache.DataLoader import DataLoaderPintos
from cache.Cache import Cache
from agents.CacheAgent import *
from agents.DQNAgent import DQNAgent
from agents.ReflexAgent import *
import numpy as np
import matplotlib.pyplot as plt
import pandas
import tensorflow
import matplotlib.pyplot

if __name__ == "__main__":
    # disk activities
    dataloader = DataLoaderPintos(["data.csv"])
    
    sizes = [1, 5, 25, 50, 100, 300]
    agent_names = ['DQN', 'Random', 'LRU', 'LFU', 'MRU']
    miss_rates_dict = {name: [] for name in agent_names}

    for cache_size in sizes:
        
        print("==================== Cache Size: %d ====================" % cache_size)

        # cache
        env = Cache(dataloader, cache_size
            , feature_selection=('Base',)
            , reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
            , allow_skip=False
        )
        
        # agents
        agents = {}
        agents['LRU'] = DQNAgent(env.n_actions, env.n_features,
            learning_rate=0.01,
            reward_decay=0.9,

            # Epsilon greedy
            e_greedy_min=(0.0, 0.1),
            e_greedy_max=(0.2, 0.8),
            e_greedy_init=(0.1, 0.5),
            e_greedy_increment=(0.005, 0.01),
            e_greedy_decrement=(0.005, 0.001),

            history_size=50,
            dynamic_e_greedy_iter=25,
            reward_threshold=20,
            explore_mentor = 'LRU',

            replace_target_iter=100,
            memory_size=10000,
            batch_size=128,

            output_graph=False,
            verbose=0
        )
        agents['Random'] = RandomAgent(env.n_actions)
        agents['LRU'] = LRUAgent(env.n_actions)
        agents['LFU'] = LFUAgent(env.n_actions)
        agents['MRU'] = MRUAgent(env.n_actions)

        for (name, agent) in agents.items():
            print("-------------------- %s --------------------" % name)
            step = 0
            miss_rates = []
            if isinstance(agent, LearnerAgent):
                episodes = 100
            elif isinstance(agent, RandomAgent):
                episodes = 20
            else:
                episodes = 1
            for episode in range(episodes):
                observation = env.reset()
                while True:
                    action = agent.choose_action(observation)
                    if env.hasDone():
                        break
                    observation_, reward = env.step(action)
                    agent.store_transition(observation, action, reward, observation_)
                    if isinstance(agent, LearnerAgent) and (step > 20) and (step % 5 == 0):
                        agent.learn()
                    observation = observation_
                    if step % 100 == 0:
                        mr = env.miss_rate()
                    step += 1
                mr = env.miss_rate()
                print("Agent=%s, Size=%d, Episode=%d: Accesses=%d, Hits=%d, MissRate=%f"
                    % (name, cache_size, episode, env.total_count, env.miss_count, mr)
                )
                miss_rates.append(mr)

            # summary
            miss_rates = np.array(miss_rates)
            print("Agent=%s, Size=%d: Mean=%f, Median=%f, Max=%f, Min=%f"
                % (name, cache_size, np.mean(miss_rates), np.median(miss_rates), np.max(miss_rates), np.min(miss_rates))
            )
            miss_rates_dict[name].append(np.mean(miss_rates))

    # Plotting
    for name, miss_rates_list in miss_rates_dict.items():
        plt.plot(sizes, miss_rates_list, label=name)

    # Add labels and legend
    plt.xlabel('Cache Capacity')
    plt.ylabel('Mean Miss Rate')
    plt.legend()
    plt.title('Miss Rate vs. Cache Capacity for Different Algorithms')
    plt.show()

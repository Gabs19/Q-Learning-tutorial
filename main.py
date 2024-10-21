import numpy as np
import gym
import random
import imageio

from IPython.display import Image
# from pyvirtualdisplay import Display
#
# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()
def initialize_q_table(start_space, action_space):
    Qtable = np.zeros((start_space, action_space))
    return Qtable

#politica de atuação
def epsilon_greedy_policy(Qtable, state, epsilon):
    random_int = random.uniform(0,1) #gerar um numero aleatorio entre 0 e 1
    if random_int > epsilon:
        action = np.argmax(Qtable[state])
    else:
        action = env.action_space.sample() #tomando decisões aleatorias

    return action

#politica de atualização
#usado para selecionar o valor mais alto no Q-table
#também usado como politica final quando o agente for treinado
def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state])
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in range(n_training_episodes):

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

        #reseta o ambiente
        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            new_state, reward, done, info = env.step(action)

            Qtable[state][action] = Qtable[state][action] = lerning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            #se for done, finaliza o episodio
            if done:
                break

            # nosso estado é o novo estado
            state = new_state

    return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episodes_rewards = []

    for episode in range(n_eval_episodes):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()

        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action = np.argmax(Q[state][:])
            new_state, reward, done, info = env.step(action)

            total_rewards_ep += reward

            if done:
                break
            state = new_state

        episodes_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episodes_rewards)
    std_reward = np.std(episodes_rewards)

    return mean_reward, std_reward

def record_video(env, Qtable, out_directory, fps=1):
  images = []
  done = False
  state = env.reset(seed=random.randint(0,500))
  img = env.render(mode='rgb_array')
  images.append(img)
  while not done:
    # Take the action (index) that have the maximum expected future reward given that state
    action = np.argmax(Qtable[state][:])
    state, reward, done, info = env.step(action) # We directly put next_state = state for recording logic
    img = env.render(mode='rgb_array')
    images.append(img)
  imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=False)

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")

qtable_frozenlake = initialize_q_table(state_space, action_space)
print(qtable_frozenlake)

#parametros de treino
n_trainning_episodes = 10000 #treinamento
lerning_rate = 0.7 #taxa de aprendizado

#parametros de avaliação
n_eval_episodes = 100 #avaliação de episodios

#parametros de ambiente
env_id = "FrozenLake-v1"
max_steps = 99
gamma = 0.95
eval_seed = []

#parametros de explocação
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005

qtable_frozenlake = train(n_trainning_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, qtable_frozenlake)
print(qtable_frozenlake)

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

video_path="/Repositorys/qLearning_tutorial/content/replay.gif"
video_fps=1
record_video(env, qtable_frozenlake, video_path, video_fps)

Image('./replay.gif')
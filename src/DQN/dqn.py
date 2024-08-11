 # import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
from collections import deque
import gdown
import os
import json

# accepts city name and does initial preprocessing
def data_download(city_nm):
  # Download the Data from Google Drive to the temporary folder
  merged_data_file_id = '1o_EEumVnswul9MVsrdDwBch5rt7JTr0m'
  merged_data_url = f'https://drive.google.com/uc?id={merged_data_file_id}'
  merged_data_filepath = '../../temporary_files/merged.csv'
  gdown.download(merged_data_url, merged_data_filepath, quiet=False)
  # import data
  merged = pd.read_csv(merged_data_filepath)
  # grab unique cities
  cities = list(merged.City.unique())
  # check if city is one of viable options
  assert city_nm in cities, f'City is not in {cities}'
  # filter to selected city and sort
  merged = merged[merged.City == city_nm]
  merged.Date = pd.to_datetime(merged.Date)
  merged.sort_values('Date', inplace = True)
  return merged

# create action space for reinforcement learning model
def action_space(merged):
  merged['pct_chng'] = merged.ZHVI.pct_change()
  # it isn't uncommon to see 7% swings in home value (0.583% month to month)
  # so will label anything within 3 - 7% as reasonable increase/decrease
  # anything less than that as relatively unchanged
  # anything more than that as significant increase/decrease
  def conditions(s):
    if s > 0.07/12: return 2
    elif s < -0.07/12: return -2
    elif s >= 0.03/12 and s <= 0.07/12: return 1
    elif s <= -0.03/12 and s >= -0.07/12: return -1
    elif s > -0.03/12 and s < 0.03/12: return 0
  # apply conditions
  merged['change'] = merged.pct_chng.apply(conditions)
  # drop pct_chng and ZHVI so no data leakage
  # merged.drop(['ZHVI', 'pct_chng'], axis = 1, inplace = True)
  return merged

# split on some arbitrary value and scale
def train_test_split(merged, split_pct, window_size):
  # split train and test based on percentage
  split_index = round(int(len(merged) * split_pct))
  train = merged.iloc[:split_index]
  # start test index at the top of the window
  test = merged.iloc[split_index - window_size + 1:]
  # init scaler and apply to numeric columns
  scaler = MinMaxScaler()
  numeric_cols = list(merged.drop(['City', 'Date', 'change'], axis = 1).columns)
  train_X = scaler.fit_transform(train[numeric_cols].astype(float))
  test_X = scaler.fit_transform(test[numeric_cols].astype(float))
  train_y = train.change.values
  test_y = test.change.values
  return train_X, test_X, train_y, test_y

# create LSTM
class QNetwork(torch.nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
    super().__init__()
    self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim,
                        num_layers = num_layers, batch_first = True)
    self.linear = nn.Linear(in_features = hidden_dim, out_features = output_dim)

  def forward(self, state):
    x, _ = self.lstm(state)
    x = x[:, -1, :]
    x = self.linear(x)
    return x

# create time series environment
class TimeSeries:
  def __init__(self, X, y, window_size):
    self.X = X
    self.y = y
    self.window_size = window_size
    self.current_step = 0
    self.data_len = len(self.X)

  def reset(self):
    self.current_step = self.window_size
    return self.X[:self.current_step, :]

  def step(self, action):
    self.current_step += 1
    done = self.current_step >= self.data_len - 1
    next_state = self.X[self.current_step - self.window_size:self.current_step]
    actual = self.y[self.current_step]
    reward = -abs(actual - action)
    return next_state, reward, done

# create agent
class DQNAgent:
  def __init__(self, input_dim, output_dim, hidden_dim, window_size, lr, gamma, eps, 
               eps_decay, min_eps, memory_size, batch_size, num_layers = 1, seed = None,
               QNetwork = QNetwork):
    # use class inheritance to modify QNetwork when needed
    self.dqn = QNetwork(input_dim, output_dim, hidden_dim, num_layers)
    self.dqn_target = QNetwork(input_dim, output_dim, hidden_dim, num_layers)
    self.dqn_target.load_state_dict(self.dqn.state_dict())
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.window_size = window_size
    # MSE loss function
    self.loss_fn = nn.MSELoss()
    # Adam optimization
    self.optim = optim.Adam(self.dqn.parameters(), lr = lr)
    self.gamma = gamma
    self.epsilon = eps
    self.epsilon_decay = eps_decay
    self.min_epsilon = min_eps
    self.batch_size = batch_size
    self.replay_memory_buffer = deque(maxlen = memory_size)
    if seed is None:
        self.rng = np.random.default_rng()
    else:
        self.rng = np.random.default_rng(seed)

  def select_action(self, state):
    # epsilon greedy action selection
    if self.rng.uniform() < self.epsilon:
      action = self.rng.choice(self.output_dim)
    else:
      state = torch.from_numpy(state).float().unsqueeze(0)
      self.dqn.eval()
      with torch.no_grad():
          q_values = self.dqn(state)
      self.dqn.train()
      action = torch.argmax(q_values).item()
    return action

  def train(self, s0, a0, r, s1, done):
    self.add_to_replay_memory(s0, a0, r, s1, done)

    if done:
      self.update_epsilon()
      self.target_update()

    if len(self.replay_memory_buffer) < self.batch_size:
      return

    mini_batch = self.get_random_sample_from_replay_mem()
    state_batch = torch.from_numpy(np.stack([i[0] for i in mini_batch])).float()
    action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).int() #reshape(1, self.batch_size, 1)
    reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float()
    next_state_batch = torch.from_numpy(np.stack([i[3] for i in mini_batch])).float()
    done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float()
    
    current_qs = self.dqn(state_batch)
    current_q  = current_qs.gather(1, action_batch.type(torch.int64))
    next_q, _  = self.dqn_target(next_state_batch).max(dim = 1)
    next_q     = next_q.view(self.batch_size, 1)
    Q_targets  = reward_batch + self.gamma * next_q * (1 - done_list)
    loss       = self.loss_fn(current_q, Q_targets.detach())
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

  def add_to_replay_memory(self, state, action, reward, next_state, done):
    self.replay_memory_buffer.append((state, action, reward, next_state, done))

  def get_random_sample_from_replay_mem(self):
    random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
    return random_sample

  def update_epsilon(self):
    if self.epsilon > self.min_epsilon:
      self.epsilon *= self.epsilon_decay
      self.epsilon = max(self.min_epsilon, self.epsilon)

  def target_update(self):
    self.dqn_target.load_state_dict(self.dqn.state_dict())

# define training looper
def episode_loop(X, y, max_reward = 0, maxlen = 100, window_size = 7, seed = 0, num_layers = 1,
                 hidden_dim = 24, lr = 0.001, gamma = 0.99, eps = 1, eps_decay = 0.995, 
                 min_eps = 0.01, memory_size = 36, batch_size = 12, num_episodes = 500, q = QNetwork):
  reward_queue = deque(maxlen = maxlen)
  all_rewards = []
  all_rewards_each_step = []
  env = TimeSeries(X, y, window_size)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  input_dim = X.shape[1]
  output_dim = len(np.unique(y[~np.isnan(y)]))
  agent = DQNAgent(input_dim, output_dim, hidden_dim, window_size, lr, gamma, eps, 
                   eps_decay, min_eps, memory_size, batch_size, num_layers, seed)
  # iterate through episodes and train
  for i in range(num_episodes):
    state = env.reset()
    done = False
    episodic_reward = 0
    episode_rewards = []
    while not done:
      action = agent.select_action(np.squeeze(state))
      next_state, reward, done = env.step(action)
      episode_rewards.append(reward)
      episodic_reward += reward
      agent.train(state, action, reward, next_state, done)
      state = next_state
    all_rewards.append(episodic_reward)
    all_rewards_each_step.append(episode_rewards)
    reward_queue.append(episodic_reward)
    if (i + 1) % 10 == 0 and len(reward_queue) == 100 and (i + 1) % 10 == 0:
      print(f'Training episode {i + 1}, reward: {episodic_reward}', end='')
    elif (i + 1) % 10 == 0: 
      print(f'Training episode {i + 1}, reward: {episodic_reward}')
    if len(reward_queue) == 100:
      avg_reward = sum(reward_queue) / 100
      if (i + 1) % 10 == 0:
          print(f', moving average reward: {avg_reward}')
  # return variables for viz
  return all_rewards, all_rewards_each_step, agent, window_size

# save artifacts using naming conventions
def artifact_save(prefix, city_nm, num_layers, dqn, all_rewards, 
                  all_rewards_each_step, window_size):
  name = f'{prefix}_{city_nm}_{num_layers}_layer_{window_size}_windows'
  torch.save(dqn, f'models/{name}.pth')
  np.save(f'rewards/averaged/{name}.npy', np.array(all_rewards))
  np.save(f'rewards/episodic/{name}.npy', np.array(all_rewards_each_step))

# create test action selection function that does NOT train
# create select_action function for testing
def select_action_test(model, state):
  state = torch.from_numpy(np.squeeze(state)).float().unsqueeze(0)
  model.eval()
  with torch.no_grad():
      q_values = model(state)
  action = torch.argmax(q_values).item()
  return action

# create test loop that uses test select action function
def test_loop(test_X, test_y, loaded_model, window_size = 7):
  env_test = TimeSeries(test_X, test_y, window_size = 7)
  state = env_test.reset()
  done = False
  total_reward = 0
  # show reward at each step using i
  i = 1
  rewards = []
  # compute total reward
  while not done:
      action = select_action_test(loaded_model, state)
      next_state, reward, done = env_test.step(action)
      total_reward += reward
      state = next_state
      print(f'Reward as step {i}: {reward}')
      i += 1
      rewards.append(reward)
  print(f"Total reward on new data: {total_reward}")
  return rewards, total_reward, window_size

# create function that dumps test results to json
def save_test_results(prefix, city_nm, num_layers, rewards,
                      total_reward, window_size):
  # create new json load
  file_loc = 'rewards/test_results.json'
  key_nm = f'{prefix}_{city_nm}_{num_layers}_layer_{window_size}_windows'
  d = {'total_reward' : total_reward,
       'reward_lst'   : rewards,
       'window_size'  : window_size}
  # if file does not exist, create it
  if os.path.isfile(file_loc) == False:
    with open(file_loc, 'w') as f:
      f.write(json.dumps({key_nm : d}))
  # else read and overwrite it
  else:
    with open(file_loc, 'r') as f:
      cur_d = json.load(f)
    cur_d[key_nm] = d
    print(cur_d)
    with open(file_loc, 'w') as f:
      f.write(json.dumps(cur_d))
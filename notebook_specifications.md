
# Trading Game Simulation with Reinforcement Learning: A Quantitative Trader's Journey

## Introduction to the Case Study

As a **CFA Charterholder and Quantitative Trader/Portfolio Manager** at a leading asset management firm, you are constantly seeking innovative ways to enhance investment strategies and manage risk. Your firm is keenly interested in exploring the cutting edge of algorithmic decision-making, particularly **Reinforcement Learning (RL)**, for its unique ability to handle sequential decisions in dynamic environments. Unlike traditional supervised or unsupervised learning, RL agents learn through direct interaction with an environment, making it a compelling paradigm for financial markets where actions have cumulative consequences.

This notebook takes you through a foundational journey into RL for single-stock trading. You will build a simulated trading environment, implement both a basic Q-learning agent and a sophisticated Deep Q-Network (DQN) agent, and rigorously evaluate their performance on historical data. The goal is not to find a perfect trading algorithm (RL in finance comes with significant practical limitations, which we will also discuss), but to provide hands-on experience, demystify core RL concepts, and foster a realistic understanding of its potential and challenges in the financial domain. This experience will equip you to critically assess and potentially integrate adaptive trading strategies into your firm's workflow.

## 1. Setup: Installing Libraries and Importing Dependencies

Before we dive into building our trading agent, we need to ensure all necessary Python libraries are installed and imported. These libraries will provide tools for data handling, environment creation, building neural networks, and plotting results.

```python
!pip install yfinance pandas numpy matplotlib gymnasium tensorflow keras collections-deque
```

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
```

## 2. Data Acquisition and Feature Engineering: Building the Market Observation Space

### Story + Context + Real-World Relevance

As a quantitative trader, your first step in developing any algorithmic strategy is to gather and prepare relevant market data. For a reinforcement learning agent, this data forms the "state" or "observation" of the environment, which it uses to make decisions. Raw price data alone is often insufficient; an agent needs meaningful features that capture market dynamics. You'll download historical data for the S&P 500 ETF (SPY) and engineer technical indicators that are commonly used in financial analysis, such as returns, moving average ratios, Relative Strength Index (RSI), and volatility. These indicators will provide the agent with a richer understanding of market conditions, moving beyond simple price movements.

The market observation space, denoted $S$, for our RL agent will be a vector of these engineered features, combined with the agent's current position.

```python
def compute_rsi(data, window=14):
    """
    Compute the Relative Strength Index (RSI).
    Args:
        data (pd.Series): Price data.
        window (int): The period for RSI calculation.
    Returns:
        pd.Series: RSI values.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_market_data(ticker='SPY', start_date='2010-01-01', end_date='2024-12-31'):
    """
    Downloads historical price data and computes technical indicators.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data download.
        end_date (str): End date for data download.
    Returns:
        pd.DataFrame: DataFrame with prices and technical features.
    """
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = pd.DataFrame(index=data.index)
    df['price'] = data['Adj Close']

    # Calculate 1-day and 5-day returns
    df['return_1d'] = df['price'].pct_change()
    df['return_5d'] = df['price'].pct_change(periods=5)

    # Calculate SMA Ratio (10-day / 50-day SMA)
    df['sma_10'] = df['price'].rolling(window=10).mean()
    df['sma_50'] = df['price'].rolling(window=50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50']

    # Calculate RSI (14-period)
    df['rsi'] = compute_rsi(df['price'], window=14)

    # Calculate Volatility (21-day rolling standard deviation of 1-day returns, annualized)
    df['volatility'] = df['return_1d'].rolling(window=21).std() * np.sqrt(252)

    df.dropna(inplace=True)
    
    # Standardize features (important for NN inputs)
    feature_columns = ['return_1d', 'return_5d', 'sma_ratio', 'rsi', 'volatility']
    df[feature_columns] = (df[feature_columns] - df[feature_columns].mean()) / df[feature_columns].std()

    print(f"Total trading days after feature engineering: {len(df)}")
    print(f"Data available from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    return df

# Download and prepare data
df_market = prepare_market_data()

# Split into training and testing periods
train_data = df_market.loc['2010-01-01':'2019-12-31']
test_data = df_market.loc['2020-01-01':'2024-12-31'] # End date will be limited by yfinance actual data

train_prices = train_data['price'].values
train_features = train_data[['return_1d', 'return_5d', 'sma_ratio', 'rsi', 'volatility']].values

test_prices = test_data['price'].values
test_features = test_data[['return_1d', 'return_5d', 'sma_ratio', 'rsi', 'volatility']].values

print(f"\nTraining data: {len(train_data)} days ({train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')})")
print(f"Testing data: {len(test_data)} days ({test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')})")
```

### Explanation of Execution

The code successfully downloaded SPY data and computed five key technical indicators: 1-day returns, 5-day returns, SMA ratio, RSI, and volatility. These features were then standardized, which is crucial for neural network-based RL agents to ensure stable training. The dataset was split into training (2010-2019) and testing (2020-2024) periods. This setup is fundamental for evaluating how well the agent learns to generalize its trading strategy to unseen market conditions, mirroring real-world backtesting practices. As a quantitative trader, ensuring robust out-of-sample performance is paramount.

## 3. Formalizing the Trading Problem as a Markov Decision Process (MDP) and Building a Custom Environment

### Story + Context + Real-World Relevance

To apply Reinforcement Learning, you must formalize the trading problem as a **Markov Decision Process (MDP)**. This involves defining the states, actions, rewards, and transitions of your trading "game." This structured approach allows the RL agent to understand its operating environment and the consequences of its decisions. You will build a custom `TradingEnv` using the `gymnasium` library, which is the standard interface for RL environments in Python. This environment will simulate the stock market, track your portfolio, apply transaction costs, and provide feedback (rewards) to the agent based on its actions.

The MDP formulation guides the agent's learning process:
-   **State ($S_t$)**: What the agent observes about the market and its own position at time $t$.
-   **Action ($A_t$)**: The decision the agent makes (Buy, Sell, Hold).
-   **Reward ($R_{t+1}$)**: The immediate feedback the agent receives after taking an action (e.g., portfolio return).
-   **Transition ($P(S_{t+1} | S_t, A_t)$)**: How the environment changes to a new state $S_{t+1}$ given the current state $S_t$ and action $A_t$.
-   **Discount Factor ($\gamma$)**: How much the agent values future rewards compared to immediate ones.

The agent's objective is to find an optimal policy $\pi^*(s)$ that maximizes the expected cumulative discounted reward:
$$
\pi^* = \text{arg} \max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{T} \gamma^t R_{t+1} \right]
$$
Where $T$ is the episode horizon and $\gamma \in [0, 1)$ is the discount factor.

```python
class TradingEnv(gym.Env):
    """
    Custom Gymnasium-compatible environment for single-stock trading.
    State: [position (0=flat, 1=long), return_1d, return_5d, sma_ratio, rsi, volatility]
    Action: 0=Hold, 1=Buy (go long), 2=Sell (go flat if long)
    Reward: Percentage portfolio return for the step, adjusted for transaction costs.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, prices, features, initial_cash=10000, transaction_cost=0.001):
        super().__init__()
        self.prices = prices # Daily adjusted close prices
        self.features = features # Technical indicators for state space
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.n_steps = len(prices) - 1 # Number of days to trade

        # Define action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3) 

        # Define observation space: [position, feature_1, ..., feature_n]
        # Position is 0 (flat) or 1 (long). Features are standardized (can be <0 or >0).
        low_obs = np.array([0] + [-np.inf] * self.features.shape[1], dtype=np.float32)
        high_obs = np.array([1] + [np.inf] * self.features.shape[1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Internal state variables
        self.current_step = None
        self.cash = None
        self.shares = None
        self.position = None # 0 = flat, 1 = long

    def _get_obs(self):
        """Returns the current observation vector."""
        # Current position (0 or 1) + market features for the current day
        return np.concatenate([[self.position], self.features[self.current_step]]).astype(np.float32)

    def _portfolio_value(self):
        """Calculates current total portfolio value (cash + shares value)."""
        return self.cash + self.shares * self.prices[self.current_step]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.position = 0 # Start flat

        initial_obs = self._get_obs()
        info = {'portfolio_value': self._portfolio_value(), 'position': self.position}
        return initial_obs, info

    def step(self, action):
        prev_value = self._portfolio_value()
        current_price = self.prices[self.current_step]

        # Execute action
        if action == 1: # Buy (only if currently flat)
            if self.position == 0:
                # Buy as many shares as possible with available cash, accounting for transaction cost
                shares_to_buy = int((self.cash * (1 - self.transaction_cost)) / current_price)
                self.shares += shares_to_buy
                self.cash -= shares_to_buy * current_price
                self.position = 1 # Now long
        elif action == 2: # Sell (only if currently long)
            if self.position == 1:
                # Sell all shares
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.shares = 0
                self.position = 0 # Now flat
        # If action is 0 (Hold), or invalid Buy/Sell, do nothing.

        self.current_step += 1
        done = self.current_step >= self.n_steps # Episode ends when data runs out

        # Calculate reward based on portfolio value change
        new_value = self._portfolio_value()
        reward = (new_value - prev_value) / prev_value # Percentage return

        obs = self._get_obs()
        info = {'portfolio_value': new_value, 'position': self.position}
        return obs, reward, done, False, info # last False is 'truncated'

# Create the training environment
train_env = TradingEnv(train_prices, train_features)
print(f"Observation space shape: {train_env.observation_space.shape}")
print(f"Action space size: {train_env.action_space.n}")
```

### Explanation of Execution

You have successfully defined a custom `TradingEnv` class that encapsulates the trading rules and market dynamics. The `reset` method initializes the environment to a starting state (flat position, initial cash), while the `step` method processes an agent's action, updates the portfolio, calculates the reward, and advances the market by one day. Transaction costs are factored into every buy and sell, which is crucial for realistic financial simulations. The observation space now correctly combines the agent's current position (a binary indicator of being long or flat) with the pre-processed technical market features, providing a complete picture for decision-making. The action space is discrete: Hold, Buy, or Sell. This MDP formulation is the bedrock for any RL algorithm to learn a trading policy.

## 4. Tabular Q-Learning: A Foundational Approach to Learning

### Story + Context + Real-World Relevance

Before diving into complex deep learning models, it's often beneficial to understand the core mechanics with a simpler, more interpretable algorithm. **Tabular Q-learning** is a fundamental RL algorithm that learns an action-value function, $Q(s, a)$, which estimates the expected cumulative discounted reward for taking action $a$ in state $s$ and then following an optimal policy thereafter. For continuous state spaces like ours, Q-learning requires **discretization** to create a finite number of states. This allows us to store Q-values in a table.

The Q-learning update rule, derived from the **Bellman equation**, iteratively refines these Q-values based on observed rewards and future expected rewards:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]
$$

Here, $\alpha$ is the learning rate, $R_{t+1}$ is the immediate reward, and $\gamma$ is the discount factor. The term in brackets is the **Temporal Difference (TD) error**, representing the difference between the observed reward-plus-estimated-future-value and the current Q-estimate.

To balance trying new actions (exploration) and taking actions known to yield high rewards (exploitation), we use an **epsilon-greedy strategy**:

$$
A_t = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \text{arg} \max_{a'} Q(S_t, a') & \text{with probability } 1 - \epsilon \end{cases}
$$

In finance, exploration is akin to a new portfolio manager trying various strategies to see what works, while exploitation is sticking to strategies that have proven successful. Initially, $\epsilon$ is high (more exploration), and it gradually decays (more exploitation) as the agent learns.

```python
def discretize_state(obs, n_bins=10):
    """
    Discretizes a continuous observation into a tuple of bin indices.
    The first element of obs (position) is discrete (0 or 1).
    Other features are assumed to be standardized (mean 0, std 1).
    """
    position = int(obs[0])
    features = obs[1:] # Market features

    # Define bins for standardized features (e.g., -3 std dev to +3 std dev)
    # n_bins-1 boundaries divide into n_bins regions
    bins = np.linspace(-3, 3, n_bins - 1) 
    
    discrete_features = np.digitize(features, bins)
    return tuple([position] + discrete_features.tolist())

# Q-table: maps (discrete_state_tuple) -> [Q_hold, Q_buy, Q_sell]
q_table = {}

# Hyperparameters for Q-learning
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
epsilon = 1.0       # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995 # Epsilon decay factor per episode
n_episodes_q_learning = 500 # Number of training episodes

rewards_history_q_learning = []

print("Starting Tabular Q-Learning Training...")
for episode in range(n_episodes_q_learning):
    obs, _ = train_env.reset()
    state = discretize_state(obs, n_bins=10)
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = train_env.action_space.sample() # Explore
        else:
            if state not in q_table:
                q_table[state] = np.zeros(train_env.action_space.n)
            action = np.argmax(q_table[state]) # Exploit

        next_obs, reward, done, _, info = train_env.step(action)
        next_state = discretize_state(next_obs, n_bins=10)
        total_reward += reward

        # Ensure next_state exists in Q-table to get its max Q-value
        if next_state not in q_table:
            q_table[next_state] = np.zeros(train_env.action_space.n)

        # Q-learning update (Bellman equation)
        best_next_q = np.max(q_table[next_state])
        
        # Ensure current state exists in Q-table
        if state not in q_table:
            q_table[state] = np.zeros(train_env.action_space.n)
            
        q_table[state][action] += alpha * (reward + gamma * best_next_q - q_table[state][action])

        state = next_state

    rewards_history_q_learning.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 100 == 0:
        avg_reward = np.mean(rewards_history_q_learning[-100:])
        print(f"Episode {episode}: Avg Reward={avg_reward:.4f}, Epsilon={epsilon:.3f}, Q-table size={len(q_table)}")

print("\nTabular Q-Learning Training Complete.")

# Plot training reward curve
plt.figure(figsize=(12, 6))
plt.plot(rewards_history_q_learning)
plt.title('Tabular Q-Learning: Training Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()

# Plot exploration decay
plt.figure(figsize=(12, 6))
epsilon_values = [1.0 * (epsilon_decay ** i) for i in range(n_episodes_q_learning)]
epsilon_values = [max(epsilon_min, e) for e in epsilon_values]
plt.plot(epsilon_values)
plt.title('Tabular Q-Learning: Epsilon Decay Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.grid(True)
plt.show()
```

### Explanation of Execution

The tabular Q-learning agent was trained on the discretized market states. The `discretize_state` function converted continuous market features into discrete bins, allowing the Q-values to be stored in a dictionary (acting as a table). The training loop applied the epsilon-greedy strategy, balancing exploration (random actions) with exploitation (actions with highest Q-value). The Q-values were updated using the Bellman equation, reflecting the agent's learning about the profitability of actions in different states.

The training reward curve shows the agent's cumulative reward per episode, ideally demonstrating an upward trend as it learns better strategies. The epsilon decay plot clearly illustrates the transition from high exploration to more exploitation over time. This foundational exercise highlights how an agent can learn a policy through trial and error, even in a simplified trading environment. While effective for small state spaces, this approach quickly becomes infeasible for complex, high-dimensional financial data due due to the curse of dimensionality.

## 5. Deep Q-Network (DQN): Scaling to Continuous State Spaces

### Story + Context + Real-World Relevance

Tabular Q-learning is limited by the "curse of dimensionality" – the number of possible states explodes with more features, making a Q-table impossible to store and update. In real-world finance, market states are continuous and high-dimensional. This is where **Deep Q-Networks (DQN)** come in. A DQN uses a neural network to approximate the Q-function, $Q_{\theta}(s, a)$, allowing it to handle continuous state spaces by learning a mapping from states to Q-values.

DQN introduced two key innovations to stabilize deep RL training:
1.  **Experience Replay**: The agent stores its experiences (state, action, reward, next state, done) in a replay buffer. During training, it samples random mini-batches from this buffer, breaking temporal correlations in sequential experiences. This is analogous to shuffling training data in supervised learning.
2.  **Target Network**: A separate, "frozen" copy of the Q-network (the target network, $Q_{\theta^-}(s', a')$) is used to calculate the target Q-values for the Bellman equation. This provides stable targets for the online network ($Q_{\theta}(s,a)$) to learn from, preventing the network from chasing a moving target (its own rapidly changing predictions), which can lead to oscillations or divergence. The target network's weights $\theta^-$ are updated periodically by copying the online network's weights $\theta$.

The DQN minimizes the mean squared TD error, where the loss function $L(\theta)$ is given by:
$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( R_{t+1} + \gamma \max_{a'} Q_{\theta^-}(S_{t+1}, a') - Q_{\theta}(S_t, A_t) \right)^2 \right]
$$
Here, $\mathcal{D}$ is the experience replay buffer, and $\theta^-$ denotes the parameters of the target network.

```python
class DQNAgent:
    """
    Deep Q-Network agent for trading, utilizing experience replay and a target network
    for stable learning in continuous state spaces.
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size) # Experience replay buffer
        self.gamma = gamma                    # Discount factor
        self.epsilon = epsilon                # Initial exploration rate
        self.epsilon_min = epsilon_min        # Minimum exploration rate
        self.epsilon_decay = epsilon_decay    # Epsilon decay factor
        self.learning_rate = learning_rate    # Adam optimizer learning rate

        self.model = self._build_model()        # Online network
        self.target_model = self._build_model() # Target network
        self.update_target_model()              # Initialize target network weights

    def _build_model(self):
        """
        Builds a neural network for approximating Q-values.
        Two hidden layers with ReLU activation, output layer with linear activation.
        """
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear') # Output Q-values for each action
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies weights from the online model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Epsilon-greedy action selection.
        Args:
            state (np.array): Current state observation.
        Returns:
            int: Selected action.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size) # Explore
        
        # Reshape state for model prediction (add batch dimension)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(q_values) # Exploit

    def replay(self, batch_size):
        """
        Trains the online network using a random mini-batch from the experience replay buffer.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Separate experiences into arrays
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Get Q-values for next states from the target network (stability)
        target_q_next = self.target_model.predict(next_states, verbose=0)
        
        # Calculate target Q-values for current states
        # R + gamma * max(Q(s', a')) if not done, else R
        targets = rewards + self.gamma * np.max(target_q_next, axis=1) * (1 - dones)

        # Get Q-values for current states from the online network
        current_q_values = self.model.predict(states, verbose=0)
        
        # Update only the Q-value for the action taken in the minibatch
        for i in range(batch_size):
            current_q_values[i][actions[i]] = targets[i]

        # Train the online network
        self.model.fit(states, current_q_values, epochs=1, verbose=0)

        # Decay epsilon after each replay step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Instantiate the DQN agent
state_size = train_env.observation_space.shape[0]
action_size = train_env.action_space.n
dqn_agent = DQNAgent(state_size, action_size, learning_rate=0.001, gamma=0.99, 
                    epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000)

n_episodes_dqn = 300 # Number of training episodes for DQN
batch_size = 32
target_update_frequency = 10 # Update target network every N episodes

dqn_rewards_history = []
dqn_epsilon_history = []

print("\nStarting Deep Q-Network (DQN) Training...")
for episode in range(n_episodes_dqn):
    obs, _ = train_env.reset()
    state = obs
    total_reward = 0
    done = False

    while not done:
        action = dqn_agent.act(state)
        next_obs, reward, done, _, info = train_env.step(action)
        
        # Store the experience
        dqn_agent.remember(state, action, reward, next_obs, done)
        
        state = next_obs
        total_reward += reward

        # Train the agent (replay)
        dqn_agent.replay(batch_size)

    dqn_rewards_history.append(total_reward)
    dqn_epsilon_history.append(dqn_agent.epsilon)

    # Periodically update the target network
    if episode % target_update_frequency == 0:
        dqn_agent.update_target_model()

    if episode % 10 == 0:
        avg_reward = np.mean(dqn_rewards_history[-10:]) if len(dqn_rewards_history) > 0 else 0
        print(f"Episode {episode}: Avg Reward={avg_reward:.4f}, Epsilon={dqn_agent.epsilon:.3f}")

print("\nDeep Q-Network (DQN) Training Complete.")

# Plot training reward curve
plt.figure(figsize=(12, 6))
plt.plot(dqn_rewards_history)
plt.title('DQN: Training Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()

# Plot exploration decay
plt.figure(figsize=(12, 6))
plt.plot(dqn_epsilon_history)
plt.title('DQN: Epsilon Decay Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.grid(True)
plt.show()
```

### Explanation of Execution

You have successfully implemented and trained a Deep Q-Network (DQN) agent. The `DQNAgent` class includes a neural network (`_build_model`) to approximate the Q-function, an `experience replay buffer` (`deque`) to store and sample past interactions, and a `target network` (`update_target_model`) for stable learning targets. During training, the agent explores actions using an epsilon-greedy strategy, collects experiences, and periodically trains its online network from sampled mini-batches. The target network is updated less frequently to provide stable Q-targets.

The training reward curve illustrates the agent's learning progress across episodes. An upward trend suggests the agent is discovering more profitable strategies. The epsilon decay plot confirms the expected behavior of gradually shifting from exploration to exploitation. By leveraging neural networks, experience replay, and target networks, the DQN agent can handle the continuous and complex state space of financial markets, overcoming the limitations of tabular Q-learning. This is a crucial step towards building more sophisticated algorithmic trading systems.

## 6. Out-of-Sample Evaluation and Strategy Analysis

### Story + Context + Real-World Relevance

After training, the most critical step for any quantitative trader is to evaluate the agent's performance on **unseen, out-of-sample data**. This tests the agent's ability to generalize its learned policy to new market conditions and prevents overfitting. You will run the trained DQN agent on the test dataset (2020-2024), which includes different market regimes (e.g., COVID crash, interest rate hikes, inflation concerns). The agent's performance will be compared against simple baselines: a **Buy-and-Hold strategy** and a **Moving Average Crossover strategy**. You will analyze key performance metrics like **Total Return**, **Sharpe Ratio**, and **Maximum Drawdown**, and visualize its trading decisions with an **action heatmap** to understand its learned policy.

The **portfolio value evolution** with transaction costs can be modeled as:
$$
V_{t+1} = \begin{cases} V_t \cdot (1 + r_{t+1}) - tc \cdot V_t & \text{if Buy at } t \\ V_t \cdot (1 + r_{t+1}) & \text{if Hold} \\ V_t \cdot (1 - tc) & \text{if Sell at } t \end{cases}
$$
where $r_{t+1}$ is the asset return and $tc$ is the transaction cost. This formulation ensures that transaction costs are accurately reflected in the agent's performance.

When evaluating agent performance, it's also important to consider **reward design alternatives**:
-   **Simple Return**: $R_t = \frac{V_{t+1} - V_t}{V_t}$. Easy to understand, but can encourage excessive risk.
-   **Log Return**: $R_t = \log \left( \frac{V_{t+1}}{V_t} \right)$. Encourages compounding and penalizes large losses more.
-   **Sharpe Reward**: $R_t = \frac{\mu_{recent}}{\sigma_{recent}}$. Directly optimizes risk-adjusted return over a recent window.
-   **Drawdown Penalty**: $R_t = r_t - \lambda \cdot DD_t$, where $DD_t$ is a measure of drawdown. Encourages capital preservation.

A critical **practitioner warning**: RL agents often overfit to training data. Good performance on the training set does NOT guarantee out-of-sample success. Rigorous testing and understanding the limitations (non-stationarity, reward hacking) are crucial.

```python
def evaluate_agent(agent, env, name="RL Agent"):
    """
    Runs the trained agent on the environment and tracks portfolio value and actions.
    Args:
        agent: The trained DQN agent.
        env: The trading environment (e.g., test_env).
        name (str): Name for the agent in output.
    Returns:
        tuple: (list of portfolio values, list of actions taken, total return, sharpe ratio, max drawdown)
    """
    obs, _ = env.reset()
    state = obs
    portfolio_values = [env.initial_cash]
    actions_taken = []
    
    # Set epsilon to 0 for evaluation (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.01 # Maintain a very small epsilon for practical robustness / exploration in test

    done = False
    while not done:
        action = agent.act(state)
        next_obs, reward, done, _, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        actions_taken.append(action)
        
        state = next_obs
    
    # Reset epsilon for potential further training
    agent.epsilon = original_epsilon

    # Calculate performance metrics
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

    # Annualized Sharpe Ratio
    # Assumes daily returns; multiply by sqrt(252) for annualization
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() != 0 else 0

    # Max Drawdown
    peak_value = np.maximum.accumulate(portfolio_values)
    drawdown = (peak_value - portfolio_values) / peak_value
    max_drawdown = np.max(drawdown)

    print(f"{name}: Return={total_return:.2%}, Sharpe={sharpe_ratio:.3f}, MaxDD={max_drawdown:.2%}")
    return portfolio_values, actions_taken, total_return, sharpe_ratio, max_drawdown

# Create the test environment
test_env = TradingEnv(test_prices, test_features, initial_cash=10000, transaction_cost=0.001)

# Evaluate the DQN agent on test data
dqn_portfolio_values, dqn_actions, dqn_total_return, dqn_sharpe_ratio, dqn_max_drawdown = evaluate_agent(dqn_agent, test_env, "DQN Agent")

# --- Baselines ---

# 1. Buy & Hold Baseline
bh_portfolio_values = [test_env.initial_cash * (price / test_prices[0]) for price in test_prices]
bh_returns = pd.Series(bh_portfolio_values).pct_change().dropna()
bh_total_return = (bh_portfolio_values[-1] / bh_portfolio_values[0]) - 1
bh_sharpe_ratio = bh_returns.mean() / bh_returns.std() * np.sqrt(252) if bh_returns.std() != 0 else 0
peak_value_bh = np.maximum.accumulate(bh_portfolio_values)
max_drawdown_bh = np.max((peak_value_bh - bh_portfolio_values) / peak_value_bh)
print(f"Buy & Hold: Return={bh_total_return:.2%}, Sharpe={bh_sharpe_ratio:.3f}, MaxDD={max_drawdown_bh:.2%}")

# 2. Simple Moving Average Crossover (SMA_10 > SMA_50 -> Buy/Hold, else Sell)
# Need to re-compute SMAs for the test_data without standardization for the rule logic
# and ensure we use original price data for calculation
raw_test_data = df_market.loc[test_data.index] # Get unstandardized data for SMA calculation
raw_test_data['sma_10'] = raw_test_data['price'].rolling(window=10).mean()
raw_test_data['sma_50'] = raw_test_data['price'].rolling(window=50).mean()
raw_test_data['sma_ratio_raw'] = raw_test_data['sma_10'] / raw_test_data['sma_50']
raw_test_data.dropna(inplace=True)

sma_actions = []
sma_portfolio_values = [test_env.initial_cash]
current_cash = test_env.initial_cash
current_shares = 0
current_position = 0 # 0=flat, 1=long

# Align with raw_test_data index
aligned_test_prices = raw_test_data['price'].values

for i in range(1, len(raw_test_data)):
    price = aligned_test_prices[i]
    prev_price = aligned_test_prices[i-1] # For calculating previous day's value
    
    # Calculate previous portfolio value for reward calculation
    prev_portfolio_value = current_cash + current_shares * prev_price
    
    sma_ratio_val = raw_test_data['sma_ratio_raw'].iloc[i]

    action = 0 # Default to Hold
    if sma_ratio_val > 1.05 and current_position == 0: # Buy if short-term SMA above long-term with a buffer
        action = 1
    elif sma_ratio_val < 0.95 and current_position == 1: # Sell if short-term SMA below long-term with a buffer
        action = 2

    # Execute action with transaction costs
    if action == 1: # Buy
        if current_position == 0:
            shares_to_buy = int((current_cash * (1 - test_env.transaction_cost)) / price)
            current_shares += shares_to_buy
            current_cash -= shares_to_buy * price
            current_position = 1
    elif action == 2: # Sell
        if current_position == 1:
            current_cash += current_shares * price * (1 - test_env.transaction_cost)
            current_shares = 0
            current_position = 0
    # Hold action (0) does nothing to shares/cash

    sma_actions.append(action)
    sma_portfolio_values.append(current_cash + current_shares * price)

sma_returns = pd.Series(sma_portfolio_values).pct_change().dropna()
sma_total_return = (sma_portfolio_values[-1] / sma_portfolio_values[0]) - 1
sma_sharpe_ratio = sma_returns.mean() / sma_returns.std() * np.sqrt(252) if sma_returns.std() != 0 else 0
peak_value_sma = np.maximum.accumulate(sma_portfolio_values)
max_drawdown_sma = np.max((peak_value_sma - sma_portfolio_values) / peak_value_sma)
print(f"SMA Crossover: Return={sma_total_return:.2%}, Sharpe={sma_sharpe_ratio:.3f}, MaxDD={max_drawdown_sma:.2%}")

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Plot Equity Curves
ax1.plot(test_data.index, dqn_portfolio_values, label='DQN Agent', linewidth=1.5, color='blue')
ax1.plot(test_data.index, bh_portfolio_values, label='Buy & Hold', linewidth=1.5, linestyle='--', color='gray')
ax1.plot(test_data.index, sma_portfolio_values, label='SMA Crossover', linewidth=1.5, linestyle=':', color='orange')
ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title('RL Trading Agent vs. Baselines (Out-of-Sample Evaluation)')
ax1.legend()
ax1.grid(True)

# Plot Action Heatmap for DQN Agent
# Use the same index as the equity curve, but actions are one day delayed
action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
action_colors = {0: 'lightgray', 1: 'green', 2: 'red'} # Adjusted colors for better visibility

# We need to ensure the length of dqn_actions matches the test_data.index for plotting
# The evaluate_agent loop runs len(prices)-1 steps, so actions_taken will be len(prices)-1
# We plot actions over the period for which they are taken (i.e., day i's action is for day i+1)
# Let's align it such that actions are plotted for the day they are *taken*, affecting the next day's price.
# This means dqn_actions corresponds to test_data.index[:-1]
# To align visually with the price action, we can plot actions for the day *on which* the action is taken.

# Make sure plot covers the same number of days for actions as for portfolio values.
# If dqn_actions has length N, and prices has length N+1, plot actions for indices 0 to N-1.
# The `evaluate_agent` function collects actions corresponding to `test_data.index[:-1]`.
# So we need to ensure the index matches.
action_plot_index = test_data.index[1:] # Actions taken on day i affect portfolio on day i+1

# Plotting actions as a heatmap/scatter on the second subplot
ax2.plot(action_plot_index, dqn_actions, marker='o', linestyle='', alpha=0.6, 
         color=[action_colors[a] for a in dqn_actions], markersize=3, label='DQN Actions')
ax2.set_yticks(list(action_names.keys()))
ax2.set_yticklabels(list(action_names.values()))
ax2.set_ylabel('Agent Actions')
ax2.set_title('DQN Agent Decisions on Test Data (Green=Buy, Red=Sell, Gray=Hold)')
ax2.grid(True, axis='y') # Only horizontal grid for actions for clarity
ax2.set_xlabel('Date')

plt.tight_layout()
plt.show()

# Display summary statistics
summary_data = {
    'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown'],
    'DQN Agent': [f"{dqn_total_return:.2%}", f"{dqn_sharpe_ratio:.3f}", f"{dqn_max_drawdown:.2%}"],
    'Buy & Hold': [f"{bh_total_return:.2%}", f"{bh_sharpe_ratio:.3f}", f"{max_drawdown_bh:.2%}"],
    'SMA Crossover': [f"{sma_total_return:.2%}", f"{sma_sharpe_ratio:.3f}", f"{max_drawdown_sma:.2%}"]
}
summary_df = pd.DataFrame(summary_data)
print("\n--- Performance Summary (Out-of-Sample) ---")
print(summary_df.to_markdown(index=False))
```

### Explanation of Execution

The out-of-sample evaluation revealed how the DQN agent performs on unseen market data. The equity curve plot visually compares the agent's portfolio growth against the Buy-and-Hold strategy and a simple SMA Crossover baseline. The summary statistics (Total Return, Sharpe Ratio, Max Drawdown) provide quantitative measures of profitability and risk. The action heatmap on the second subplot offers a crucial insight into the agent's learned policy, showing when it decided to Buy, Sell, or Hold over time.

As a CFA Charterholder, analyzing these results helps you understand the agent's strengths and weaknesses. Often, RL agents may not outperform simple baselines out-of-sample due to challenges like market non-stationarity and overfitting. This reinforces the critical lesson that backtested performance in RL is not a guarantee of future live performance. A truly robust strategy requires careful validation, ongoing monitoring, and a realistic understanding of its limitations in complex, adaptive environments like financial markets. This analysis is a starting point for further qualitative inspection and potentially refining the reward function or agent architecture.

## 7. Limitations and Practical Considerations for RL in Finance

### Story + Context + Real-World Relevance

While reinforcement learning offers a powerful paradigm for sequential decision-making, its application in finance comes with significant practical limitations. As a financial professional, recognizing these challenges is crucial for fostering a realistic perspective and avoiding over-reliance on purely autonomous systems.

1.  **Overfitting to Training Data**: RL agents, especially deep ones, are prone to memorizing training patterns rather than learning robust, generalized strategies. Financial markets are highly non-stationary; past patterns may not repeat. An agent that performs exceptionally well on historical data might fail dramatically in live trading, a phenomenon often referred to as the **simulation-to-reality gap**.

2.  **Non-Stationarity of Financial Markets**: The underlying data distribution in financial markets changes constantly due to evolving economic conditions, regulations, and human behavior. RL models assume a stationary MDP, which is rarely true in finance. This makes it challenging for agents to adapt to new regimes without extensive retraining or sophisticated meta-learning techniques.

3.  **Reward Design Complexity**: The agent will learn to maximize *whatever reward function you specify*. If your reward function doesn't perfectly align with your true investment objectives (e.g., only maximizing simple return might lead to excessively risky behavior), the agent could find "reward hacking" strategies that exploit flaws in the simulation rather than genuine market inefficiencies. Designing a comprehensive reward function that balances return, risk, and other constraints (like transaction costs, drawdowns, liquidity) is a complex, iterative process and a financial design decision, not merely a technical one.

4.  **Sample Efficiency**: Deep RL agents typically require millions of environmental interactions to learn. Generating this amount of high-quality, diverse historical financial data for training can be challenging. Simulators are crucial but can introduce biases if not realistic enough.

5.  **Interpretability and Explainability**: Understanding *why* an RL agent made a particular trading decision can be difficult, especially with deep neural networks. For compliance, risk management, and client communication, financial professionals often need explainable models, which is an active research area in RL.

6.  **Ethical Considerations and Oversight**: Autonomous trading systems raise critical questions about accountability, market impact, and fairness. The CFA Institute emphasizes that RL agents should assist human traders, not replace them without supervision, advocating for "controlling autonomy in high-stakes domains."

**Where RL Actually Works in Finance (More Successfully):**
While direct stock prediction is highly challenging, RL has seen more success in narrower, better-defined financial problems with clearer reward signals and shorter horizons:
*   **Optimal Trade Execution**: Splitting large orders to minimize market impact.
*   **Portfolio Rebalancing Timing**: Deciding when to rebalance a multi-asset portfolio to optimize utility and manage transaction costs.
*   **Options Hedging**: Learning dynamic hedge ratios for derivatives portfolios, incorporating transaction costs and market frictions.

This lab provided a simplified environment to grasp core RL concepts. In real-world applications, further sophistication (multi-asset, continuous action spaces, more realistic transaction costs, diverse reward functions, and robust validation techniques) is required, always with an eye on the inherent limitations.

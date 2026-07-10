import numpy as np
from torch.utils.data import Dataset


def discount_returns(rewards, gamma, scale):
    """
    Compute discounted cumulative returns from rewards.
    
    Args:
        rewards: list or array of reward values
        gamma: discount factor (typically 0 to 1)
        scale: scaling factor to normalize returns (should be > 0)
    
    Returns:
        list of discounted returns
    """
    # Input validation
    if not rewards or len(rewards) == 0:
        return []
    
    if scale == 0:
        raise ValueError(f"Scale cannot be zero. Got scale={scale}")
    
    if gamma < 0 or gamma > 1:
        raise ValueError(f"Gamma should be between 0 and 1. Got gamma={gamma}")
    
    # Convert to numpy array for easier handling
    rewards_arr = np.array(rewards, dtype=np.float32)
    
    # Check for invalid values in rewards
    if np.any(np.isnan(rewards_arr)):
        print("[WARNING] NaN values found in rewards")
        rewards_arr = np.nan_to_num(rewards_arr, nan=0.0)
    
    if np.any(np.isinf(rewards_arr)):
        print("[WARNING] Inf values found in rewards")
        rewards_arr = np.nan_to_num(rewards_arr, posinf=1e6, neginf=-1e6)
    
    # Compute discounted returns
    returns = np.zeros_like(rewards_arr)
    returns[-1] = rewards_arr[-1]
    
    for i in reversed(range(len(rewards_arr) - 1)):
        returns[i] = rewards_arr[i] + gamma * returns[i + 1]
        
        # Check for overflow/underflow
        if np.isnan(returns[i]) or np.isinf(returns[i]):
            print(f"[WARNING] NaN/Inf detected at index {i}: {returns[i]}")
            print(f"  rewards[i]={rewards_arr[i]}, gamma={gamma}, returns[i+1]={returns[i + 1]}")
            returns[i] = np.clip(returns[i], -1e6, 1e6)
    
    # Scale down returns
    returns = returns / scale
    
    # Final check and clip extreme values
    if np.any(np.isnan(returns)):
        print("[WARNING] NaN values in final returns after scaling")
        returns = np.nan_to_num(returns, nan=0.0)
    
    if np.any(np.isinf(returns)):
        print("[WARNING] Inf values in final returns after scaling")
        returns = np.clip(returns, -1e6, 1e6)
    
    return returns.tolist()


class ExperienceDataset(Dataset):
    """
    A dataset class that wraps the experience pool.
    """
    def __init__(self, exp_pool, gamma=1., scale=10, max_length=30, sample_step=None) -> None:
        """
        :param exp_pool: the experience pool
        :param gamma: the reward discounted factor
        :param scale: the factor to scale the return
        :param max_length: the w value in our paper, see the paper for details.
        """
        if sample_step is None:
            sample_step = max_length

        self.exp_pool = exp_pool
        self.exp_pool_size = len(exp_pool)
        self.gamma = gamma
        self.scale = scale
        self.max_length = max_length

        self.returns = []
        self.timesteps = []
        self.rewards = []

        self.exp_dataset_info = {}

        self._normalize_rewards()
        self._compute_returns()
        self.exp_dataset_info.update({
            'max_action': max(self.actions),
            'min_action': min(self.actions)
        })

        self.dataset_indices = list(range(0, self.exp_pool_size - max_length + 1, min(sample_step, max_length)))
    
    def sample_batch(self, batch_size=1, batch_indices=None):
        """
        Sample a batch of data from the experience pool.
        :param batch_size: the size of a batch. For CJS task, batch_size should be set to 1 due to the unstructural data format.
        """
        if batch_indices is None:
            batch_indices = np.random.choice(len(self.dataset_indices), size=batch_size)
        batch_states, batch_actions, batch_returns, batch_timesteps = [], [], [], []
        for i in range(batch_size):
            states, actions, returns, timesteps = self[batch_indices[i]]
            batch_states.append(states)
            batch_actions.append(actions)
            batch_returns.append(returns)
            batch_timesteps.append(timesteps)
        return batch_states, batch_actions, batch_returns, batch_timesteps
    
    @property
    def states(self):
        return self.exp_pool.states

    @property
    def actions(self):
        return self.exp_pool.actions
    
    @property
    def dones(self):
        return self.exp_pool.dones
    
    def __len__(self):
        return len(self.dataset_indices)
    
    def __getitem__(self, index):
        start = self.dataset_indices[index]
        end = start + self.max_length
        return self.states[start:end], self.actions[start:end], self.returns[start:end], self.timesteps[start:end]

    def _normalize_rewards(self):
        min_reward, max_reward = min(self.exp_pool.rewards), max(self.exp_pool.rewards)
        rewards = (np.array(self.exp_pool.rewards) - min_reward) / (max_reward - min_reward)
        self.rewards = rewards.tolist()
        print(f"[INFO] Normalized rewards with min_reward={min_reward}, max_reward={max_reward}")
        self.exp_dataset_info.update({
            'max_reward': max_reward,
            'min_reward': min_reward,
        })

    def _compute_returns(self):
        """
        Compute returns (discounted cumulative rewards)
        """
        episode_start = 0
        while episode_start < self.exp_pool_size:
            try:
                episode_end = self.dones.index(True, episode_start) + 1
            except ValueError:
                episode_end = self.exp_pool_size
            self.returns.extend(discount_returns(self.rewards[episode_start:episode_end], self.gamma, self.scale))
            self.timesteps += list(range(episode_end - episode_start))
            episode_start = episode_end
        assert len(self.returns) == len(self.timesteps)
        self.exp_dataset_info.update({
            # for normalizing rewards/returns
            'max_return': max(self.returns),
            'min_return': min(self.returns),

            # to help determine the maximum size of timesteps embedding
            'min_timestep': min(self.timesteps),
            'max_timestep': max(self.timesteps),
        })

    # def _compute_returns(self):
    #     """
    #     Compute returns (discounted cumulative rewards)
    #     """
    #     print("[DEBUG] Starting _compute_returns")
    #     episode_start = 0
    #     print(f"[DEBUG] exp_pool_size={self.exp_pool_size}")
    #
    #     while episode_start < self.exp_pool_size:
    #         print(f"\n[DEBUG] New episode starting at index {episode_start}")
    #
    #         # Find episode end based on done flags
    #         try:
    #             episode_end = self.dones.index(True, episode_start) + 1
    #             print(f"[DEBUG] Found done=True at {episode_end - 1}, episode_end={episode_end}")
    #         except ValueError:
    #             episode_end = self.exp_pool_size
    #             print(f"[DEBUG] No done=True found; using exp_pool_size={episode_end}")
    #
    #         # Slice rewards for this episode
    #         reward_slice = self.rewards[episode_start:episode_end]
    #         print(f"[DEBUG] reward_slice length={len(reward_slice)} values={reward_slice}")
    #
    #         # Compute discounted returns
    #         episode_returns = discount_returns(reward_slice, self.gamma, self.scale)
    #         print(f"[DEBUG] episode_returns length={len(episode_returns)} values={episode_returns}")
    #
    #         self.returns.extend(episode_returns)
    #
    #         # Add timesteps
    #         episode_timesteps = list(range(episode_end - episode_start))
    #         print(f"[DEBUG] episode_timesteps={episode_timesteps}")
    #         self.timesteps += episode_timesteps
    #
    #         # Move to next episode
    #         episode_start = episode_end
    #
    #     print("\n[DEBUG] Finished processing episodes")
    #     print(f"[DEBUG] Total returns={len(self.returns)}, Total timesteps={len(self.timesteps)}")
    #     assert len(self.returns) == len(self.timesteps)
    #
    #     # Update dataset info
    #     self.exp_dataset_info.update({
    #         'max_return': max(self.returns),
    #         'min_return': min(self.returns),
    #         'min_timestep': min(self.timesteps),
    #         'max_timestep': max(self.timesteps),
    #     })
    #
    #     print("\n[DEBUG] Dataset info updated:")
    #     print(f"[DEBUG]   max_return={self.exp_dataset_info['max_return']}")
    #     print(f"[DEBUG]   min_return={self.exp_dataset_info['min_return']}")
    #     print(f"[DEBUG]   min_timestep={self.exp_dataset_info['min_timestep']}")
    #     print(f"[DEBUG]   max_timestep={self.exp_dataset_info['max_timestep']}")
    #     print("[DEBUG] _compute_returns complete\n")


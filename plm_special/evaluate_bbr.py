import numpy as np
import torch
import time
import json
import psutil
import GPUtil
from munch import Munch
from torch.utils.data import DataLoader
import pandas as pd

import os
from datetime import datetime
import random
import pickle
from plm_special.utils.utils import process_batch
from plm_special.data.dataset import ExperienceDataset



col_dict = {
    'location': 0,
    't': 1,
    'bps': 2,
    'retransmits': 3,
    'snd_cwnd': 4,
    'snd_wnd': 5,
    'rtt': 6,
    'rttvar': 7,
    'Action': 10
}

results_to_csv = []


def tensor_to_list(tensor):
    # Detach the tensor and then convert it to a NumPy array and then to a list
    return tensor.detach().cpu().numpy().tolist()


def convert_exp_pool_to_dataframe(exp_pool, csv_output_path='exp_pool_data.csv', dict_output_path='exp_pool_dict.pkl'):
    """
    Converts the given experience pool into a pandas DataFrame.
    Optionally saves the DataFrame to a CSV file and the experience pool as a dictionary to a pickle file.

    Args:
        exp_pool (object): The experience pool object containing states, actions, rewards, and dones.
        csv_output_path (str): Path to save the resulting DataFrame as a CSV file (default: 'exp_pool_data.csv').

    Returns:
        pd.DataFrame: The DataFrame representation of the experience pool.
    """

    # Step 1: Convert the Experience Pool to a DataFrame

    # Create state column names based on the length of each state vector
    state_columns = [f'state_{i}' for i in range(len(exp_pool.states[0]))]  # Assuming each state is a 1D array

    # Flatten the states into individual columns
    expanded_states = np.array([state for state in exp_pool.states])

    # Create the DataFrame with expanded states
    df = pd.DataFrame(expanded_states, columns=state_columns)

    # Add actions, rewards, and dones as columns to the DataFrame
    df['actions'] = exp_pool.actions
    df['rewards'] = exp_pool.rewards
    df['dones'] = exp_pool.dones

    # Step 2: Save the DataFrame to a CSV file
    df.to_csv(csv_output_path, index=False)
    # print(f"DataFrame saved successfully to: {csv_output_path}")
    return df


def find_nearest_length(df, user_input):
    # print("||||||||||||||||" * 40)
    # print("df in function find_nearest_length")
    if df.empty:
        # Handle the empty DataFrame case
        # print("DataFrame is empty, returning None.")
        return None  # Return None or another suitable default value

    # Calculate the absolute difference with user input
    # print("user_input", user_input)

    nearest_idx = (df['state_6'] - user_input).abs().idxmin()
    # print("22nearest_idx", nearest_idx)

    if nearest_idx >= len(df):
        print("Outside df_ats limits")

    if nearest_idx is None:
        # print("No valid index found, returning None.")
        return None  # Return None or another suitable default value

    # print("||||||||||||||||" * 40)

    return nearest_idx


def test_step(args, model, loss_fn, raw_batch, target_return):
    # Assuming raw_batch is a tuple of numpy arrays or lists
    states, actions, returns, timesteps = raw_batch

    # Convert states to tensor and ensure correct shape
    states = torch.tensor(states[0], dtype=torch.float32).to(args.device).unsqueeze(0)  # Shape [1, 8]

    # Convert actions, returns, and timesteps to tensors
    actions = torch.tensor(actions, dtype=torch.float32).to(args.device)  # Shape [1, 1]
    returns = torch.tensor(returns, dtype=torch.float32).to(args.device)  # Shape [1, 1]
    timesteps = torch.tensor(timesteps, dtype=torch.int32).to(args.device)  # Shape [1, 1]

    # Create a batch with the correctly formatted tensors
    # Wrap states in a list to avoid TypeError in process_batch
    batch = ([states], [actions], [returns], [timesteps])  # Ensure states is a list

    # Call process_batch
    states, actions, returns, timesteps, labels = process_batch(batch, device=args.device)

    # Predict actions using the model
    # actions_pred1 = model(states, actions, returns, timesteps)
    queue_action = 0
    actions_pred1, queue_action = model.sample(states, target_return, timesteps)

    # Permute for loss calculation
    actions_pred = actions_pred1.permute(0, 2, 1)
    loss = loss_fn(actions_pred, labels)

    # print("actions_pred1", actions_pred1)
    # print("actions_pred", actions_pred)
    # print("llm-queue_action", queue_action)
    # print("actual-queue_action", labels)

    return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred


def otest_step(args, model, loss_fn, raw_batch, target_return):
    # Assuming raw_batch is a tuple of numpy arrays or lists
    states, actions, returns, timesteps = raw_batch

    # Convert states to tensor and ensure correct shape
    states = torch.tensor(states[0], dtype=torch.float32).to(args.device).unsqueeze(0)  # Shape [1, 8]
    # print("Tensor states:", states)
    # print("Tensor states.shape:", states.shape)  # Should be [1, 8]

    # Convert actions, returns, and timesteps to tensors
    actions = torch.tensor(actions, dtype=torch.float32).to(args.device)  # Shape [1, 1]
    returns = torch.tensor(returns, dtype=torch.float32).to(args.device)  # Shape [1, 1]
    timesteps = torch.tensor(timesteps, dtype=torch.int32).to(args.device)  # Shape [1, 1]

    # Create a batch with the correctly formatted tensors
    # Wrap states in a list to avoid TypeError in process_batch
    batch = ([states], [actions], [returns], [timesteps])  # Ensure states is a list

    # Call process_batch
    states, actions, returns, timesteps, labels = process_batch(batch, device=args.device)

    # Predict actions using the model
    # actions_pred1 = model(states, actions, returns, timesteps)
    actions_pred1 = model(states, actions, returns, timesteps)

    # Permute for loss calculation
    actions_pred = actions_pred1.permute(0, 2, 1)
    loss = loss_fn(actions_pred, labels)

    queue_action = 0

    # print("actions_pred1", actions_pred1)
    # print("actions_pred", actions_pred)
    # print("actual-queue_action", labels)

    return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred


import os
import torch


def evaluate_on_simulated_env(args, model, exp_pool, target_return, loss_fn, process_reward_fn=None, seed=0):
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    # Create experience dataset and convert the experience pool to a DataFrame
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=1)
    df = convert_exp_pool_to_dataframe(exp_pool)
    max_ep_len = len(df)
    state_columns = [f'state_{i}' for i in range(len(exp_pool.states[0]))]

    # Truncate the log file (if needed)
    with open('output_log.txt', 'w') as file:
        pass

    # List to store new_action for each episode
    new_action_list = []

    for ep_index in range(max_ep_len):
        print(f"ep_index: {ep_index}")
        row = df.iloc[ep_index]
        state = np.array(row[state_columns], dtype=np.float32)
        current_action = row['actions']
        reward = row['rewards']
        done = 0
        batch = [state], [current_action], [reward], [done]

        # Run the test step to get predictions
        _, _, _, _, _, _, _, actions_pred = otest_step(args, model, loss_fn, batch, target_return)

        # Compute new_action by taking the argmax over predictions; assuming one action per episode
        new_action = actions_pred.detach().cpu().numpy().argmax(axis=1).flatten()
        new_action_list.append(new_action[0])

    # Define the CSV filename
    csv_filename = "ep_action_results.csv"

    # Load existing CSV if it exists; otherwise create a new DataFrame with an ep_index column
    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
    else:
        df_existing = pd.DataFrame({'ep_index': list(range(max_ep_len))})

    # Determine a new column name based on existing new_action_run columns
    # existing_columns = [col for col in df_existing.columns if col.startswith("new_action_run")]
    # run_number = len(existing_columns) + 1
    new_column_name = f"{args.plm_type}"

    # If necessary, extend the DataFrame to match the current number of episodes
    if len(df_existing) < max_ep_len:
        additional_rows = pd.DataFrame({'ep_index': list(range(len(df_existing), max_ep_len))})
        df_existing = pd.concat([df_existing, additional_rows], ignore_index=True)

    # Add the new column; if there are extra rows, fill them with None
    df_existing[new_column_name] = new_action_list + [None] * (len(df_existing) - len(new_action_list))

    # Save the updated DataFrame to CSV
    df_existing.to_csv(csv_filename, index=False)
    print(f"✅ Results saved successfully to '{csv_filename}' under column '{new_column_name}'.")



import os
import random
import numpy as np
from plm_special.utils.constants import ACTION_LEVELS
try:  # baselines use a different conda environment without torch, so we need to skip ModuleNotFoundError when runing baselines
    import torch
except ModuleNotFoundError:
    pass


def _process_tensors(batch, device):
    states, actions, returns, timesteps = batch

    states = torch.cat(states, dim=0).unsqueeze(0).float().to(device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(1, -1)
    labels = actions.long()
    if torch.any(actions != labels) or torch.any(labels < 0) or torch.any(labels >= ACTION_LEVELS):
        raise ValueError(
            f"Action labels must be integer indices in [0, {ACTION_LEVELS - 1}]"
        )
    actions = ((actions + 1) / ACTION_LEVELS).unsqueeze(2)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device).reshape(1, -1, 1)
    timesteps = torch.as_tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)

    return states, actions, returns, timesteps, labels


def process_batch(batch, device='cpu'):
    """Process a legacy non-BBR batch."""
    if len(batch) != 4:
        raise ValueError("process_batch expects states, actions, returns, and timesteps")
    return _process_tensors(batch, device)


def process_bbr_batch(batch, device='cpu'):
    """Process a BBR batch and preserve its per-sample macro-phases."""
    if len(batch) != 5:
        raise ValueError(
            "BBR batches require states, actions, returns, timesteps, and phases"
        )

    states, actions, returns, timesteps, collated_phases = batch
    tensors = _process_tensors((states, actions, returns, timesteps), device)
    phases = []
    for item in collated_phases:
        if isinstance(item, str):
            phases.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], str):
            phases.append(item[0])
        else:
            raise ValueError("Unsupported collated BBR phase value: {!r}".format(item))

    labels = tensors[-1].reshape(-1).tolist()
    if len(phases) != len(labels):
        raise ValueError(
            "BBR phase count {} does not match action count {}".format(
                len(phases), len(labels)
            )
        )

    from utils.bbr import validate_phase_action
    for phase, action in zip(phases, labels):
        validate_phase_action(phase, action)

    return (*tensors, tuple(phases))


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def calc_mean_reward(result_files, test_dir, str, skip_first_reward=True):
    matching = [s for s in result_files if str in s]
    reward = []
    count = 0
    for log_file in matching:
        count += 1
        first_line = True
        with open(test_dir + '/' + log_file, 'r') as f:
            for line in f:
                parse = line.split()
                if len(parse) <= 1:
                    break
                if first_line:
                    first_line = False
                    if skip_first_reward:
                        continue
                reward.append(float(parse[7]))
    print(count)
    return np.mean(reward)


def clear_dir(directory):
    file_list = os.listdir(directory)
    for file in file_list:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

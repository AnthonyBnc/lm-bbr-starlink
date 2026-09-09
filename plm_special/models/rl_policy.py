import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

from plm_special.classical_head import ClassicalBottleneckActionHead
from plm_special.quantum_head import QuantumActionHead
from utils.bbr import ACTION_LEVELS, mask_action_logits, validate_phase
    

INF = 1e5


class OfflineRLPolicy(nn.Module):
    def __init__(
            self,
            state_feature_dim,
            action_levels,
            state_encoder,
            plm,
            plm_embed_size,
            max_length=None,
            max_ep_len=100,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out = None,
            residual = False, 
            conv_size = 4,  
            which_layer = -1,  # for early stopping: specify which layer to stop
            head_type = "classical",
            quantum_config = None,
            **kwargs
    ):
        super().__init__()

        if action_levels != ACTION_LEVELS:
            raise ValueError(
                f"OfflineRLPolicy requires exactly {ACTION_LEVELS} BBR actions; "
                f"got {action_levels}"
            )
        
        if device_out is None:
            device_out = device

        self.action_levels = action_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size

        # =========== multimodal encoder (start) ===========
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        self.embed_state1 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state2 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    
        self.embed_state3 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    
        self.embed_state4 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    
        self.embed_state5 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state6 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state7 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state8 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state9 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
 

        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        # =========== multimodal encoder (end) ===========

        self.head_type = head_type
        self.quantum_config = None
        if head_type == "classical":
            self.action_head = nn.Linear(plm_embed_size, action_levels).to(device)
        elif head_type == "classical_twin":
            config = quantum_config or {}
            self.action_head = ClassicalBottleneckActionHead(
                plm_embed_size,
                action_levels,
                bottleneck_dim=int(config.get("n_qubits", 4)),
                input_layernorm=bool(config.get("input_layernorm", False)),
                temperature=float(config.get("temperature", 1.0)),
            ).to(device)
        elif head_type == "quantum":
            config = quantum_config or {}
            self.action_head = QuantumActionHead(
                plm_embed_size,
                action_levels,
                n_qubits=int(config.get("n_qubits", 4)),
                depth=int(config.get("depth", 2)),
                ansatz=config.get("ansatz", "trainable_ry_layers"),
                input_layernorm=bool(config.get("input_layernorm", False)),
                temperature=float(config.get("temperature", 1.0)),
                angle_scale=config.get("angle_scale", "pi"),
                backend=config.get("backend", "qiskit"),
            ).to(device)
            self.quantum_config = self.action_head.manifest_config()
        else:
            raise ValueError(
                "Unknown head_type {!r}; expected classical, classical_twin, or quantum".format(
                    head_type
                )
            )

        print("rl_policy: action_levels",action_levels)

        self.device = device
        self.device_out = device_out

        # the following are used for evaluation
        self.states_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.actions_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)

        self.residual = residual
        self.which_layer = which_layer
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_ln, 
            self.embed_state1, self.embed_state2, self.embed_state3, self.embed_state4, self.embed_state5,
            self.embed_state6,self.embed_state7,self.embed_state8, self.embed_state9, self.action_head
        ])

    def _action_head_input_dtype(self):
        return next(self.action_head.parameters()).dtype

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        Forward function, used for training.
        """
        assert actions.shape[0] == 1, 'batch size should be 1 to avoid CUDA memory exceed'

        # Step 1: process actions, returns and timesteps first as they are simple
        actions = actions.to(self.device)  # shape: (1, seq_len, 1)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)

        # 1.1 embed action, return, timestep
        action_embeddings = self.embed_action(actions)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns)  # shape: (1, seq_len, embed_size)
        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)

        # 1.2 time embeddings are treated similar to positional embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Step 2: process states, turn them into embeddings.
        states = states.to(self.device)  # shape: (1, seq_len, 6, 6)
        states_features = self.state_encoder(states)
        states_embeddings1 = self.embed_state1(states_features[0]) + time_embeddings
        states_embeddings2 = self.embed_state2(states_features[1]) + time_embeddings
        states_embeddings3 = self.embed_state3(states_features[2]) + time_embeddings
        states_embeddings4 = self.embed_state4(states_features[3]) + time_embeddings
        states_embeddings5 = self.embed_state5(states_features[4]) + time_embeddings
        states_embeddings6 = self.embed_state6(states_features[5]) + time_embeddings
        states_embeddings7 = self.embed_state7(states_features[6]) + time_embeddings
        states_embeddings8 = self.embed_state8(states_features[7]) + time_embeddings
        states_embeddings9 = self.embed_state9(states_features[8]) + time_embeddings

        
        # Step 3: stack returns, states, actions embeddings.
        # this makes the sequence look like (R_1, s_1-1, s_1-2, ..., s_1-n, a_1, R_2, s_2-1, ..., s_2-m, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = []
        action_embed_positions = []  # record the positions of action embeddings
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((returns_embeddings[0, i:i + 1], states_embeddings1[0, i:i + 1], states_embeddings2[0, i:i + 1], 
                                       states_embeddings3[0, i:i + 1], states_embeddings4[0, i:i + 1], states_embeddings5[0, i:i + 1], 
                                       states_embeddings6[0, i:i + 1], states_embeddings7[0, i:i + 1], states_embeddings8[0, i:i + 1],
                                       states_embeddings9[0, i:i + 1], action_embeddings[0, i:i + 1]), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions.append((i + 1) * (2 + 9))
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        max_context = getattr(self.plm.config, 'max_position_embeddings', None)
        if max_context is not None and stacked_inputs.shape[1] > max_context:
            raise ValueError(
                f"Structured sequence has {stacked_inputs.shape[1]} tokens, "
                f"exceeding model context length {max_context}"
            )
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
        
        # Step 4: feed stacked embeddings into the plm
        # 4.1 create attention mask
        if attention_mask is None:
            # 1 if can be attended to, 0 if not
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        plm_dtype = next(self.plm.parameters()).dtype
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln.to(dtype=plm_dtype),
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Handle different model output formats
        if isinstance(transformer_outputs, dict):
            if 'last_hidden_state' in transformer_outputs:
                logits = transformer_outputs['last_hidden_state']
            elif 'hidden_states' in transformer_outputs and len(transformer_outputs['hidden_states']) > 0:
                logits = transformer_outputs['hidden_states'][-1]
            else:
                # Try to get the first tensor output
                logits = next(iter(transformer_outputs.values()))
        else:
            # If output is a tuple/object with attributes
            if hasattr(transformer_outputs, 'last_hidden_state'):
                logits = transformer_outputs.last_hidden_state
            elif hasattr(transformer_outputs, 'hidden_states') and len(transformer_outputs.hidden_states) > 0:
                logits = transformer_outputs.hidden_states[-1]
            else:
                # Default to first element if it's a tuple
                logits = transformer_outputs[0]
        
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 5: predict actions
        # we need to locate the logits corresponding to the state embeddings
        # simply using `action_embed_positions[i] - 2` will do.
        action_embed_positions = torch.as_tensor(
            action_embed_positions, dtype=torch.long, device=logits.device
        )
        logits_used = logits[:, action_embed_positions - 2]
        logits_used = logits_used.to(dtype=self._action_head_input_dtype())
        action_pred = self.action_head(logits_used)

        return action_pred
    
    def reset_dq(self):
        self.states_dq = deque([torch.zeros((1, 0, self.plm_embed_size), device=self.device)], maxlen=self.max_length)
        self.returns_dq = deque([torch.zeros((1, 0, self.plm_embed_size), device=self.device)], maxlen=self.max_length)
        self.actions_dq = deque([torch.zeros((1, 0, self.plm_embed_size), device=self.device)], maxlen=self.max_length)

    def sample(self, state, target_return, timestep, phase=None, **kwargs):
        """
        Sample action function, used for evaluation/testing.
        """
        phase = validate_phase(phase)

        # Step 1: stack previous state, action, return features in the dequeue
        prev_stacked_inputs = []
        for i in range(len(self.states_dq)):
            prev_return_embeddings = self.returns_dq[i]
            prev_state_embeddings = self.states_dq[i]
            prev_action_embeddings = self.actions_dq[i]
            prev_stacked_inputs.append(torch.cat((prev_return_embeddings, prev_state_embeddings, prev_action_embeddings), dim=1))
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        # Step 2: process target return and timesteps
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)

        return_embeddings = return_embeddings + time_embeddings

        # Step 4: process state
        state = state.to(self.device)
        state_features = self.state_encoder(state)
        state_embeddings1 = self.embed_state1(state_features[0]) + time_embeddings
        state_embeddings2 = self.embed_state2(state_features[1]) + time_embeddings
        state_embeddings3 = self.embed_state3(state_features[2]) + time_embeddings
        state_embeddings4 = self.embed_state4(state_features[3]) + time_embeddings
        state_embeddings5 = self.embed_state5(state_features[4]) + time_embeddings
        state_embeddings6 = self.embed_state6(state_features[5]) + time_embeddings
        state_embeddings7 = self.embed_state7(state_features[6]) + time_embeddings
        state_embeddings8 = self.embed_state8(state_features[7]) + time_embeddings
        state_embeddings9 = self.embed_state9(state_features[8]) + time_embeddings

        state_embeddings = torch.cat([state_embeddings1, state_embeddings2, state_embeddings3, state_embeddings4,
                                      state_embeddings5, state_embeddings6,state_embeddings7, state_embeddings8,
                                      state_embeddings9], dim=1)


        # Step 5: stack return, stage and previous embeddings
        stacked_inputs = torch.cat((return_embeddings, state_embeddings), dim=1)  # mind the order
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)  # mind the order
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization

        # 1 if can be attended to, 0 if not
        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 6: predict the bitrate for next chunk
        logits_used = logits[:, -1:]
        logits_used = logits_used.to(dtype=self._action_head_input_dtype())
        action_pred = self.action_head(logits_used)
        action_pred = mask_action_logits(action_pred, phase)
        action_pred1 = action_pred.reshape(-1)
        bitrate, _ = self._sample(action_pred1)

        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = (bitrate + 1) / self.action_levels
        action_embeddings = self.embed_action(action_tensor) + time_embeddings
        
        # update deques
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings) 
        self.actions_dq.append(action_embeddings)

        return action_pred,bitrate
    
    def clear_dq(self):
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))

# When given weights, `random.choices()` selects elements from the sequence based on their relative probabilities, 
# with higher weights making an element more likely to be chosen.
    # def _sample(self, logits):
    #     pi = F.softmax(logits, 0).cpu().detach().numpy()
    #     idx = random.choices(np.arange(pi.size), pi)[0]
    #     lgprob = np.log(pi[idx])
    #     return idx, lgprob

    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().detach().numpy()
        idx = np.argmax(pi)  # Select the index with the highest probability
        lgprob = np.log(pi[idx])
        return idx, lgprob

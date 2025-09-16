import torch
import torch.optim as optim
import joblib
import numpy as np
from pathlib import Path
from . import schemas
import gym
from gym import spaces
from collections import namedtuple

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"

class MLP_C(torch.nn.Module):
    def __init__(self, input_dim=47):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 1024), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1200)
        )
    def forward(self, x):
        return self.net(x)

class MLP_S(torch.nn.Module):
    def __init__(self, input_dim=46):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1200)
        )
    def forward(self, x):
        return self.net(x)

class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class DesignService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Design service is using device: {self.device}")

        self.model_c = MLP_C().to(self.device)
        self.model_c.load_state_dict(torch.load(ASSETS_DIR / "mlp_xyz_C.pt", map_location=self.device))
        self.model_c.eval()
        self.scaler_x_c = joblib.load(ASSETS_DIR / "scaler_X_C.pkl")
        self.scaler_y_c = joblib.load(ASSETS_DIR / "scaler_Y_C.pkl")
        print("Convex (C) models for environment loaded successfully.")
        
        self.model_s = MLP_S().to(self.device)
        self.model_s.load_state_dict(torch.load(ASSETS_DIR / "mlp_xyz_S.pt", map_location=self.device))
        self.model_s.eval()
        self.scaler_x_s = joblib.load(ASSETS_DIR / "scaler_X_S.pkl")
        self.scaler_y_s = joblib.load(ASSETS_DIR / "scaler_Y_S.pkl")
        print("Concave (S) models for environment loaded successfully.")


    def run_design_c(self, inputs: schemas.DesignInput) -> schemas.DesignOutput:
        target_warpage_mm = inputs.target_warpage_um / 1000.0
        fixed_params = [inputs.copper, inputs.substrate] + inputs.sbthk_vals + inputs.material_vals
        env = WarpageEnv(self.model_c, self.scaler_x_c, self.scaler_y_c, fixed_params, target_warpage=target_warpage_mm)

        input_dim = 47
        output_dim = int(np.prod(env.action_space.nvec))

        policy_net = DQN(input_dim, output_dim).to(self.device)
        target_net = DQN(input_dim, output_dim).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
        memory = PrioritizedReplayBuffer(50000)

        EPISODES = 1000
        BATCH_SIZE = 64
        GAMMA = 0.95
        EPS_START, EPS_END, EPS_DECAY = 1.0, 0.001, 0.998
        EPS_OSC_FREQ, EPS_OSC_AMPLITUDE = 300, 0.1
        
        best_reward = float('-inf')
        best_params_info = {}

        for episode in range(EPISODES):
            beta = min(1.0, 0.4 + episode * (1.0 - 0.4) / EPISODES)
            epsilon = max(EPS_END, EPS_START * (EPS_DECAY ** episode))
            epsilon += EPS_OSC_AMPLITUDE * np.sin(2 * np.pi * episode / EPS_OSC_FREQ)
            epsilon = float(np.clip(epsilon, EPS_END, 1.0))

            state = env.reset()

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = policy_net(state_tensor).detach().cpu().numpy()[0]
                best_action_index = int(np.argmax(q_values))
                action = np.unravel_index(best_action_index, env.action_space.nvec)

            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)

            if len(memory) >= BATCH_SIZE:
                batch, indices, weights = memory.sample(BATCH_SIZE, beta)
                states_td = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
                actions_idx_td = [np.ravel_multi_index(a, env.action_space.nvec) for a in batch.action]
                actions_td = torch.tensor(actions_idx_td, dtype=torch.int64, device=self.device).unsqueeze(1)
                rewards_td = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
                next_states_td = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
                dones_td = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

                current_q_td = policy_net(states_td).gather(1, actions_td).squeeze()
                next_actions_td = policy_net(next_states_td).argmax(1, keepdim=True)
                next_q_td = target_net(next_states_td).gather(1, next_actions_td).detach().squeeze()
                target_q_td = rewards_td + (1.0 - dones_td) * GAMMA * next_q_td
                td_errors = (current_q_td - target_q_td).abs().detach().cpu().numpy()
                memory.update_priorities(indices, td_errors + 1e-5)

                loss = ((current_q_td - target_q_td) ** 2) * torch.tensor(weights, device=self.device)
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if episode % 30 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if reward > best_reward:
                best_reward = reward
                best_params_info = {"reward": reward, "warpage": info["warpage"], "inputs": info["inputs"]}
        
        if not best_params_info:
            raise RuntimeError("DDQN training finished without finding any valid parameters.")

        final_inputs = best_params_info.get('inputs', np.zeros(47))
        best_params = schemas.BestParameters(
            tool_height=float(final_inputs[0]),
            magnet=int(final_inputs[1]),
            jig=float(final_inputs[2]),
            b1=int(final_inputs[4]),
            w1=int(final_inputs[5]),
        )

        return schemas.DesignOutput(
            achieved_warpage_um=best_params_info.get('warpage', float('nan')) * 1000.0,
            best_parameters=best_params
        )

    def run_design_s(self, inputs: schemas.DesignInput) -> schemas.DesignOutput:
        target_warpage_mm = inputs.target_warpage_um / 1000.0
        fixed_params = [inputs.copper, inputs.substrate] + inputs.sbthk_vals + inputs.material_vals
        env = WarpageEnvS(self.model_s, self.scaler_x_s, self.scaler_y_s, fixed_params, target_warpage=target_warpage_mm)

        input_dim = 46
        output_dim = int(np.prod(env.action_space.nvec))

        policy_net = DQN(input_dim, output_dim).to(self.device)
        target_net = DQN(input_dim, output_dim).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
        memory = PrioritizedReplayBuffer(50000)

        EPISODES = 1000
        BATCH_SIZE = 64
        GAMMA = 0.95
        EPS_START, EPS_END, EPS_DECAY = 1.0, 0.001, 0.998
        EPS_OSC_FREQ, EPS_OSC_AMPLITUDE = 300, 0.1
        
        best_reward = float('-inf')
        best_params_info = {}

        for episode in range(EPISODES):
            beta = min(1.0, 0.4 + episode * (1.0 - 0.4) / EPISODES)
            epsilon = max(EPS_END, EPS_START * (EPS_DECAY ** episode))
            epsilon += EPS_OSC_AMPLITUDE * np.sin(2 * np.pi * episode / EPS_OSC_FREQ)
            epsilon = float(np.clip(epsilon, EPS_END, 1.0))

            state = env.reset()

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = policy_net(state_tensor).detach().cpu().numpy()[0]
                best_action_index = int(np.argmax(q_values))
                action = np.unravel_index(best_action_index, env.action_space.nvec)

            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)

            if len(memory) >= BATCH_SIZE:
                batch, indices, weights = memory.sample(BATCH_SIZE, beta)
                states_td = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
                actions_idx_td = [np.ravel_multi_index(a, env.action_space.nvec) for a in batch.action]
                actions_td = torch.tensor(actions_idx_td, dtype=torch.int64, device=self.device).unsqueeze(1)
                rewards_td = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
                next_states_td = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
                dones_td = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

                current_q_td = policy_net(states_td).gather(1, actions_td).squeeze()
                next_actions_td = policy_net(next_states_td).argmax(1, keepdim=True)
                next_q_td = target_net(next_states_td).gather(1, next_actions_td).detach().squeeze()
                target_q_td = rewards_td + (1.0 - dones_td) * GAMMA * next_q_td
                td_errors = (current_q_td - target_q_td).abs().detach().cpu().numpy()
                memory.update_priorities(indices, td_errors + 1e-5)

                loss = ((current_q_td - target_q_td) ** 2) * torch.tensor(weights, device=self.device)
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if episode % 30 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if reward > best_reward:
                best_reward = reward
                best_params_info = {"reward": reward, "warpage": info["warpage"], "inputs": info["inputs"]}
        
        if not best_params_info:
            raise RuntimeError("DDQN training finished without finding any valid parameters.")

        final_inputs = best_params_info.get('inputs', np.zeros(46))
        best_params = schemas.BestParameters(
            magnet=int(final_inputs[0]),
            jig=float(final_inputs[1]),
            b1=int(final_inputs[3]),
            w1=int(final_inputs[4]),
        )

        return schemas.DesignOutput(
            achieved_warpage_um=best_params_info.get('warpage', float('nan')) * 1000.0,
            best_parameters=best_params
        )

class WarpageEnv(gym.Env):
    def __init__(self, model, scaler_X, scaler_y, fixed_params, target_warpage=0.025):
        super().__init__()
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.target = float(target_warpage)
        self.fixed = np.array(fixed_params, dtype=np.float32)

        self.magnet_options = list(range(10, 41))
        self.jig_options = [0.5, 1.0, 1.5, 2.0]
        B1_values = list(range(40, 61))
        W1_values = list(range(47, 68))
        raw_hole_combos = [(B1, W1) for B1 in B1_values for W1 in W1_values]
        SB = float(self.fixed[1])
        self.hole_combos = [(B1, W1) for (B1, W1) in raw_hole_combos if (B1 <= SB and W1 <= SB)]
        if not self.hole_combos:
            raise ValueError(f"No valid (B1, W1) combinations for Substrate size {SB}. Please adjust.")
        self.tool_heights = [round(i * 0.001, 3) for i in range(51)]

        self.action_space = spaces.MultiDiscrete([len(self.magnet_options), len(self.jig_options), len(self.hole_combos), len(self.tool_heights)])
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(47,), dtype=np.float32)

    def reset(self):
        return np.zeros(47, dtype=np.float32)

    def step(self, action):
        idx_mag, idx_jig, idx_hole, idx_tool = action
        variable_input = np.array([
            self.tool_heights[idx_tool], self.magnet_options[idx_mag], self.jig_options[idx_jig],
            self.fixed[0], self.hole_combos[idx_hole][0], self.hole_combos[idx_hole][1], self.fixed[1]
        ], dtype=np.float32)

        full_input = np.concatenate([variable_input, self.fixed[2:]]).reshape(1, -1)
        x_scaled = self.scaler_X.transform(full_input)

        with torch.no_grad():
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.model.net[0].weight.device)
            y_pred_scaled = self.model(x_tensor).detach().cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        z_def = y_pred[0][800:]
        warpage = float(np.max(z_def) - np.min(z_def))
        error = warpage - self.target
        abs_error = abs(error)

        if abs_error < 1e-4: reward = 45.0
        elif abs_error < 5e-4: reward = 20.0
        elif abs_error < 1e-3: reward = 1.0
        else: reward = -abs_error * 100.0
        if error < 0: reward += 5.0

        info = {"inputs": full_input.flatten(), "warpage": warpage}
        return x_scaled.flatten().astype(np.float32), reward, True, info

class WarpageEnvS(gym.Env):
    def __init__(self, model, scaler_X, scaler_y, fixed_params, target_warpage=-0.014):
        super().__init__()
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.target = float(target_warpage)
        self.fixed = np.array(fixed_params, dtype=np.float32)

        self.magnet_options = list(range(10, 41))
        self.jig_options = [0.5, 1.0, 1.5, 2.0]
        B1_values = list(range(40, 61))
        W1_values = list(range(47, 68))
        raw_hole_combos = [(B1, W1) for B1 in B1_values for W1 in W1_values]
        SB = float(self.fixed[1])
        self.hole_combos = [(B1, W1) for (B1, W1) in raw_hole_combos if (B1 <= SB and W1 <= SB)]
        if not self.hole_combos:
            raise ValueError(f"No valid (B1, W1) combinations for Substrate size {SB}. Please adjust.")

        self.action_space = spaces.MultiDiscrete([len(self.magnet_options), len(self.jig_options), len(self.hole_combos)])
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(46,), dtype=np.float32)

    def reset(self):
        return np.zeros(46, dtype=np.float32)

    def step(self, action):
        idx_mag, idx_jig, idx_hole = action
        variable_input = np.array([
            self.magnet_options[idx_mag], self.jig_options[idx_jig],
            self.fixed[0], self.hole_combos[idx_hole][0], self.hole_combos[idx_hole][1], self.fixed[1]
        ], dtype=np.float32)

        full_input = np.concatenate([variable_input, self.fixed[2:]]).reshape(1, -1)
        x_scaled = self.scaler_X.transform(full_input)

        with torch.no_grad():
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.model.net[0].weight.device)
            y_pred_scaled = self.model(x_tensor).detach().cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        z_def = y_pred[0][800:]
        warpage = -(float(np.max(z_def) - np.min(z_def))) # Note the negative sign for concave
        error = warpage - self.target
        abs_error = abs(error)

        if abs_error < 1e-4: reward = 45.0
        elif abs_error < 5e-4: reward = 20.0
        elif abs_error < 1e-3: reward = 1.0
        else: reward = -abs_error * 100.0
        if error < 0: reward += 5.0

        info = {"inputs": full_input.flatten(), "warpage": warpage}
        return x_scaled.flatten().astype(np.float32), reward, True, info


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if not self.buffer: raise ValueError("ReplayBuffer is empty")
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return Transition(*zip(*samples)), indices, weights.tolist()

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

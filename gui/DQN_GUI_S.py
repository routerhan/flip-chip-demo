import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from collections import namedtuple
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox, QDialog, QFormLayout, QScrollArea, QLineEdit, QFileDialog
)
from gym import spaces
import gym
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =======================
# 全域裝置設定
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# MLP 預測模型（47 → 1200）
# =======================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            

            nn.Linear(64, 128),
            nn.ReLU(),
           
            nn.Linear(128, 64),
            nn.ReLU(),
          

            nn.Linear(64, 1200)
        )

    def forward(self, x):
        return self.net(x)

ROOT = Path(__file__).resolve().parent

def load_model_and_scaler():
    """載入縮放器與 MLP 模型（47 → 1200）。"""
    scaler_X_path = ROOT / "scaler_X_S.pkl"
    scaler_y_path = ROOT / "scaler_Y_S.pkl"
    model_path = ROOT / "mlp_xyz_S.pt"

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    model = MLP(input_dim=46).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, scaler_X, scaler_y

# =======================
# Gym 環境（單步 contextual bandit）
# =======================
class WarpageEnv(gym.Env):
    def __init__(self, model, scaler_X, scaler_y, fixed_params, target_warpage=-0.014):
        super(WarpageEnv, self).__init__()
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.target = float(target_warpage)
        # fixed_params: [copper_ratio, SB] + 33*SBthk + 7*material = 42
        self.fixed = np.array(fixed_params, dtype=np.float32)

        # 1) 磁鐵數量：10~40（含）
        self.magnet_options = list(range(10, 41))  # 31 種

        # 2) Jig 厚度：0.5、1.0、1.5、2.0
        self.jig_options = [0.5, 1.0, 1.5, 2.0]    # 4 種

        # 3) 中心孔 (B1, W1)：依你最終版（註解說 20×20；實際上 range(40,61)/range(47,68) 會是 21×21）
        B1_values = list(range(40, 61))  # 40..60（實際 21 個）
        W1_values = list(range(47, 68))  # 47..67（實際 21 個）
        raw_hole_combos = [(B1, W1) for B1 in B1_values for W1 in W1_values]  # 441

        # 4) 幾何限制：SB 不得小於 B1 或 W1
        SB = float(self.fixed[1])
        self.hole_combos = [(B1, W1) for (B1, W1) in raw_hole_combos if (B1 <= SB and W1 <= SB)]
        if len(self.hole_combos) == 0:
            raise ValueError(f"[設定錯誤] 依 SB={SB} 套用限制後，(B1, W1) 無可用組合。請調整 SB 或 B1/W1 範圍。")


        # 動作空間（多維離散）
        self.action_space = spaces.MultiDiscrete([
            len(self.magnet_options),   # 31
            len(self.jig_options),      # 4
            len(self.hole_combos),      # <= 441（經 SB 過濾後）
        ])

        # 觀察空間：StandardScaler 後可能為負，設定寬鬆範圍
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(46,), dtype=np.float32)

    def reset(self):
        # 單步 bandit：固定回傳即可
        return np.zeros(46, dtype=np.float32)

    def step(self, action):
        idx_mag, idx_jig, idx_hole = action
        magnet = self.magnet_options[idx_mag]
        jig = self.jig_options[idx_jig]
        B1, W1 = self.hole_combos[idx_hole]

        # 組裝 47 維輸入
        variable_input = np.array([
            magnet,
            jig,
            self.fixed[0],   # copper_ratio
            B1,
            W1,
            self.fixed[1]    # substrate (SB)
        ], dtype=np.float32)

        full_input = np.concatenate([variable_input, self.fixed[2:]]).reshape(1, -1)
        x_scaled = self.scaler_X.transform(full_input)

        with torch.no_grad():
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
            y_pred_scaled = self.model(x_tensor).detach().cpu().numpy()
            y_pred_scaled = np.clip(y_pred_scaled, -5, 5)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # 後 400 維為 Z_def
        z_def = y_pred[0][800:]
        warpage = -(float(np.max(z_def) - np.min(z_def)))
        error = warpage - self.target
        abs_error = abs(error)

        # 獎勵（對齊你的最終版）
        if abs_error == 0:
            reward = 50.0
        elif abs_error < 1e-4:
            reward = 45.0
        elif abs_error < 5e-4:
            reward = 20.0
        elif abs_error < 1e-3:
            reward = 1.0
        else:
            reward = -abs_error * 100.0
        if error < 0:
            reward += 5.0

        info = {"inputs": full_input.flatten(), "warpage": warpage}
        # 單步環境：done=True
        return x_scaled.flatten().astype(np.float32), reward, True, info

# =======================
# Prioritized Replay Buffer（含 done）
# =======================
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
        if len(self.buffer) == 0:
            raise ValueError("ReplayBuffer is empty")
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        batch = Transition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# =======================
# DDQN 網路
# =======================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# =======================
# 訓練主程式（整合最終版邏輯，接受 GUI 參數）
# =======================
def train_ddqn(substrate, copper_ratio, sbthk, material, target_warpage=0.025):
    # --- 檔案載入 ---
    model, scaler_X, scaler_y = load_model_and_scaler()

    # 固定參數排列： [copper_ratio, SB] + SBthk(33) + material(7) ＝ 42
    fixed_params = [copper_ratio, substrate] + sbthk + material

    env = WarpageEnv(model, scaler_X, scaler_y, fixed_params, target_warpage=target_warpage)

    input_dim = 46
    output_dim = int(np.prod(env.action_space.nvec))

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = PrioritizedReplayBuffer(50000)

    # 超參數：對齊你的最終版
    EPS_START, EPS_END, EPS_DECAY = 1.0, 0.001, 0.998
    EPS_OSC_FREQ, EPS_OSC_AMPLITUDE = 300, 0.1
    GAMMA, BATCH_SIZE = 0.95, 64
    EPISODES = 5000
    EARLY_STOP_START, EARLY_STOP_PATIENCE = 2000, 500

    no_improve_count = 0
    reward_history = []
    best_reward = float('-inf')
    best_params = {}

    def select_action(state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor).detach().cpu().numpy()[0]
        best_action_index = int(np.argmax(q_values))
        return np.unravel_index(best_action_index, env.action_space.nvec)

    def compute_td_error(batch):
        states = torch.tensor(batch.state, dtype=torch.float32, device=device)
        actions_index = [np.ravel_multi_index(a, env.action_space.nvec) for a in batch.action]
        actions = torch.tensor(actions_index, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=device)

        current_q = policy_net(states).gather(1, actions).squeeze()
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions).detach().squeeze()
        target_q = rewards + (1.0 - dones) * GAMMA * next_q  # 單步：done=1 → 只剩 rewards
        td_error = (current_q - target_q).abs().detach().cpu().numpy()
        return td_error

    def replay(beta):
        if len(memory) < BATCH_SIZE:
            return
        batch, indices, weights = memory.sample(BATCH_SIZE, beta)
        td_errors = compute_td_error(batch)
        memory.update_priorities(indices, td_errors + 1e-5)

        states = torch.tensor(batch.state, dtype=torch.float32, device=device)
        actions_index = [np.ravel_multi_index(a, env.action_space.nvec) for a in batch.action]
        actions = torch.tensor(actions_index, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=device)

        current_q = policy_net(states).gather(1, actions).squeeze()
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions).detach().squeeze()
        target_q = rewards + (1.0 - dones) * GAMMA * next_q

        loss = ((current_q - target_q) ** 2) * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # 與最終版一致
        optimizer.step()

    for episode in range(EPISODES):
        beta = min(1.0, 0.4 + episode * (1.0 - 0.4) / EPISODES)
        epsilon = max(EPS_END, EPS_START * (EPS_DECAY ** episode))
        epsilon += EPS_OSC_AMPLITUDE * np.sin(2 * np.pi * episode / EPS_OSC_FREQ)
        epsilon = float(np.clip(epsilon, EPS_END, 1.0))

        state = env.reset()
        action = select_action(state, epsilon)
        next_state, reward, done, info = env.step(action)

        memory.push(state, action, reward, next_state, done)
        replay(beta)
        state = next_state
        reward_history.append(reward)

        if episode % 30 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if reward > best_reward:
            best_reward = reward
            best_params = {"reward": reward, "warpage": info["warpage"] * 1000, "inputs": info["inputs"]}
            if episode >= EARLY_STOP_START:
                no_improve_count = 0
        else:
            if episode >= EARLY_STOP_START:
                no_improve_count += 1

        if episode >= EARLY_STOP_START and no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"🛑 Early stopping at episode {episode}")
            break

    # 儲存訓練曲線（不阻塞 GUI）
    try:
        if len(reward_history) > 0:
            plt.figure(figsize=(12, 5))
            plt.plot(reward_history, label='Reward per Episode', alpha=0.5)
            if len(reward_history) > 50:
                ma = np.convolve(reward_history, np.ones(50)/50, mode='valid')
                plt.plot(ma, label='Moving Average (50)', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('DDQN Training Reward Trend')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.close()
    except Exception:
        pass

    # 回傳 GUI 友善結果字串
    inputs = best_params.get('inputs', np.zeros(47))
    magnet = int(inputs[0]) if len(inputs) > 1 else 0
    jig = float(inputs[1]) if len(inputs) > 2 else 0.0
    b1 = int(inputs[3]) if len(inputs) > 4 else 0
    w1 = int(inputs[4]) if len(inputs) > 5 else 0
    warpage = best_params.get('warpage', float('nan'))

    out = []
    out.append("【最佳參數】")
    out.append(f" 磁體數量: {magnet}")
    out.append(f" Jig厚度: {jig:.3g} (mm)")
    out.append(f" Jig中心矩形孔: {b1}x{w1} (mm²)")
    out.append("")
    out.append(f" Warpage: {warpage:.2f} (μm)")

    return "\n".join(out)

# =======================
# SBthk 視窗（支援 Excel 動態匯入）
# =======================
class SBthkDialog(QDialog):
    def __init__(self, parent=None, init_values=None):
        super().__init__(parent)
        self.setWindowTitle("Substrate層數設定")
        self.sb_inputs = []

        self.presets = {
            "預設1": [
                0.015, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03,
                0.015, 0.03, 0.015, 0.03, 0.018, 1.24, 0.018, 0.03, 0.015, 0.03, 0.015,
                0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.018
            ]
        }

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("")
        self.preset_combo.addItems(self.presets.keys())
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("輸入層數和層厚"))
        layout.addWidget(self.preset_combo)

        self.scroll = QScrollArea()
        scroll_widget = QWidget()
        self.form_layout = QFormLayout()

        for i in range(1, 34):
            line = QLineEdit()
            line.setPlaceholderText("(mm)")
            self.sb_inputs.append(line)
            self.form_layout.addRow(f"第{i}層", line)

        scroll_widget.setLayout(self.form_layout)
        self.scroll.setWidget(scroll_widget)
        self.scroll.setWidgetResizable(True)

        self.import_btn = QPushButton("從 Excel 匯入")
        self.import_btn.clicked.connect(self.import_from_excel)
        layout.addWidget(self.import_btn)

        if init_values:
            for i, val in enumerate(init_values):
                if i < len(self.sb_inputs):
                    self.sb_inputs[i].setText(str(val))

        self.confirm_btn = QPushButton("確認")
        self.confirm_btn.clicked.connect(self.accept)

        layout.addWidget(self.scroll)
        layout.addWidget(self.confirm_btn)
        self.setLayout(layout)

    def apply_preset(self):
        selected = self.preset_combo.currentText()
        if selected in self.presets:
            values = self.presets[selected]
            for i, val in enumerate(values):
                if i < len(self.sb_inputs):
                    self.sb_inputs[i].setText(str(val))

    def get_values(self):
        vals = []
        for line in self.sb_inputs:
            txt = line.text().strip()
            if txt == "":
                continue
            vals.append(float(txt))
        return vals

    def import_from_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇 Excel 檔案", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            try:
                df = pd.read_excel(file_path, header=None)
                values = df.iloc[1].tolist()
                if len(values) < 33:
                    values += [0.0] * (33 - len(values))
                elif len(values) > 33:
                    values = values[:33]

                for line in self.sb_inputs:
                    line.deleteLater()
                self.sb_inputs.clear()

                form_layout = QFormLayout()
                for i in range(33):
                    line = QLineEdit()
                    line.setText(str(values[i]))
                    self.sb_inputs.append(line)
                    form_layout.addRow(f"第{i+1}層", line)

                new_scroll_widget = QWidget()
                new_scroll_widget.setLayout(form_layout)
                self.scroll.setWidget(new_scroll_widget)

            except Exception as e:
                QMessageBox.critical(self, "匯入錯誤", str(e))

# =======================
# 材料參數視窗（7 個參數）
# =======================
class MaterialDialog(QDialog):
    def __init__(self, parent=None, init_values=None):
        super().__init__(parent)
        self.setWindowTitle("Substrate材料參數設定")
        self.inputs = {}

        self.presets = {
            "PP": [14900, 0.43, 500, 0.43, 1.10e-5, 3.70e-5, 130],
            "PI": [3000, 0.34, 2500, 0.34, 3.5e-5, 5.0e-5, 360]
        }

        layout = QVBoxLayout()
        layout.addWidget(QLabel("輸入材料參數"))

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("")
        self.preset_combo.addItems(self.presets.keys())
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        layout.addWidget(self.preset_combo)

        self.import_btn = QPushButton("從 Excel 匯入")
        self.import_btn.clicked.connect(self.import_from_excel)
        layout.addWidget(self.import_btn)

        form_layout = QFormLayout()
        labels = [
            "楊氏模數1 (MPa)", "蒲松比1",
            "楊氏模數2 (MPa)", "蒲松比2", "CTE1(ppm/K)",
            "CTE2(ppm/K)", "Tg (°C)",
        ]
        self.keys = [
            "young_modulus_1", "poisson_ratio_1",
            "young_modulus_2", "poisson_ratio_2", "cte1",
            "cte2", "tg2",
        ]

        for label, key in zip(labels, self.keys):
            inp = QLineEdit()
            self.inputs[key] = inp
            form_layout.addRow(label, inp)

        if init_values:
            for i, val in enumerate(init_values):
                if i < len(self.keys):
                    self.inputs[self.keys[i]].setText(str(val))

        layout.addLayout(form_layout)

        self.confirm_btn = QPushButton("確認")
        self.confirm_btn.clicked.connect(self.accept)
        layout.addWidget(self.confirm_btn)
        self.setLayout(layout)

    def apply_preset(self):
        selected = self.preset_combo.currentText()
        if selected in self.presets:
            values = self.presets[selected]
            for i, val in enumerate(values):
                if i < len(self.keys):
                    self.inputs[self.keys[i]].setText(str(val))

    def get_values(self):
        return [float(self.inputs[k].text()) for k in self.keys]

    def import_from_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇 Excel 檔案", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            try:
                df = pd.read_excel(file_path, header=None)
                values = df.iloc[1].tolist()
                if len(values) != len(self.keys):
                    raise ValueError(f"Excel 檔案必須有 {len(self.keys)} 個數值")
                for i, key in enumerate(self.keys):
                    self.inputs[key].setText(str(values[i]))
            except Exception as e:
                QMessageBox.critical(self, "匯入錯誤", str(e))

# =======================
# 主視窗（GUI）
# =======================
class MainWindowDQNS(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI設計（DDQN 整合版）")

        self.substrate = 55
        self.copper_ratio = 100
        self.sbthk_values = []
        self.material_values = []

        layout = QVBoxLayout()

        substrate_layout = QHBoxLayout()
        substrate_layout.addWidget(QLabel("Substrate規格 (mm²):"))
        self.substrate_combo = QComboBox()
        for val in ["55", "65", "75", "85", "105"]:
            self.substrate_combo.addItem(f"{val}x{val}", userData=int(val))
        self.substrate_combo.currentIndexChanged.connect(
            lambda i: setattr(self, "substrate", self.substrate_combo.itemData(i))
        )
        substrate_layout.addWidget(self.substrate_combo)
        layout.addLayout(substrate_layout)

        copper_layout = QHBoxLayout()
        copper_layout.addWidget(QLabel("Copper Ratio (%):"))
        self.copper_combo = QComboBox()
        self.copper_combo.addItems(["100", "90", "85", "80", "75", "70"])
        self.copper_combo.setCurrentText("")
        self.copper_combo.currentTextChanged.connect(lambda val: setattr(self, "copper_ratio", int(val)))
        copper_layout.addWidget(self.copper_combo)
        layout.addLayout(copper_layout)

        self.sbthk_btn = QPushButton("Substrate層數設定")
        self.sbthk_btn.clicked.connect(self.open_sbthk_dialog)
        layout.addWidget(self.sbthk_btn)

        self.material_btn = QPushButton("Substrate材料參數設定")
        self.material_btn.clicked.connect(self.open_material_dialog)
        layout.addWidget(self.material_btn)

        warpage_layout = QHBoxLayout()
        warpage_layout.addWidget(QLabel("Target Warpage (μm):"))
        self.target_warpage_input = QLineEdit("")
        warpage_layout.addWidget(self.target_warpage_input)
        layout.addLayout(warpage_layout)

        self.train_btn = QPushButton("開始訓練")
        self.train_btn.clicked.connect(self.run_training)
        layout.addWidget(self.train_btn)

        self.status_label = QLabel("狀態：尚未訓練")
        layout.addWidget(self.status_label)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        layout.addWidget(self.result_box)
        self.setLayout(layout)

    def open_sbthk_dialog(self):
        dialog = SBthkDialog(parent=self, init_values=self.sbthk_values)
        if dialog.exec():
            self.sbthk_values = dialog.get_values()

    def open_material_dialog(self):
        dialog = MaterialDialog(parent=self, init_values=self.material_values)
        if dialog.exec():
            self.material_values = dialog.get_values()

    def run_training(self):
        # 基本檢查
        if len(self.sbthk_values) == 0:
            QMessageBox.warning(self, "輸入不足", "請先設定 Substrate 層數與層厚 (SBthk)")
            return
        if len(self.material_values) == 0:
            QMessageBox.warning(self, "輸入不足", "請先設定 Substrate 材料參數")
            return
        try:
            target_um = float(self.target_warpage_input.text()) if self.target_warpage_input.text().strip() != '' else 25.0
        except ValueError:
            QMessageBox.warning(self, "輸入錯誤", "請輸入有效的目標 warpage 數值 (μm)")
            return

        self.status_label.setText("狀態：訓練中...請稍候")
        self.repaint()
        try:
            result = train_ddqn(
                substrate=self.substrate,
                copper_ratio=self.copper_ratio,
                sbthk=self.sbthk_values,
                material=self.material_values,
                target_warpage=target_um/1000.0,  # 轉成 mm
            )
            self.status_label.setText("狀態：訓練完成")
            self.result_box.setPlainText(result)
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))
            self.status_label.setText("狀態：錯誤")

# =======================
# 進入點
# =======================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindowDQNS()
    window.show()
    sys.exit(app.exec())

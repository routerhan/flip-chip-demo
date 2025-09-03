import sys
import torch
import joblib
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFormLayout, QVBoxLayout, QMessageBox, QComboBox, QTextEdit, QDialog,
    QScrollArea, QFileDialog, QHBoxLayout, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata
import pandas as pd

# ---------------- 路徑 ----------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "mlp_xyz_C.pt"
SCALER_X_PATH = ROOT / "scaler_X_C.pkl"
SCALER_Y_PATH = ROOT / "scaler_Y_C.pkl"

# ---------------- 模型 ----------------
class MLP(torch.nn.Module):
    def __init__(self, input_dim=47):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(128, 1200)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- SBthk 視窗 ----------------
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

        scroll = QScrollArea()
        scroll_widget = QWidget()
        form_layout = QFormLayout()

        for i in range(1, 34):
            line = QLineEdit()
            line.setPlaceholderText("(mm)")
            self.sb_inputs.append(line)
            form_layout.addRow(f"第{i}層", line)

        scroll_widget.setLayout(form_layout)
        scroll.setWidget(scroll_widget)
        self.scroll = scroll
        scroll.setWidgetResizable(True)

        self.import_btn = QPushButton("從 Excel 匯入")
        self.import_btn.clicked.connect(self.import_from_excel)
        layout.addWidget(self.import_btn)

        if init_values:
            for i, val in enumerate(init_values):
                self.sb_inputs[i].setText(str(val))

        self.confirm_btn = QPushButton("確認")
        self.confirm_btn.clicked.connect(self.accept)

        layout.addWidget(scroll)
        layout.addWidget(self.confirm_btn)
        self.setLayout(layout)

    def apply_preset(self):
        selected = self.preset_combo.currentText()
        if selected in self.presets:
            values = self.presets[selected]
            for i, val in enumerate(values):
                self.sb_inputs[i].setText(str(val))

    def get_values(self):
        return [float(line.text()) if line.text() != "" else 0.0 for line in self.sb_inputs]

    def import_from_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇 Excel 檔案", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            try:
                df = pd.read_excel(file_path, header=None)
                # 取得第 2 行數值（index=1）
                values = df.iloc[1].tolist()
                # 補 0 或截斷到 33 個
                if len(values) < 33:
                    values += [0.0] * (33 - len(values))
                elif len(values) > 33:
                    values = values[:33]

                # 重新建立欄位
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

# ---------------- 材料參數視窗 ----------------
class MaterialDialog(QDialog):
    def __init__(self, parent=None, init_values=None):
        super().__init__(parent)
        self.setWindowTitle("Substrate材料參數設定")
        self.inputs = {}

        self.presets = {
            "PP": [
                14900, 0.43, 500, 0.43,
                1.10e-5, 3.70e-5, 130
            ],
            "PI": [
                3000, 0.34, 2500, 0.34,
                3.5e-5, 5.0e-5, 360
            ]
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
                self.inputs[self.keys[i]].setText(str(val))

    def get_values(self):
        return [float(self.inputs[k].text()) if self.inputs[k].text() != "" else 0.0 for k in self.keys]

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

# ---------------- 主視窗 ----------------
class MainWindowMLPC(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BGA-AI")
        self.setMinimumWidth(360)

        # 模型與 scaler
        self.model = MLP()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        self.model.eval()
        self.scaler_X = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)

        # 預設值（避免尚未開視窗就預測時維度錯誤）
        self.sbthk_vals = [0.0] * 33
        self.material_vals = [0.0] * 7

        # --- 固定選單 ---
        self.substrate_box = QComboBox()
        self.substrate_box.addItems(["55x55", "65x65", "75x75", "85x85", "105x105"])

        self.copper_box = QComboBox()
        self.copper_box.addItems(["100", "90", "85", "80", "75", "70"])

        # --- 改為數值輸入 ---
        # 磁鐵數量（整數）
        self.magnet_input = QSpinBox()
        self.magnet_input.setRange(0, 100)
        self.magnet_input.setValue(10)

        # Jig 厚度（浮點）
        self.jig_input = QDoubleSpinBox()
        self.jig_input.setRange(0.0, 10.0)
        self.jig_input.setDecimals(1)
        self.jig_input.setSingleStep(0.1)
        self.jig_input.setValue(1.0)

        # Jig 中心矩形孔（口 × 口）
        self.b1_input = QSpinBox()
        self.b1_input.setRange(1, 2000)
        self.b1_input.setValue(40)
        self.w1_input = QSpinBox()
        self.w1_input.setRange(1, 2000)
        self.w1_input.setValue(47)
        hole_row = QWidget()
        hole_layout = QHBoxLayout(hole_row)
        hole_layout.setContentsMargins(0, 0, 0, 0)
        hole_layout.addWidget(self.b1_input)
        x_label = QLabel("×")
        x_label.setAlignment(Qt.AlignCenter)
        x_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        hole_layout.addWidget(x_label)

        hole_layout.addWidget(self.w1_input)

        # Tool 高度（浮點）
        self.tool_input = QDoubleSpinBox()
        self.tool_input.setRange(0.0, 0.05)
        self.tool_input.setDecimals(3)
        self.tool_input.setSingleStep(0.001)
        self.tool_input.setValue(0.0)

        # SBthk & Material 視窗
        self.btn_sbthk = QPushButton("Substrate層數設定")
        self.btn_sbthk.clicked.connect(self.open_sbthk)

        self.btn_material = QPushButton("Substrate材料參數設定")
        self.btn_material.clicked.connect(self.open_material)

        # 執行按鈕與結果
        self.predict_button = QPushButton("確定")
        self.predict_button.clicked.connect(self.predict)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        # 版面
        form_layout = QFormLayout()
        form_layout.addRow("Substrate規格(mm²):", self.substrate_box)
        form_layout.addRow("Copper Ratio(%):", self.copper_box)
        form_layout.addRow("磁鐵數量:", self.magnet_input)
        form_layout.addRow("Jig厚度(mm):", self.jig_input)
        form_layout.addRow("Jig中心矩形孔(mm):", hole_row)  # 口 × 口
        form_layout.addRow("Tool高度(mm):", self.tool_input)
        form_layout.addRow(self.btn_sbthk)
        form_layout.addRow(self.btn_material)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_text)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_sbthk(self):
        dialog = SBthkDialog(self, init_values=self.sbthk_vals)
        if dialog.exec():
            try:
                vals = dialog.get_values()
                if len(vals) != 33:
                    raise ValueError("層數需要 33 個數值")
                self.sbthk_vals = vals
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"請輸入有效的數值：{e}")

    def open_material(self):
        dialog = MaterialDialog(self, init_values=self.material_vals)
        if dialog.exec():
            try:
                vals = dialog.get_values()
                if len(vals) != 7:
                    raise ValueError("材料參數需要 7 個數值")
                self.material_vals = vals
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"請輸入有效的材料參數：{e}")

    def predict(self):
        try:
            # 讀取輸入
            substrate = int(self.substrate_box.currentText().split("x")[0])
            copper = int(self.copper_box.currentText())
            magnet = int(self.magnet_input.value())
            jig = float(self.jig_input.value())
            B1 = int(self.b1_input.value())
            W1 = int(self.w1_input.value())
            tool_height = float(self.tool_input.value())

            # 基本特徵順序需與訓練時一致
            basic = [tool_height, magnet, jig, copper, B1, W1, substrate]

            # 檢查維度
            if len(self.sbthk_vals) != 33 or len(self.material_vals) != 7:
                raise ValueError("請先完成 Substrate 層數與材料參數設定（需 33 + 7 個數值）。")

            X = np.array([basic + self.sbthk_vals + self.material_vals], dtype=float)
            X_scaled = self.scaler_X.transform(X)

            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                y_pred_scaled = self.model(X_tensor).cpu().numpy()
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0]

            X_def = y_pred[0:400]
            Y_def = y_pred[400:800]
            Z_def = y_pred[800:1200]

            # 計算 warpage (mm -> μm)
            warpage = (float(np.max(Z_def)) - float(np.min(Z_def))) * 1000.0

            # 顯示輸入與結果
            input_summary = (
                "【輸入參數】\n"
                f" Substrate規格: {substrate} x {substrate} (mm²)\n"
                f" Copper Ratio: {copper} (%)\n"
                f" 磁鐵數量: {magnet}\n"
                f" Jig厚度: {jig:.3f} (mm)\n"
                f" Jig中心矩形孔: {B1} × {W1} (mm²)\n"
                f" Tool高度: {tool_height:.4f} (mm)\n"
            )
            result_summary = (
                "\n【預測結果】\n"
                f" 預測 Warpage：{warpage:.2f} (μm)\n"
            )
            self.result_text.setPlainText(input_summary + result_summary)

            # ====== 進行插值處理並繪圖 ======
            grid_x, grid_y = np.mgrid[
                np.min(X_def):np.max(X_def):100j,
                np.min(Y_def):np.max(Y_def):100j
            ]
            grid_z = griddata(
                points=(X_def, Y_def),
                values=Z_def,
                xi=(grid_x, grid_y),
                method='cubic'
            )

            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="turbo", edgecolor='none', antialiased=True)

            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            

            # 等比例座標
            def set_axes_equal(ax_):
                x_limits = ax_.get_xlim3d()
                y_limits = ax_.get_ylim3d()
                z_limits = ax_.get_zlim3d()
                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])
                max_range = max(x_range, y_range, z_range)
                x_middle = np.mean(x_limits)
                y_middle = np.mean(y_limits)
                z_middle = np.mean(z_limits)
                ax_.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
                ax_.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
                ax_.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

            set_axes_equal(ax)

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=25, label="Z (mm)")
            fig.canvas.manager.window.move(950, 200)   # 讓圖在主視窗右邊
            fig.canvas.manager.window.resize(600, 500) # 調整大小避免擋住
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindowMLPC()
    window.show()
    sys.exit(app.exec())

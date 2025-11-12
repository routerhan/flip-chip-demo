import torch
import joblib
import numpy as np
from pathlib import Path
from . import schemas
from scipy.interpolate import griddata

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

class PredictionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Prediction service is using device: {self.device}")

        self.model_c = MLP_C().to(self.device)
        self.model_c.load_state_dict(torch.load(ASSETS_DIR / "mlp_xyz_C.pt", map_location=self.device))
        self.model_c.eval()
        self.scaler_x_c = joblib.load(ASSETS_DIR / "scaler_X_C.pkl")
        self.scaler_y_c = joblib.load(ASSETS_DIR / "scaler_Y_C.pkl")
        print("Convex (C) models loaded successfully.")

        self.model_s = MLP_S().to(self.device)
        self.model_s.load_state_dict(torch.load(ASSETS_DIR / "mlp_xyz_S.pt", map_location=self.device))
        self.model_s.eval()
        self.scaler_x_s = joblib.load(ASSETS_DIR / "scaler_X_S.pkl")
        self.scaler_y_s = joblib.load(ASSETS_DIR / "scaler_Y_S.pkl")
        print("Concave (S) models loaded successfully.")

    def run_prediction_c(self, inputs: schemas.PredictionInputC) -> schemas.PredictionOutput:
        basic_features = [
            inputs.tool_height, inputs.magnet, inputs.jig, inputs.copper,
            inputs.b1, inputs.w1, inputs.substrate
        ]
        X = np.array([basic_features + inputs.sbthk_vals + inputs.material_vals], dtype=float)
        X_scaled = self.scaler_x_c.transform(X)
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            y_pred_scaled = self.model_c(X_tensor).cpu().numpy()
            y_pred = self.scaler_y_c.inverse_transform(y_pred_scaled)[0]

        x_def, y_def, z_def = y_pred[0:400], y_pred[400:800], y_pred[800:1200]
        warpage_um = (float(np.max(z_def)) - float(np.min(z_def))) * 1000.0

        input_summary = {
            "Substrate規格": f"{inputs.substrate}x{inputs.substrate} (mm²)",
            "Copper Ratio": f"{inputs.copper} (%)",
            "磁鐵數量": inputs.magnet,
            "Jig厚度": f"{inputs.jig:.3f} (mm)",
            "Jig中心矩形孔": f"{inputs.b1}x{inputs.w1} (mm²)",
            "Tool高度": f"{inputs.tool_height:.4f} (mm)"
        }

        grid_x, grid_y = np.mgrid[np.min(x_def):np.max(x_def):100j, np.min(y_def):np.max(y_def):100j]
        grid_z = griddata(points=(x_def, y_def), values=z_def, xi=(grid_x, grid_y), method='cubic')
        grid_z_list = np.where(np.isnan(grid_z), None, grid_z).tolist()

        plot_data = schemas.PlotData(x=grid_x[:, 0].tolist(), y=grid_y[0, :].tolist(), z=grid_z_list)

        return schemas.PredictionOutput(warpage_um=warpage_um, input_summary=input_summary, plot_data=plot_data)

    def run_prediction_s(self, inputs: schemas.PredictionInputS) -> schemas.PredictionOutput:
        basic_features = [inputs.magnet, inputs.jig, inputs.copper, inputs.b1, inputs.w1, inputs.substrate]
        X = np.array([basic_features + inputs.sbthk_vals + inputs.material_vals], dtype=float)
        X_scaled = self.scaler_x_s.transform(X)
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            y_pred_scaled = self.model_s(X_tensor).cpu().numpy()
            y_pred = self.scaler_y_s.inverse_transform(y_pred_scaled)[0]

        x_def, y_def, z_def = y_pred[0:400], y_pred[400:800], y_pred[800:1200]
        warpage_um = -(float(np.max(z_def)) - float(np.min(z_def))) * 1000.0

        input_summary = {
            "Substrate規格": f"{inputs.substrate}x{inputs.substrate} (mm²)",
            "Copper Ratio": f"{inputs.copper} (%)",
            "磁鐵數量": inputs.magnet,
            "Jig厚度": f"{inputs.jig:.3f} (mm)",
            "Jig中心矩形孔": f"{inputs.b1}x{inputs.w1} (mm²)",
        }

        grid_x, grid_y = np.mgrid[np.min(x_def):np.max(x_def):100j, np.min(y_def):np.max(y_def):100j]
        grid_z = griddata(points=(x_def, y_def), values=z_def, xi=(grid_x, grid_y), method='cubic')
        grid_z_list = np.where(np.isnan(grid_z), None, grid_z).tolist()

        plot_data = schemas.PlotData(x=grid_x[:, 0].tolist(), y=grid_y[0, :].tolist(), z=grid_z_list)

        return schemas.PredictionOutput(warpage_um=warpage_um, input_summary=input_summary, plot_data=plot_data)

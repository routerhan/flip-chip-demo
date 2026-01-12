# BGA-AI: Warpage Prediction and Design (GUI)

## Overview

A desktop application for predicting and optimizing BGA (Ball Grid Array) warpage during the manufacturing process. Built with PySide6, it provides an intuitive graphical interface for:

- **Warpage Prediction**: Predicts 3D warpage surface for both convex and concave scenarios.
- **AI Design**: Finds optimal process parameters to achieve target warpage using Reinforcement Learning (DDQN).

## Project Structure

```
.
├── gui/                        # GUI application
│   ├── Main.py                 # Main entry point
│   ├── MLP_GUI_C.py            # Convex prediction GUI
│   ├── MLP_GUI_S.py            # Concave prediction GUI
│   ├── DQN_GUI_C.py            # Convex AI design GUI
│   ├── DQN_GUI_S.py            # Concave AI design GUI
│   ├── mlp_xyz_C.pt            # Convex prediction model
│   ├── mlp_xyz_S.pt            # Concave prediction model
│   ├── scaler_*.pkl            # Data scalers
│   ├── Material Parameters.xlsx
│   └── Substrate Layers.xlsx
└── requirements.txt            # Python dependencies
```

## Setup and Installation

### Prerequisites

- [pyenv](https://github.com/pyenv/pyenv) with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugin
- An NVIDIA GPU with CUDA is recommended for best performance, but CPU works too

### Step 1: Create Virtual Environment

```bash
# Install Python 3.12.0 if not already installed
pyenv install 3.12.0

# Create and activate virtual environment
pyenv virtualenv 3.12.0 bga-ai
pyenv activate bga-ai
```

### Step 2: Install PyTorch

Install PyTorch matching your system's hardware. Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) for the correct command.

**For CPU-only systems:**
```bash
pip install torch torchvision torchaudio
```

**For systems with NVIDIA GPU (example for CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

## How to Run the GUI

Navigate to the `gui/` directory and run the main application:

```bash
cd gui
python Main.py
```

## How to Use

The application window has 4 tabs:

| Tab | Function |
|-----|----------|
| **翹曲預測(凸)** | Convex warpage prediction |
| **翹曲預測(凹)** | Concave warpage prediction |
| **AI設計(凸)** | AI-based convex design optimization |
| **AI設計(凹)** | AI-based concave design optimization |

### Warpage Prediction (MLP)

1. Select either the **Convex** or **Concave** prediction tab.
2. Enter the required process parameters in the input fields.
3. Click the **Predict** button.
4. View the predicted warpage value and 3D surface plot.

### AI Design (DQN)

1. Select either the **Convex** or **Concave** design tab.
2. Set your **target warpage value**.
3. Configure the parameter constraints if needed.
4. Click the **Start** button to begin the optimization.
5. The AI will find optimal parameters to achieve your target warpage.

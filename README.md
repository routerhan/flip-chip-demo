1. The "requirements.txt" is availible.
2. Perform this case with "python gui/Main.py"
# BGA-AI: Warpage Prediction and Design

## Overview

This project provides a suite of tools for predicting and optimizing against BGA (Ball Grid Array) warpage during the manufacturing process. It has been migrated from a PySide6 desktop application to a modern web architecture, featuring:

-   A **FastAPI Backend** that serves pre-trained machine learning models for warpage prediction.
-   A **Vanilla JS & HTML Frontend** that provides an interactive user interface for making predictions and visualizing the results in 3D.

The core functionalities include:
-   **Warpage Prediction**: Predicts the 3D warpage surface for both convex and concave scenarios based on a set of process parameters.
-   **AI Design (WIP)**: A future module to find optimal process parameters to achieve a target warpage value using Reinforcement Learning (DDQN).

## Project Structure

```
.
├── assets/             # Stores all model (.pt) and scaler (.pkl) files
├── backend/            # Contains the FastAPI application
│   ├── main.py         # API endpoints and application setup
│   ├── prediction_service.py # Core logic for loading models and running predictions
│   └── schemas.py      # Pydantic models for API data validation
├── frontend/           # Contains the user-facing web application
│   └── index.html      # The single-page application UI
├── gui/                # (Legacy) Original PySide6 desktop application code
└── requirements.txt    # Python dependencies for the project
```

## Setup and Installation

### Prerequisites
-   [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management.
-   An NVIDIA GPU with CUDA is recommended for best performance, but a CPU will also work.

### Step 1: Create Conda Environment
Create and activate a new Conda environment. Python 3.9 is recommended.
```bash
conda create -n bga-ai python=3.9
conda activate bga-ai
```

### Step 2: Install PyTorch
For optimal performance, install PyTorch separately using Conda, matching your system's hardware. Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to find the correct command for your setup.

**For CPU-only systems:**
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

**For systems with NVIDIA GPU (example for CUDA 11.8):**
```bash
conda install pytorch torchvision torchaudio pytorch-cudatoolkit=11.8 -c pytorch -c nvidia
```

### Step 3: Install Other Dependencies
Install all other required Python packages using `pip` and the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## How to Run

The application consists of two main parts: the backend server and the frontend interface. They must be run separately.

### 1. Start the Backend API Server
Navigate to the project's root directory in your terminal and run the following command to start the FastAPI server:
```bash
uvicorn backend.main:app --reload
```
-   The server will be available at `http://127.0.0.1:8000`.
-   The `--reload` flag enables hot-reloading, which automatically restarts the server when you make changes to the code.

### 2. Open the Frontend Interface
Navigate to the `frontend/` directory in your file explorer and open the `index.html` file with a modern web browser (e.g., Chrome, Firefox, Edge).

## How to Use

### Using the Frontend Interface
The web interface is the primary way to interact with the prediction models.

1.  **Select Prediction Type**: Use the tabs at the top to choose between "Convex Prediction" and "Concave Prediction".
2.  **Enter Parameters**: Fill in all the required process parameters in the form on the left. The form will adapt based on the selected prediction type.
3.  **Start Prediction**: Click the "Start Prediction" button.
4.  **View Results**:
    -   **Prediction Result**: The panel on the right will display a summary of your input parameters and the final predicted warpage value in micrometers (μm).
    -   **3D Surface Plot**: A fully interactive 3D plot of the predicted warpage surface will be rendered. You can rotate, pan, and zoom to inspect the result in detail.

### Using the API Directly (for Developers)

The API provides two main endpoints for prediction. You can explore them interactively via the auto-generated Swagger UI documentation.

-   **API Docs**: `http://127.0.0.1:8000/docs`

Here are `curl` examples for calling the endpoints directly:

#### Convex Prediction
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/convex' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "tool_height": 0, "magnet": 10, "jig": 1, "copper": 100, "b1": 40, "w1": 47, "substrate": 55,
  "sbthk_vals": [0.015, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.018, 1.24, 0.018, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.018],
  "material_vals": [14900, 0.43, 500, 0.43, 1.1e-05, 3.7e-05, 130]
}'
```

#### Concave Prediction
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/concave' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "magnet": 10, "jig": 1, "copper": 100, "b1": 40, "w1": 47, "substrate": 55,
  "sbthk_vals": [0.015, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.018, 1.24, 0.018, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.03, 0.015, 0.018],
  "material_vals": [14900, 0.43, 500, 0.43, 1.1e-05, 3.7e-05, 130]
}'
```

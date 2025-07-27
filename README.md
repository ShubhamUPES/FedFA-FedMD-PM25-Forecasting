# FedFA-FedMD-PM25-Forecasting

## 📆 Project Overview

A federated learning framework for PM2.5 forecasting across Indian cities using two collaborative strategies:

* **FedFA**: Federated Fully Asynchronous (a novel aggregation method designed by the author)
* **FedMD**: Federated Model Distillation

This project enables decentralized PM2.5 prediction using private city-wise datasets while preserving data privacy and improving generalization through collaborative training.

## ✨ Features

* City-wise dataset preprocessing (Delhi, Bengaluru, Hyderabad)
* Multiple client support with different architectures (LSTM, TCN, TST)
* Implementation of a novel **FedFA** strategy for asynchronous aggregation
* Centralized and federated evaluation modes
* Visualization, anomaly detection, and per-round logging

## 🚀 Getting Started

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/ShubhamUPES/FedFA-FedMD-PM25-Forecasting.git
cd FedFA-FedMD-PM25-Forecasting
```

## 📂 Project Structure

```
FedFA-FedMD-PM25-Forecasting/
├── clients/              # Client scripts for LSTM, TCN, TST models
├── utils/                # Utilities: model, data, distillation
├── server/               # Federated server logic
├── centralized/          # Centralized training and evaluation
├── experiments/          # Scripts for ablation and comparison
├── data/                 # City-wise CSV datasets
├── results/              # Output logs and metrics
├── notebooks/            # Jupyter notebooks for analysis
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## 🌐 Usage

### Prepare Data

```bash
python utils/data_utils.py --input data/raw --output data/processed
```

### Train Federated Model with FedFA

```bash
python server/fedfa_server.py --config config/fedfa.yaml
```

### Train Federated Model with FedMD

```bash
python server/fedmd_server.py --config config/fedmd.yaml
```

### Evaluate

```bash
python centralized/evaluate.py --model-path results/model.pth
```

## 📈 Results

Sample performance (RMSE/MAE) from experiments:

```
City       RMSE     MAE
Delhi      11.93    9.41
Hyderabad  10.56    8.72
Bengaluru  12.15    9.87
```

## 🧪 Configuration

Example hyperparameters:

```yaml
rounds: 10
batch_size: 32
epochs: 50
learning_rate: 0.0001
```

## 🔒 License

This project is licensed under the MIT License.

## 📚 Citation

The FedFA method is a novel contribution by the author and is intended to be submitted for publication. Please contact the author before citing.

---

For questions or contributions, open an issue or contact [Shubham Sahu](mailto:shubhamsahu.upes@gmail.com).

import os, time
import numpy as np
import pandas as pd
from utils.data_utils import load_and_preprocess_data
from utils.model_utils import build_lstm_model
from utils.distill_utils import compute_logits, save_logits, load_consensus_logits
from sklearn.metrics import mean_squared_error, r2_score

def run_client_lstm(rounds=3):
    client_name = "LSTM_Delhi"
    os.makedirs("logs", exist_ok=True)

    # Load private and public data
    X, y, scaler = load_and_preprocess_data("private_datasets/Delhi_PM25.csv")
    public_X, _, _ = load_and_preprocess_data("public_dataset/public_data.csv")

    # Reshape for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))
    public_X = public_X.reshape((public_X.shape[0], public_X.shape[1], 1))  # Fixed here

    # Build LSTM model
    model = build_lstm_model((X.shape[1], 1))

    # Setup logging
    metrics_path = f"logs/{client_name}_metrics.csv"
    pd.DataFrame(columns=["Round", "Loss", "R2", "Time"]).to_csv(metrics_path, index=False)

    for rnd in range(1, rounds + 1):
        start = time.time()

        # Train on private data
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        # Compute logits on public data
        logits = compute_logits(model, public_X)
        save_logits(client_name, rnd, logits)

        # Wait for server consensus logits
        while not os.path.exists(f"logs/consensus_logits_round{rnd}.pkl"):
            time.sleep(1)

        consensus = load_consensus_logits(rnd)

        # Train on consensus logits
        model.fit(public_X, consensus, epochs=5, batch_size=32, verbose=0)

        # Evaluate on private data
        preds = model.predict(X).flatten()
        loss = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        duration = time.time() - start

        # Save metrics
        pd.DataFrame([[rnd, loss, r2, duration]],
                     columns=["Round", "Loss", "R2", "Time"]).to_csv(metrics_path, mode='a', index=False, header=False)

if __name__ == "__main__":
    run_client_lstm()

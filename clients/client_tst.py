import os
import time
import numpy as np
import pandas as pd
from utils.data_utils import load_and_preprocess_data
from utils.model_utils import build_transformer_model
from utils.distill_utils import compute_logits
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import pickle

def run_client_transformer(max_updates=10):
    client_name = "TST_Hyderabad"
    os.makedirs("logs", exist_ok=True)

    X, y = load_and_preprocess_data("private_datasets/Hyderabad_PM25.csv")
    public_X, _ = load_and_preprocess_data("public_dataset/public_data.csv")

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_transformer_model(X.shape[1:])

    metrics_path = f"logs/{client_name}_metrics_fedfa.csv"
    if not os.path.exists(metrics_path):
        pd.DataFrame(columns=["UpdateID", "PrivateTime", "DistillTime", "WaitTime", "TotalTime", "Loss", "R2"]).to_csv(metrics_path, index=False)

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    update_id = 1
    while update_id <= max_updates:
        round_start = time.time()
        print(f"[{client_name}] Starting local training (Update {update_id})...")
        t1 = time.time()
        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[early_stop]
        )
        private_time = time.time() - t1

        print(f"[{client_name}] Computing public logits...")
        logits = compute_logits(model, public_X)
        with open(f"logs/{client_name}_logits_update.pkl", "wb") as f:
            pickle.dump(logits, f)

        wait_start = time.time()
        while not os.path.exists(f"logs/consensus_logits_latest.pkl"):
            time.sleep(0.5)
        wait_time = time.time() - wait_start

        with open(f"logs/consensus_logits_latest.pkl", "rb") as f:
            consensus = pickle.load(f)
        consensus = consensus.flatten()

        print(f"[{client_name}] Consensus shape: {consensus.shape}, Public_X shape: {public_X.shape}")
        t2 = time.time()
        model.fit(public_X, consensus, epochs=10, batch_size=32, verbose=0)
        distill_time = time.time() - t2

        preds = model.predict(X_val).flatten()
        loss = mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        total_time = time.time() - round_start

        pd.DataFrame([[update_id, private_time, distill_time, wait_time, total_time, loss, r2]],
                     columns=["UpdateID", "PrivateTime", "DistillTime", "WaitTime", "TotalTime", "Loss", "R2"]).to_csv(metrics_path, mode='a', index=False, header=False)

        print(f"[{client_name}] Update {update_id} complete | R2: {r2:.4f}, Loss: {loss:.4f}")
        update_id += 1

if __name__ == "__main__":
    run_client_transformer(max_updates=10)

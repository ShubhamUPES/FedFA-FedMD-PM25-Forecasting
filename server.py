import os
import time
import numpy as np
import pickle
import pandas as pd
from collections import deque

THRESHOLDS = [30, 60, 90, 120, 250, 500]
BUFFER_SIZE = 5  # Should be >= number of clients

def closest_threshold(value, thresholds=THRESHOLDS):
    return min(thresholds, key=lambda t: abs(value - t))

def hybrid_consensus(logits_list, public_pm25):
    consensus = []
    num_samples = len(public_pm25)
    for i in range(num_samples):
        ref_thresh = closest_threshold(public_pm25[i])
        weights = []
        for client_logits in logits_list:
            pred = client_logits[i][0]
            dist = abs(pred - ref_thresh)
            weight = 1 / (dist + 1e-8)
            weights.append(weight)
        weights = np.array(weights)
        weights /= np.sum(weights)
        weighted_avg = sum(w * client_logits[i][0] for w, client_logits in zip(weights, logits_list))
        consensus.append(weighted_avg)
    return np.array(consensus).reshape(-1, 1)

def run_server(max_updates=30, timeout_sec=600):
    os.makedirs("logs", exist_ok=True)
    consensus_log = "logs/server_metrics_fedfa.csv"
    if not os.path.exists(consensus_log):
        pd.DataFrame(columns=["UpdateID", "Client", "ConsensusTime", "BufferClients", "TotalTime"]).to_csv(consensus_log, index=False)

    client_names = ["LSTM_Delhi", "TCN_Bengaluru", "TST_Hyderabad"]
    public_df = pd.read_csv("public_dataset/public_data.csv")
    public_pm25 = public_df["PM2.5"].values

    buffer = deque(maxlen=BUFFER_SIZE)
    buffer_clients = deque(maxlen=BUFFER_SIZE)
    update_id = 1
    start_time = time.time()

    while update_id <= max_updates:
        found = False
        for client in client_names:
            path = f"logs/{client}_logits_update.pkl"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    logits = pickle.load(f)
                os.remove(path)
                if logits.ndim == 1:
                    logits = logits.reshape(-1, 1)
                elif logits.ndim > 2:
                    logits = logits.reshape(logits.shape[0], -1)
                buffer.append(logits)
                buffer_clients.append(client)
                found = True
                print(f"[Server] Received update from {client} (Update {update_id})")
                break  # Only process one update at a time

        if not found:
            time.sleep(0.5)
            continue

        if len(buffer) == 0:
            continue

        consensus_start = time.time()
        min_samples = min(l.shape[0] for l in buffer)
        trimmed_logits = [l[:min_samples] for l in buffer]
        consensus_logits = hybrid_consensus(trimmed_logits, public_pm25[:min_samples])
        consensus_time = time.time() - consensus_start

        # Save latest consensus for clients
        with open(f"logs/consensus_logits_latest.pkl", "wb") as f:
            pickle.dump(consensus_logits, f)

        # Log
        total_time = time.time() - start_time
        pd.DataFrame([[update_id, client, consensus_time, ",".join(buffer_clients), total_time]],
                     columns=["UpdateID", "Client", "ConsensusTime", "BufferClients", "TotalTime"]
        ).to_csv(consensus_log, mode='a', index=False, header=False)

        print(f"[Server] Update {update_id} complete | Buffer: {list(buffer_clients)} | Consensus time: {consensus_time:.4f}s")
        update_id += 1

    print("[Server] Max updates reached. Shutting down.")

if __name__ == "__main__":
    run_server(max_updates=30)

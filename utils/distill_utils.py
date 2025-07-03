import numpy as np
import pickle
import os
import time

def compute_logits(model, public_X):
    print(f"[Compute Logits] Input shape: {public_X.shape}")
    logits = model.predict(public_X, verbose=0)
    print(f"[Compute Logits] Output shape: {logits.shape}")
    return logits.reshape(-1, 1)

def save_logits(client_name, round_num, logits):
    path = f"logs/{client_name}_logits_round{round_num}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(logits, f)

def save_consensus_logits(round_num, logits):
    with open(f"logs/consensus_logits_round{round_num}.pkl", 'wb') as f:
        pickle.dump(logits, f)

def load_consensus_logits(round_num):
    path = f"logs/consensus_logits_round{round_num}.pkl"
    while not os.path.exists(path):
        print(f"[Load Consensus] Waiting for {path}...")
        time.sleep(5)
    with open(path, 'rb') as f:
        logits = pickle.load(f)
    print(f"[Load Consensus] Loaded shape: {logits.shape}")
    return logits

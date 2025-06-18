import numpy as np
import pickle

def compute_logits(model, public_X):
    return model.predict(public_X, verbose=0)

def save_logits(client_name, round_num, logits):
    path = f"logs/{client_name}_logits_round{round_num}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(logits, f)

def load_all_logits(client_names, round_num):
    all_logits = []
    for name in client_names:
        path = f"logs/{name}_logits_round{round_num}.pkl"
        with open(path, 'rb') as f:
            all_logits.append(pickle.load(f))
    return np.mean(np.stack(all_logits), axis=0)

def save_consensus_logits(round_num, logits):
    with open(f"logs/consensus_logits_round{round_num}.pkl", 'wb') as f:
        pickle.dump(logits, f)

def load_consensus_logits(round_num):
    with open(f"logs/consensus_logits_round{round_num}.pkl", 'rb') as f:
        return pickle.load(f)
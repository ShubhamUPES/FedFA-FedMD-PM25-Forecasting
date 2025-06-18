import time
import os

from utils.distill_utils import load_all_logits, save_consensus_logits

def run_server(rounds=3):
    client_names = ["LSTM_Delhi", "GRU_Bengaluru", "Transformer_Hyderabad"]
    for rnd in range(1, rounds + 1):
        print(f"[Server] Waiting for client logits of round {rnd}...")
        while not all([
            os.path.exists(f"logs/{name}_logits_round{rnd}.pkl") for name in client_names
        ]):
            time.sleep(2)

        consensus = load_all_logits(client_names, rnd)
        save_consensus_logits(rnd, consensus)
        print(f"[Server] Round {rnd} consensus logits saved.")

if __name__ == "__main__":
    run_server()
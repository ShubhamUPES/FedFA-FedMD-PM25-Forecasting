import subprocess
import multiprocessing
import time
import os

def run_client(script):
    subprocess.run(["python", script])

def run_server(server_script):
    subprocess.run(["python", server_script])

if __name__ == "__main__":
    max_updates = 30  # Total updates for the experiment (set as needed)

    client_scripts = [
        "clients/client_lstm.py",
        "clients/client_tcn.py",
        "clients/client_tst.py"
    ]

    server_script = "server.py"

    # Start server process
    server_proc = multiprocessing.Process(target=run_server, args=(server_script,))
    server_proc.start()
    time.sleep(2)  # Give server a moment to start

    # Start client processes
    processes = []
    for script in client_scripts:
        p = multiprocessing.Process(target=run_client, args=(script,))
        p.start()
        processes.append(p)

    # Wait for all clients to finish
    for p in processes:
        p.join()

    # Wait for server to finish
    server_proc.join()

    print("\n✅ FedFa run completed.\n")

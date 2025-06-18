import multiprocessing
import time
from server import run_server
from clients.client_lstm import run_client_lstm
from clients.client_gru import run_client_gru
from clients.client_transformer import run_client_transformer

if __name__ == '__main__':
    rounds = 3

    p1 = multiprocessing.Process(target=run_client_lstm, args=(rounds,))
    p2 = multiprocessing.Process(target=run_client_gru, args=(rounds,))
    p3 = multiprocessing.Process(target=run_client_transformer, args=(rounds,))
    p4 = multiprocessing.Process(target=run_server, args=(rounds,))

    p1.start()
    p2.start()
    p3.start()
    time.sleep(3)  # ensure clients start first
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
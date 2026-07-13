import glskf
from data_gen import set_all_seeds

def run_glskf(I, Omega, signal, test_mask, seed):
    rho = 10
    gamma = 20

    rows = []

    set_all_seeds(seed)
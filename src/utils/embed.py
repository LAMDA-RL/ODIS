import numpy as np

def polynomial_embed(v, length, v_min, v_max, exp=0.5):
    rel_v = (v - v_min) / (v_max - v_min)
    n_filled = round(rel_v**exp * length)
    embed_vec = np.zeros(length,)
    embed_vec[:n_filled] = 1
    return embed_vec

def binary_embed(v, length, v_max):
    assert 2**length - 1 >= v_max
    embed_vec = np.zeros(length,)
    bin_v = [int(item) for item in list(bin(v)[2:])]
    embed_vec[-len(bin_v):] = bin_v
    return embed_vec
    
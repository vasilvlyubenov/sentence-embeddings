import numpy as np

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(1) * np.linalg.norm(b))
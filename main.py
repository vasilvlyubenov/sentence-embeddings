import numpy as np
from utils import cosine

embeddings = {
    "love": np.array([1.0, 0.5]),
    "hate": np.array([-1.0, -0.5]),
    "coding": np.array([0.9, 0.7]),
    "pizza": np.array([0.2, 0.9]),
}

def sentence_vector(sentence):
    words = sentence.lower().split()
    vectors = []
    
    for word in words:
        if word in embeddings:
            vectors.append(embeddings[word])
        else:
            print(f"Unknown word {word}")
    
    if not vectors:
        return None
    
    return np.mean(vectors, axis=0)

print("=== Vector Similarity CLI ===")
print("Type 'exit' to quit the program\n")

while True:
    s1 = input("Sentence 1: ")
    
    if s1 == 'exit':
        break
    
    s2 = input("Sentence 2: ")
    
    if s2 == 'exit':
        break
    
    v1 = sentence_vector(s1)
    v2 = sentence_vector(s2)
    
    if v1 is None or v2 is None:
        print("Cannot compute\n")
        continue
    
    similarity = cosine(v1,v2)
    
    print(f"Similarity: {similarity:.4f}\n")
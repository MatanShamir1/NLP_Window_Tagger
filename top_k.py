import numpy as np


def cosine_sim(vector, array):
    """
    Returns the cosine similarity between two vectors.
    """
    # Compute the cosine similarity between u and v
    # cosine_sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # return cosine_sim
    dot_product = np.dot(array, vector)

    # Calculate the norm of the vector and the array
    vector_norm = np.linalg.norm(vector)
    array_norm = np.linalg.norm(array, axis=1)

    # Calculate the cosine similarity between the vector and the array
    cosine_similarity = dot_product / (array_norm * vector_norm)
    return cosine_similarity


def top_k_similar(word, vocab, vecs, k=5):
    word_idx = vocab.index(word)
    word_vec = vecs[word_idx].astype(dtype=float)
    vecs = vecs.astype(dtype=float)
    sim = cosine_sim(word_vec, vecs)
    top_k_idx = (sim.argsort(kind="stable")[-k - 1 :][::-1])[1:]
    distances = [sim[x] for x in top_k_idx]
    sim_words = [vocab[i] for i in top_k_idx]
    return sim_words, distances

def run():
    words_sim = ["dog", "england", "john", "explode", "office"]
    vecs = np.loadtxt("wordVectors.txt")
    with open("vocab.txt", "r", encoding="utf-8") as f:
        vocab = f.readlines()
        vocab = [word.strip() for word in vocab]
    word_to_top_k_sim = {word: top_k_similar(word, vocab, vecs) for word in words_sim}
    for word in words_sim:
        for i, word_sim in enumerate(word_to_top_k_sim[word][0]):
            print(f"{word} -> {word_sim}, distance: {word_to_top_k_sim[word][1][i].round(4)}")


if __name__ == "__main__":
    run()

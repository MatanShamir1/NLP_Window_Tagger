import numpy as np


def cos_sim(vector, array):
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


def most_similar(word, vocab, vecs, k=5):
    word_idx = vocab.index(word)
    word_vec = vecs[word_idx].astype(dtype=float)
    vecs = vecs.astype(dtype=float)
    # in our case u is word_vec, v is word_vecs
    sim = cos_sim(word_vec, vecs)
    # sort the arguments by similarity, take the k+1 last ones,
    # i.e highest, sort and return the last k, to ignore the word itself
    top_k_idx = (sim.argsort(kind="stable")[-k - 1 :][::-1])[1:]
    distances = [sim[x] for x in top_k_idx]
    return top_k_idx, distances


if __name__ == "__main__":
    words_sim = ["dog", "england", "john", "explode", "office"]
    with open("vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]
    vecs = np.loadtxt("wordVectors.txt")
    sim_dict = {word: most_similar(word, vocab, vecs) for word in words_sim}
    sim_dict_strings = {
        word: ([vocab[i] for i in sim_dict[word][0]], [i for i in sim_dict[word][1]])
        for word in words_sim
    }
    for word in words_sim:
        for i, w in enumerate(sim_dict_strings[word][0]):
            print(f"{word} -> {w}, distance: {sim_dict_strings[word][1][i].round(4)}")

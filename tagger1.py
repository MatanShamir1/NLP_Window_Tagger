import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

EMBEDDING_DIMS = 50

class Tagger(nn.Module):
    """
    A sequence tagger,where
    the input is a sequence of items(in our case, a sentence of natural-language words),
    and an output is a label for each of the item.
    The tagger will be greedy/local and window-based. For a sequence of words
    w1,...,wn, the tag of word wi will be based on the words in a window of
    two words to each side of the word being tagged: wi-2,wi-1,wi,wi+1,wi+2.
    'Greedy/local' here means that the tag assignment for word i will not depend on the tags of other words
    each word in the window will be assigned a 50 dimensional embedding vector, using an embedding matrix E.
    MLP with one hidden layer and tanh activation function.
    The output of the MLP will be passed through a softmax transformation, resulting in a probability distribution.
    The network will be trained with a cross-entropy loss.
    The vocabulary of E will be based on the words in the training set (you are not allowed to add to E words that appear only in the dev set).
    """

    def __init__(self, task, words_vocabulary, tags_vocabulary, embedding_matrix, window_size=2):
        super(Tagger, self).__init__()

        # 5 concat of 50 dimensional embedding vectors
        input_size = embedding_matrix.embedding_dim * (2 * window_size + 1)
        hidden_size = 250
        output_size = len(tags_vocabulary)
        self.embedding_matrix = embedding_matrix
        self.input = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x, windows):
        # get embedding vector for input
        x = idx_to_window_torch(x, windows, self.embedding_matrix)
        # x = self.embedding_matrix(x)
        x = self.input(x)
        x = self.tanh(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


def train_model(model, input_data, windows, epochs=1, lr=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for j in range(epochs):
        train_loader = DataLoader(
            input_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            optimizer.zero_grad(set_to_none=True)
            y_hat = model.forward(x, windows)
            # loss = loss_fn(y_hat, y)
            # loss.backward()
            loss = F.cross_entropy(y_hat, y)
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {j}, Loss: {train_loss/i}")


def test_model(model, input_data, windows):
    model.eval()
    total_loss = 0
    for i, data in enumerate(input_data, 0):
        x, y = data
        y_hat = model.forward(x, windows)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()


def idx_to_window_torch(idx, windows, embedding_matrix):
    """
    Convert a tensor of word indices into a tensor of word embeddings
    for a given window size and embedding matrix.

    Args:
        idx (torch.Tensor): A tensor of word indices of shape (batch_size, window_size).
        windows (int): The window size.
        embedding_matrix (torch.Tensor): A tensor of word embeddings.

    Returns:
        torch.Tensor: A tensor of word embeddings of shape (batch_size, window_size * embedding_size).
    """
    embedding_size = embedding_matrix.embedding_dim
    batch_size = idx.shape[0]

    # Map the idx_to_window() function to each index in the tensor using PyTorch's map() function
    windows = torch.tensor(list(map(lambda x: windows[x][0], idx.tolist())))
    window_size = windows.size()[1]
    # Use torch.reshape to flatten the tensor of windows into a 2D tensor
    windows_flat = torch.reshape(windows, (batch_size, -1))

    # Index into the embedding matrix to get the embeddings for each word in each window
    # Add a check to ensure that the input word index is within the bounds of the embedding matrix
    embeddings = []
    for i in range(batch_size):
        window = windows_flat[i]
        window_embeddings = []
        for j in range(window_size):
            word_idx = window[j].item()
            if word_idx >= embedding_matrix.num_embeddings or word_idx == -1:
                # If the word index is out of bounds, use the zero vector as the embedding
                embed = torch.zeros((embedding_size,))
            else:
                embed = embedding_matrix(torch.tensor(word_idx))
            window_embeddings.append(embed)
        embeddings.append(torch.cat(window_embeddings))

    # Use torch.stack to stack the tensor of embeddings into a 2D tensor
    embeddings = torch.stack(embeddings)

    return embeddings


def read_tagged_file(file_name, window_size=2):
    """
    Read data from a file and return token and label indices, vocabulary, and label vocabulary.

    Args:
        fname (str): The name of the file to read data from.
        window_size (int, optional): The size of the window for the token,from each side of the word. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - tokens_idx (numpy.ndarray): An array of token indices.
            - labels_idx (numpy.ndarray): An array of label indices.
            - vocab (set): A set of unique tokens in the data.
            - labels_vocab (set): A set of unique labels in the data.
    """
    words = []
    tags = []
    with open(file_name) as file:
        word_tag_lines = file.readlines()
        for word_tag_line in word_tag_lines:
            word_tag_line = word_tag_line.strip()
            if word_tag_line:
                word, tag = word_tag_line.split()
                words.append(word)
                tags.append(tag)

    words_vocabulary = set(words)
    words_vocabulary.add("<PAD>")  # add a padding token
    words_vocabulary.add("<UUUNKKK>")  # add an unknown token
    tags_vocabulary = set(tags)

    # Map words to their corresponding index in the vocabulary (word:idx)
    words_idx_dict = {word: i for i, word in enumerate(words_vocabulary)}
    tags_idx_dict = {tag: i for i, tag in enumerate(tags_vocabulary)}

    # For each window, map tokens to their index in the vocabulary
    words_idx = [words_idx_dict[word] for word in words]

    # tokens_idx = torch.from_numpy(tokens_idx)
    tags_idx = [tags_idx_dict[tag] for tag in tags]

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)
    windows = []
    for i in range(len(words)):
        window = []
        if i < window_size:
            for j in range(window_size - i):
                window.append(words_idx_dict["<PAD>"])
        window.extend(words_idx[max(0, i - window_size):min(len(words_idx), i + window_size + 1)])
        if i > len(words_idx) - window_size - 1:
            for j in range(i - (len(words_idx) - window_size - 1)):
                window.append(-1)
        windows.append((window, tags_idx[i]))

    return torch.tensor(words_idx), torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary


if __name__ == "__main__":
    words_idx, tags_idx, windows, words_vocabulary, tags_vocabulary = read_tagged_file("pos/train")
    embedding_matrix = nn.Embedding(len(words_vocabulary), EMBEDDING_DIMS)
    # initialize the embedding matrix to random values using xavier initialization which is a good initialization for NLP tasks
    nn.init.xavier_uniform_(embedding_matrix.weight)
    model = Tagger("pos", words_vocabulary, tags_vocabulary, embedding_matrix)
    dataset = TensorDataset(words_idx, tags_idx)
    train_model(model, input_data=dataset, epochs=10, windows=windows)

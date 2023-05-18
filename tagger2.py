import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

idx_to_label = {}
idx_to_word = {}


class Tagger(nn.Module):
    def __init__(self, task, vocab, labels_vocab, embedding_matrix):
        """
        Initializes a Tagger object.

        Args:
            task (str): The task to perform with the model (e.g., "ner" or "pos").
            vocab (set): The vocabulary object for the input data.
            labels_vocab (set): The vocabulary object for the output labels.
            embedding_matrix (torch.nn.Embedding): The matrix of pre-trained embeddings.

        Returns:
            None
        """
        super(Tagger, self).__init__()

        output_size = len(labels_vocab)  # TODO check if it works for us
        hidden_size = 150
        window_size = 5
        input_size = (
            embedding_matrix.embedding_dim * window_size
        )  # 5 concat. 50 dimensional embedding vectors, output over labels
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax()
        self.embedding_matrix = embedding_matrix
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the tagger.
        Expects x of shape (batch_size, window total size), which means 32 windows for example.

        Args:
            x: A tensor of shape (batch_size, seq_len) of word indices.

        Returns:
            A tensor of shape (batch_size, output_dim).
        """
        # Concatenate the word embedding vectors of the words in the window to create a single vector for the window.
        # x.shape[1] is the size of window, which is 5 in our case.
        # self.embedding_matrix(x[:,i]) is the 32 x 50 embedding matrix of the i'th items in the window.
        # torch.cat concatenates the 5 32 x 50 embedding matrix of the i'th items in the window to a 32 x 250 matrix, which we can pass to the linear layer.
        # x = torch.cat(
        #     [self.embedding_matrix(x[:, i]) for i in range(x.shape[1])], dim=1
        # )
        x = self.embedding_matrix(x).view(
            -1, 250
        )  # Faster than the above and equivalent
        x = self.in_linear(x)
        x = self.activate(x)
        x = self.out_linear(x)
        return x


def train_model(
    model, input_data, dev_data, windows, epochs=1, lr=0.01, input_data_win_index=None
):
    """
    Trains a given model using the provided input and development data.
    Args:
            model: The model to train.
            input_data: The training data.
            dev_data: The development data to evaluate the model.
            windows: The size of the windows to use.
            epochs: The number of epochs to train the model for. Default value is 1.
            lr: The learning rate to use. Default value is 0.01.
            input_data_win_index: The index of the window in the input data. Default value is None.
    Returns:
            A list of the accuracy of the model on the development data at the end of each epoch.
    """

    global idx_to_label
    BATCH_SIZE = 256
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    dropout = nn.Dropout(p=0.5)
    dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, windows)
    print(
        f"Before Training, Dev Loss: {dev_loss}, Dev Acc: {dev_acc} Acc No O:{dev_acc_clean}"
    )

    dev_loss_results = []
    dev_acc_results = []
    dev_acc_no_o_results = []
    for j in range(epochs):
        model.train()
        train_loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            optimizer.zero_grad(set_to_none=True)
            y_hat = model.forward(x)
            y_hat = dropout(y_hat)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # Evalaute model on dev at the end of each epoch.
        dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, windows)
        print(
            f"Epoch {j}/{epochs}, Loss: {train_loss/i}, Dev Loss: {dev_loss}, Dev Acc: {dev_acc} Acc No O:{dev_acc_clean}"
        )
        sched.step()
        dev_loss_results.append(dev_loss)
        dev_acc_results.append(dev_acc)
        dev_acc_no_o_results.append(dev_acc_clean)
    return dev_loss_results, dev_acc_results, dev_acc_no_o_results


def test_model(model, input_data, windows):
    """
    This function tests a PyTorch model on given input data and returns the validation loss, overall accuracy, and
    accuracy excluding "O" labels. It takes in the following parameters:

    - model: a PyTorch model to be tested
    - input_data: a dataset to test the model on
    - windows: a parameter that is not used in the function

    The function first initializes a batch size of 32 and a global variable idx_to_label. It then creates a DataLoader
    object with the input_data and the batch size, and calculates the validation loss, overall accuracy, and accuracy
    excluding "O" labels. These values are returned as a tuple.
    """

    BATCH_SIZE = 32
    global idx_to_label

    loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)
    running_val_loss = 0
    with torch.no_grad():
        model.eval()
        count = 0
        count_no_o = 0
        to_remove = 0
        for k, data in enumerate(loader, 0):
            x, y = data
            y_hat = model.forward(x)
            # y_hat = dropout(y_hat)
            val_loss = F.cross_entropy(y_hat, y)
            # Create a list of predicted labels and actual labels
            y_hat_labels = [idx_to_label[i.item()] for i in y_hat.argmax(dim=1)]
            y_labels = [idx_to_label[i.item()] for i in y]
            # Count the number of correct labels th
            y_agreed = sum(
                [
                    1 if (i == j and j != "O") else 0
                    for i, j in zip(y_hat_labels, y_labels)
                ]
            )
            count += sum(y_hat.argmax(dim=1) == y).item()
            count_no_o += y_agreed
            to_remove += y_labels.count("O")
            running_val_loss += val_loss.item()

    return (
        running_val_loss / k,
        count / (k * BATCH_SIZE),
        count_no_o / ((k * BATCH_SIZE) - to_remove),
    )


def replace_rare(dataset):
    from collections import Counter

    # Define a threshold for word frequency
    threshold = 2

    # Load the dataset into a list of strings (one string per document)

    # Tokenize the dataset into a list of words
    words = [word for doc in dataset for word in doc.split()]

    # Count the frequency of each word
    word_counts = Counter(words)

    # Find the set of rare words (words that occur less than the threshold)
    rare_words = set(word for word in word_counts if word_counts[word] < threshold)

    # Print the rare words
    # print(rare_words)
    updated = [word if word not in rare_words else "<UNK>" for word in dataset]
    return updated


def read_data(
    fname, window_size=2, vocab=None, labels_vocab=None, type="train", task=None
):
    """
    Reads in data from a file and preprocesses it for use in a neural network model.

    Args:
        fname (str): The name of the file to read data from.
        window_size (int, optional): The size of the context window to use when creating windows. Defaults to 2.
        vocab (set, optional): A set of unique tokens to use as the vocabulary. If not provided, a vocabulary will be built from the data. Defaults to None.
        labels_vocab (set, optional): A set of unique labels to use for classification. If not provided, labels will be inferred from the data. Defaults to None.
        type (str, optional): A string indicating the type of data being read. Defaults to "train".

    Returns:
        tuple: A tuple containing the preprocessed data:
            - tokens_idx (torch.Tensor): A tensor of indices representing the tokens in the data.
            - labels_idx (torch.Tensor): A tensor of indices representing the labels in the data.
            - windows (list): A list of tuples representing the context windows and labels for each token in the data.
            - vocab (set): A set of unique tokens used as the vocabulary.
            - labels_vocab (set): A set of unique labels used for classification.
            - windows_dict (dict): A dictionary mapping token indices to their corresponding context windows and labels.
            - task (str): The task to perform.
    """

    global idx_to_label
    global idx_to_word
    if task == "ner":
        SEPARATOR = "\t"
    else:
        SEPARATOR = " "

    with open(fname) as f:
        lines = f.readlines()
        # lines = [line.strip() for line in lines if line.strip()]
        tokens = []
        labels = []
        sentences = []
        for line in lines:
            if line == "\n":
                sentences.append((tokens, labels))
                tokens = []
                labels = []
                continue
            if type != "test":
                token, label = line.split(SEPARATOR)
            else:
                token = line.strip()
                label = ""
            if any(char.isdigit() for char in token) and label == "O":
                token = "$NUM"
            tokens.append(token)
            labels.append(label)
    # Preprocess data
    sentences = sentences[1:]  # remove docstart
    all_tokens = []
    all_labels = []
    for sentence in sentences:
        tokens, labels = sentence
        for i in range(len(tokens)):
            tokens[i] = tokens[i].strip()
            labels[i] = labels[i].strip()

        all_tokens.extend(tokens)
        all_labels.extend(labels)
        tokens = np.array(tokens)
        labels = np.array(labels)
        sentence[0].clear()
        sentence[0].extend(tokens)
        sentence[1].clear()
        sentence[1].extend(labels)
    # tokens = replace_rare(tokens)
    if not vocab:
        vocab = set(all_tokens)  # build a vocabulary of unique tokens
    vocab.add("<PAD>")  # add a padding token
    vocab.add("<UNK>")  # add an unknown token
    if not labels_vocab:
        labels_vocab = set(all_labels)

    # Map words to their corresponding index in the vocabulary (word:idx)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    labels_to_idx = {word: i for i, word in enumerate(labels_vocab)}

    idx_to_label = {i: label for label, i in labels_to_idx.items()}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)
    windows = []
    windows_dict = {}
    tokens_idx_all = []
    labels_idx_all = []
    for sentence in sentences:
        tokens, labels = sentence
        # map tokens to their index in the vocabulary
        tokens_idx = [
            word_to_idx[word] if word in word_to_idx else word_to_idx["<UNK>"]
            for word in tokens
        ]
        tokens_idx_all.extend(tokens_idx)
        labels_idx = [
            labels_to_idx[label] if label in labels_to_idx else labels_to_idx["O"]
            for label in labels
        ]
        labels_idx_all.extend(labels_idx)

        for i in range(len(tokens_idx)):
            start = max(0, i - window_size)
            end = min(len(tokens_idx), i + window_size + 1)
            context = (
                [word_to_idx["<PAD>"]] * (window_size - i + start)
                + tokens_idx[start:i]
                + tokens_idx[i:end]
                + [word_to_idx["<PAD>"]] * (window_size - end + i + 1)
            ) # list of indices of each window surrounding a word and the index of padding if the word is on the side.
            label = labels_idx[i]
            windows.append((context, label))
            # windows_dict[i] = (context, label)
    tokens_idx = torch.tensor(tokens_idx_all)
    labels_idx = torch.tensor(labels_idx_all)
    return tokens_idx, labels_idx, windows, vocab, labels_vocab, windows_dict


def load_embedding_matrix(embedding_path):
    with open("vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]
        vocab = set(vocab)
    vecs = np.loadtxt("wordVectors.txt")
    vecs = torch.from_numpy(vecs)
    vecs = vecs.float()
    return vocab, vecs


def plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task):
    # # Plot the dev loss, and save
    plt.plot(dev_loss, label="dev loss")
    plt.title(f"{task} task")
    plt.savefig(f"loss_{task}.png")
    plt.show()

    # # Plot the dev accuracy, and save
    plt.plot(dev_accuracy, label="dev accuracy")
    plt.title(f"{task} task")
    plt.savefig(f"accuracy_{task}.png")
    plt.show()

    # # Plot the dev accuracy no O, and save
    plt.plot(dev_accuracy_no_o, label="dev accuracy no o")
    plt.title(f"{task} task")
    plt.savefig(f"accuracy_no_O_{task}.png")
    plt.show()


def main(task="ner"):
    embeddings_vocab, vecs = load_embedding_matrix(f"wordVectors.txt" )

    tokens_idx, labels_idx, windows, vocab, labels_vocab, windows_dict = read_data(
        f"./{task}/train", task=task
    )

    train_missing_embeddings = [word for word in vocab if word not in embeddings_vocab]

    # create an empty embedding matrix, each vector is size 50
    # embedding_matrix = nn.Embedding(len(vocab), 50, _freeze=False)
    embedding_matrix = nn.Embedding.from_pretrained(
        vecs,
        freeze=False,
    )
    embedding_matrix.weight.requires_grad = True

    # initialize the embedding matrix to random values using xavier initialization which is a good initialization for NLP tasks
    # nn.init.xavier_uniform_(embedding_matrix.weight)

    model = Tagger(task, vocab, labels_vocab, embedding_matrix)

    # Make a new tensor out of the windows, so the tokens are windows of size window_size in the dataset
    tokenx_idx_new = torch.tensor([window for window, label in windows])

    dataset = TensorDataset(tokenx_idx_new, labels_idx)

    # Load the dev data
    (
        tokens_idx_dev,
        labels_idx_dev,
        windows_dev,
        vocab,
        labels_vocab,
        windows_dict,
    ) = read_data(f"./{task}/dev", task=task, vocab=vocab, labels_vocab=labels_vocab)

    tokenx_idx_dev_new = torch.tensor([window for window, label in windows_dev])
    dev_dataset = TensorDataset(tokenx_idx_dev_new, labels_idx_dev)
    # Get the dev loss from the model training
    dev_loss, dev_accuracy, dev_accuracy_no_o = train_model(
        model, input_data=dataset, dev_data=dev_dataset, epochs=10, windows=windows
    )

    plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task)


    # test_data
    # dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, windows)

    # print(
    #     f"Test Loss: {dev_loss}, Test Acc: {dev_acc} Acc No O:{dev_acc_clean}"
    # )
    # tokens_idx_test, labels_idx_test, windows_test, vocab_test, labels_vocab_test, windows_dict_test = read_data(
    #     "./ner/test",type="test"
    # )

    # tokens_idx, labels_idx, windows, vocab, labels_vocab = read_data("./ner/test")


if __name__ == "__main__":
    main("pos")

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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

    def __init__(self, task, tags_vocabulary, embedding_matrix, window_size=2):
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
        self.dropout = nn.Dropout(p=0.3)
        self.task = task

    def forward(self, x):
        # get embedding vector for input
        x = self.embedding_matrix(x).view(-1, 250)
        # x = self.embedding_matrix(x)
        x = self.input(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        # x = self.softmax(x)
        return x

def train_model(model, input_data, dev_data, tags_idx_dict, epochs=1, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train() # set model to training mode

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    dev_loss_results = []
    dev_acc_results = []
    dev_acc_no_o_results = []

    for j in range(epochs):
        train_loader = DataLoader(
            input_data, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True
        )
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            optimizer.zero_grad(set_to_none=True)
            y_hat = model.forward(x)
            loss = F.cross_entropy(y_hat, y)  # maybe don't need softmax and this does it alone
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # print(f"iteration {i} curr loss: {loss}")

        dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, tags_idx_dict)
        dev_loss_results.append(dev_loss)
        dev_acc_results.append(dev_acc)
        dev_acc_no_o_results.append(dev_acc_clean)

        print(
            f"Epoch {j+1}/{epochs}, Loss: {train_loss / i}, Dev Loss: {dev_loss}, Dev Acc: {dev_acc} Acc No O:{dev_acc_clean}"
        )
        scheduler.step()
    return dev_loss_results, dev_acc_results, dev_acc_no_o_results


def test_model(model, input_data, tags_idx_dict):
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

    BATCH_SIZE = 1024

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
            y_hat_labels = [i.item() for i in y_hat.argmax(dim=1)]
            y_labels = [i.item() for i in y]
            # Count the number of correct labels th
            if model.task == "ner":
                y_agreed = sum(
                    [
                        1 if (i == j and j != tags_idx_dict["O"]) else 0
                        for i, j in zip(y_hat_labels, y_labels)
                    ]
                )
            else:
                y_agreed = sum(
                    [
                        1 if (i == j) else 0
                        for i, j in zip(y_hat_labels, y_labels)
                    ]
                )
            count += sum(y_hat.argmax(dim=1) == y).item()
            count_no_o += y_agreed
            if model.task == "ner":
                to_remove += y_labels.count(tags_idx_dict["O"])
            running_val_loss += val_loss.item()

    return (
        running_val_loss / k,
        count / (k * BATCH_SIZE),
        count_no_o / ((k * BATCH_SIZE) - to_remove),
    )


def read_tagged_file(file_name, window_size=2, words_vocabulary=None, tags_vocabulary=None, file_type="train"):
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
    all_words = []
    all_tags = []

    with open(file_name) as f:
        lines = f.readlines()
        # lines = [line.strip() for line in lines if line.strip()]
        words = []
        tags = []
        sentences = []
        for line in lines:
            if line == "\n":
                sentences.append((np.array(words), np.array(tags)))
                words = []
                tags = []
                continue
            if file_type != "test":
                if "ner" in file_name:
                    word, tag = line.split('\t')
                else:
                    word, tag = line.split(' ')
                word = word.strip()
                tag = tag.strip()
                all_tags.append(tag)
                all_words.append(word)
            else:
                word = line.strip()
                tag = ""
            if any(char.isdigit() for char in word) and tag == "O":
                word = "<NUM>"
            words.append(word)
            tags.append(tag)

    if "ner" in file_name:
        sentences = sentences[1:]  # remove docstart

    if not words_vocabulary:
        words_vocabulary = set(all_words)
        words_vocabulary.add("<PAD>")  # add a padding token
        words_vocabulary.add("<UUUNKKK>")  # add an unknown token

    if not tags_vocabulary:
        tags_vocabulary = set(all_tags)

    # Map words to their corresponding index in the vocabulary (word:idx)
    words_idx_dict = {word: i for i, word in enumerate(words_vocabulary)}
    tags_idx_dict = {tag: i for i, tag in enumerate(tags_vocabulary)}

    # For each window, map tokens to their index in the vocabulary
    words_idx = [words_idx_dict[word] if word in words_idx_dict.keys() else words_idx_dict["<UUUNKKK>"] for word in all_words]

    # tokens_idx = torch.from_numpy(tokens_idx)
    tags_idx = [tags_idx_dict[tag] for tag in all_tags]

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)
    windows = []

    for sentence in sentences:
        words, tags = sentence
        for i in range(len(words)):
            window = []
            if i < window_size:
                for j in range(window_size - i):
                    window.append(words_idx_dict["<PAD>"])
            extra_words = words[max(0, i - window_size):min(len(words), i + window_size + 1)]
            window.extend([words_idx_dict[word] if word in words_idx_dict.keys() else words_idx_dict["<UUUNKKK>"] for word in extra_words])
            if i > len(words) - window_size - 1:
                for j in range(i - (len(words) - window_size - 1)):
                    window.append(words_idx_dict["<PAD>"])
            windows.append((window, tags_idx_dict[tags[i]]))

    return torch.tensor(words_idx), torch.tensor(tags_idx), windows, words_vocabulary,\
           tags_vocabulary, words_idx_dict, tags_idx_dict


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



if __name__ == "__main__":

    # preprocess data files
    task = "pos"
    words_idx, tags_idx, windows, words_vocabulary, tags_vocabulary, words_idx_dict, tags_idx_dict = read_tagged_file(
        f"./{task}/train", file_type="train")
    words_idx_dev, tags_idx_dev, windows_dev, _, _, _, _ = read_tagged_file(
        f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev")

    # initialize model and embedding matrix and dataset
    embedding_matrix = nn.Embedding(len(words_vocabulary), EMBEDDING_DIMS)
    nn.init.xavier_uniform_(embedding_matrix.weight)
    model = Tagger("pos", tags_vocabulary, embedding_matrix)
    word_window_idx = torch.tensor([window for window, tag in windows])
    word_window_idx_dev = torch.tensor([window for window, tag in windows_dev])
    dataset = TensorDataset(word_window_idx, tags_idx)
    dev_dataset = TensorDataset(word_window_idx_dev, tags_idx_dev)

    # train model
    dev_loss, dev_accuracy, dev_accuracy_no_o = train_model(
        model, input_data=dataset, dev_data=dev_dataset, tags_idx_dict=tags_idx_dict, epochs=10)

    # plot the results
    plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task)

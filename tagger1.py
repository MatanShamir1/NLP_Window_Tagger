import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys

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
        self.softmax = nn.Softmax(dim=1)
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
    model.train()  # set model to training mode

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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
            f"Epoch {j + 1}/{epochs}, Loss: {train_loss / i}, Dev Loss: {dev_loss}, Dev Acc: {dev_acc} Acc No O:{dev_acc_clean}"
        )
        # scheduler.step()
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


def create_test_predictions(model, input_data, task, idx_tags_dict, all_test_words):
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

    BATCH_SIZE = 256

    loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    predictions = []
    with torch.no_grad():
        model.eval()
        j = 0
        for _, data in enumerate(loader, 0):
            x, y = data
            y_hat = model.forward(x)
            y_hat = model.softmax(y_hat)
            x_words = [all_test_words[i + j] for i, _ in enumerate(x)]
            y_hat_labels = [idx_tags_dict[i.item()] for i in y_hat.argmax(dim=1)]
            predictions.extend(zip(x_words, y_hat_labels))
            j += BATCH_SIZE

    with open(f"test1.{task}", "w") as f:
        for pred in predictions:
            f.write(f"{pred[0]} {pred[1]}" + "\n")


def read_tagged_file(file_name, window_size=2, words_vocabulary=None, tags_vocabulary=None, file_type="train",
                     pretrained_words_vocab=None, embedding_vecs=None, is_pretrained=False, is_subword=False):
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
    all_pre_words = []
    all_suf_words = []
    suf_vocab = None
    pre_vocab = None

    # the sentences always remain the same, regardless of the method used: an index for each word.
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
            if file_type != "test":  # keep words in the same case if in test.
                if "ner" in file_name:
                    word, tag = line.split('\t')
                else:
                    word, tag = line.split(' ')
                word = word.strip().lower()
                tag = tag.strip()
                all_tags.append(tag)
            else:
                word = line.strip()
                tag = ""
            if any(char.isdigit() for char in word) and tag == "O":
                word = "NUM"
            all_words.append(word)
            if is_subword:
                if len(word) < 3:
                    all_pre_words.append(word)
                    all_suf_words.append(word)
                else:
                    all_pre_words.append(word[:3])
                    all_suf_words.append(word[-3:])
            words.append(word)
            tags.append(tag)

    if not words_vocabulary:  # if in train case
        if is_subword:
            pre_vocab = set(all_pre_words)
            suf_vocab = set(all_suf_words)
            pre_vocab.add("PAD")  # add a padding token
            pre_vocab.add("UNK")  # add an unknown token
            suf_vocab.add("PAD")  # add a padding token
            suf_vocab.add("UNK")  # add an unknown token
            # no need to add NUM to pre and suf because it was already generated

        words_vocabulary = set(all_words)
        words_vocabulary.add("PAD")  # add a padding token
        words_vocabulary.add("UNK")  # add an unknown token

        if is_pretrained:  # if tagger2 case, loading from embedding matrix
            train_missing_embeddings = [word for word in words_vocabulary if word not in pretrained_words_vocab]
            for word in train_missing_embeddings:
                size = embedding_vecs.shape[1]  # Get the size of the vector
                vector = torch.empty(1, size)
                nn.init.xavier_uniform_(vector)  # Apply Xavier uniform initialization
                embedding_vecs = torch.cat((embedding_vecs, vector), dim=0)  # add as last row
                pretrained_words_vocab.append(word)  # add as last word
            # if we're using pretrained embeddings, the vocabulary changes. turning this into a set doesn't change
            # the size, these are already unique words.
            words_vocabulary = pretrained_words_vocab

    if not tags_vocabulary:  # if in train case
        tags_vocabulary = set(all_tags)

    # Map words to their corresponding index in the vocabulary (word:idx)
    # this adjusted to the new vocabulary if we use pretrained too.
    words_idx_dict = {word: i for i, word in enumerate(words_vocabulary)}
    tags_idx_dict = {tag: i for i, tag in enumerate(tags_vocabulary)}

    pre_words_idx_dict = {pre_word: i for i, pre_word in enumerate(pre_vocab)}
    suf_words_idx_dict = {suf_word: i for i, suf_word in enumerate(suf_vocab)}

    # tokens_idx = torch.from_numpy(tokens_idx)
    tags_idx = [tags_idx_dict[tag] for tag in all_tags]

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)
    windows = []

    if is_subword:
        for sentence in sentences:
            words, tags = sentence
            for i in range(len(words)):
                window = []
                if i < window_size:
                    for j in range(window_size - i):
                        window.append((pre_words_idx_dict["PAD"], words_idx_dict["PAD"], suf_words_idx_dict["PAD"]))
                extra_words = words[max(0, i - window_size):min(len(words), i + window_size + 1)]
                window.extend(
                    [(pre_words_idx_dict[word.lower()], words_idx_dict[word.lower()], suf_words_idx_dict[word.lower()])
                     if word.lower() in words_idx_dict.keys() else
                     (pre_words_idx_dict["UNK"], words_idx_dict["UNK"], suf_words_idx_dict["UNK"])
                     for word in extra_words]
                )
                if i > len(words) - window_size - 1:
                    for j in range(i - (len(words) - window_size - 1)):
                        window.append((pre_words_idx_dict["PAD"], words_idx_dict["PAD"], suf_words_idx_dict["PAD"]))
                if file_type == "test":
                    windows.append((window, 0))
                else:
                    windows.append((window, tags_idx_dict[tags[i]]))
    else:
        for sentence in sentences:
            words, tags = sentence
            for i in range(len(words)):
                window = []
                if i < window_size:
                    for j in range(window_size - i):
                        window.append(words_idx_dict["PAD"])
                extra_words = words[max(0, i - window_size):min(len(words), i + window_size + 1)]
                window.extend(
                    [words_idx_dict[word.lower()] if word.lower() in words_idx_dict.keys() else words_idx_dict["UNK"]
                     for word in extra_words])
                if i > len(words) - window_size - 1:
                    for j in range(i - (len(words) - window_size - 1)):
                        window.append(words_idx_dict["PAD"])
                if file_type == "test":
                    windows.append((window, 0))
                else:
                    windows.append((window, tags_idx_dict[tags[i]]))

    if is_subword:
        if embedding_vecs != None:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, \
                embedding_vecs, pre_vocab, suf_vocab
        else:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, None, \
                pre_vocab, suf_vocab

    else:
        if embedding_vecs != None:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, \
                embedding_vecs, None, None
        else:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, None, \
                None, None


import matplotlib.pyplot as plt


def plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task):
    # Plot the dev loss, and save
    plt.figure()  # Create a new figure
    plt.plot(dev_loss, label=f"{task} dev loss")
    plt.title(f"{task} task")
    plt.savefig(f"{task}_loss.jpg")
    plt.show()

    # Plot the dev accuracy, and save
    plt.figure()  # Create a new figure
    plt.plot(dev_accuracy, label=f"{task} dev accuracy")
    plt.title(f"{task} task")
    plt.savefig(f"{task}_accuracy.jpg")
    plt.show()

    if task == "ner":
        # Plot the dev accuracy no O, and save
        plt.figure()  # Create a new figure
        plt.plot(dev_accuracy_no_o, label=f"{task} dev accuracy without O")
        plt.title(f"{task} task")
        plt.savefig(f"{task}_without_O_accuracy.jpg")
        plt.show()


def load_pretrained_embedding(vocab_path, word_vectors_path):
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]
    vecs = np.loadtxt(word_vectors_path)
    vecs = torch.from_numpy(vecs)
    vecs = vecs.float()
    return vocab, vecs


def run(task, embed, sub_word_method):
    if sub_word_method == "sub_word":
        if embed == "pre":
            words_embedding_vocabulary, embedding_vecs = load_pretrained_embedding("vocab.txt", "wordVectors.txt")
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, embedding_vecs = \
                read_tagged_file(
                    f"./{task}/train", file_type="train", pretrained_words_vocab=words_embedding_vocabulary,
                    embedding_vecs=embedding_vecs, is_pretrained=True, is_subword=True)

            tags_idx_dev, windows_dev, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=True, is_subword=True)

            embedding_matrix = nn.Embedding.from_pretrained(embedding_vecs, freeze=False)
        else:
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, _ = read_tagged_file(
                f"./{task}/train", file_type="train", is_pretrained=False, is_subword=True)

            tags_idx_dev, windows_dev, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=False, is_subword=True)
            # initialize model and embedding matrix and dataset
            embedding_matrix = nn.Embedding(len(words_vocabulary), EMBEDDING_DIMS)
            nn.init.xavier_uniform_(embedding_matrix.weight)

    else:
        if embed == "pre":
            words_embedding_vocabulary, embedding_vecs = load_pretrained_embedding("vocab.txt", "wordVectors.txt")
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, embedding_vecs = \
                read_tagged_file(
                    f"./{task}/train", file_type="train", pretrained_words_vocab=words_embedding_vocabulary,
                    embedding_vecs=embedding_vecs, is_pretrained=True, is_subword=False)

            tags_idx_dev, windows_dev, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=True, is_subword=False)

            embedding_matrix = nn.Embedding.from_pretrained(embedding_vecs, freeze=False)
        else:
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, _ = read_tagged_file(
                f"./{task}/train", file_type="train", is_pretrained=False, is_subword=False)

            tags_idx_dev, windows_dev, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=False, is_subword=False)

            # initialize model and embedding matrix and dataset
            embedding_matrix = nn.Embedding(len(words_vocabulary), EMBEDDING_DIMS)
            nn.init.xavier_uniform_(embedding_matrix.weight)

    model = Tagger(task, tags_vocabulary, embedding_matrix)
    word_window_idx = torch.tensor([window for window, tag in windows])

    # the windows were generated according to the new pretrained vocab (plus missing from train) if we use pretrained.
    word_window_idx_dev = torch.tensor([window for window, tag in windows_dev])
    dataset = TensorDataset(word_window_idx, tags_idx)
    dev_dataset = TensorDataset(word_window_idx_dev, tags_idx_dev)

    lr = 0.0006
    epochs = 10

    if task == "pos":
        lr = 0.0001
        epochs = 15

    if embed == "pre":
        lr += 0.0004

    # train model
    dev_loss, dev_accuracy, dev_accuracy_no_o = train_model(
        model, input_data=dataset, dev_data=dev_dataset, tags_idx_dict=tags_idx_dict, epochs=epochs, lr=lr)

    # plot the results
    plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task)

    print("Test")
    _, windows_test, _, _, _, all_test_words, _ = \
        read_tagged_file(f"./{task}/test", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary,
                         file_type="test")

    word_window_idx_test = torch.tensor([window for window, tag in windows_test])
    test_dataset = TensorDataset(word_window_idx_test, torch.tensor([0] * len(word_window_idx_test)))
    idx_tags_dict = {i: tag for i, tag in enumerate(tags_vocabulary)}

    create_test_predictions(model, test_dataset, task, idx_tags_dict, all_test_words)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] in ["ner", "pos"] and sys.argv[2] in ["pre", "rand"] \
            and sys.argv[3] in ["sub_word", "all_word"]:
        run(sys.argv[1], sys.argv[2], sys.argv[3])

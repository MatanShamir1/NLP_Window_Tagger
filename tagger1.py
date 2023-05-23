import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import math
import string

EMBEDDING_DIMS = 50


class Tagger(nn.Module):

    def __init__(self, task, tags_vocabulary, embedding_matrix, window_size=2,
                 pre_embedding_matrix=None, suf_embedding_matrix=None, char_embedding=None,
                 num_filters=30, filter_size=3, padding=2):
        super(Tagger, self).__init__()

        # 5 concat of 50 dimensional embedding vectors
        self.window_size = window_size
        self.input_size = embedding_matrix.embedding_dim * (2 * window_size + 1)
        self.hidden_size = 250
        self.output_size = len(tags_vocabulary)
        self.embedding_matrix = embedding_matrix
        self.pre_embedding_matrix = pre_embedding_matrix
        self.suf_embedding_matrix = suf_embedding_matrix
        self.char_embedding = char_embedding
        if char_embedding != None:
            self.chars_cnn = CharsCNN(char_embedding=char_embedding, num_filters=num_filters, filter_size=filter_size,
                                      padding=padding)
            # change input size, because we concat char embeddings after CNN too. CNN output=
            self.input_size += num_filters * (
                    2 * window_size + 1)  # each word size grew by num_filters, which is output of CNN
        self.input = nn.Linear(self.input_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.task = task

    def forward(self, x):
        # get embedding vector for input
        if self.pre_embedding_matrix != None and self.suf_embedding_matrix != None:
            pre_x, word_x, suf_x = x[:, :, 0], x[:, :, 1], x[:, :, 2]
            x = self.pre_embedding_matrix(pre_x).view(-1, (self.window_size * 2 + 1) * self.pre_embedding_matrix.embedding_dim) + \
                self.embedding_matrix(word_x).view(-1, (self.window_size * 2 + 1) * self.embedding_matrix.embedding_dim) + \
                self.suf_embedding_matrix(suf_x).view(-1, (self.window_size * 2 + 1) * self.suf_embedding_matrix.embedding_dim)  # after the view its 32*250 instead of 32*5*50
        elif self.char_embedding != None:
            # transform the batch of windows of regular words to batch of windows of concatenated words with CNN output:
            # 32*5*50 -> 32*5*80: 32*5*50 concat 32*5*30 -> 32*5*80
            addition_to_x = torch.empty((x.shape[0], self.window_size * 2 + 1, self.chars_cnn.num_filters))
            for j in range(x.shape[1]):
                addition_to_x[:, j, :] = self.chars_cnn.forward(x[:, j, :max(x[:, j, x.shape[2] - 2])]).reshape(x.shape[0],
                                                                                         self.chars_cnn.num_filters)

            x = torch.cat([self.embedding_matrix(x[:, :, x.shape[2] - 1]), addition_to_x], dim=2)  # 32*5*50 -> 32*5*80
            x = x.view(-1,
                       (self.chars_cnn.num_filters + self.embedding_matrix.embedding_dim) * (self.window_size * 2 + 1))
        else:
            x = self.embedding_matrix(x).view(-1, (self.window_size * 2 + 1) * self.embedding_matrix.embedding_dim)
        x = self.input(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class CharsCNN(nn.Module):
    def __init__(
            self, char_embedding, num_filters=30, filter_size=3, padding=2
    ):
        super(CharsCNN, self).__init__()
        # Get the matrix of char embeddings
        self.padding = padding
        self.num_filters = num_filters
        self.char_embedding = char_embedding
        self.char_embedding_dim = char_embedding.embedding_dim
        # must be size of window (how many chars in one filter) times embedding vector size.
        self.filter_size = filter_size
        self.conv_output_dim = num_filters
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, idx_lst):  # receives a list of indices, of places of chars in the embedding chars matrix.
        self.conv_input_dim = self.char_embedding_dim  # look at the sequence as one long vector
        # next layer is num_filters*num_chars, and max_pooling on num_chars gives num_filters
        self.conv1d = nn.Conv1d(
            self.conv_input_dim,
            self.num_filters,
            self.filter_size,
            padding=self.padding
        )
        # kernel size is number of columns, we want to get one item from each row. they're derived from previous layer
        self.max_pool = nn.MaxPool1d(kernel_size=idx_lst.shape[1] + self.padding,
                                     stride=idx_lst.shape[1] + self.padding)  # make sure padding keeps input size

        out = self.char_embedding(idx_lst).permute(0,2,1)
        out = self.dropout(out)
        out = self.conv1d(out)
        out = self.max_pool(out)
        return out


def train_model(model, input_data, dev_data, tags_idx_dict, epochs=1, lr=0.0001):
    BATCH_SIZE = 256

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()  # set model to training mode

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    dev_loss_results = []
    dev_acc_results = []
    dev_acc_no_o_results = []

    for j in range(epochs):
        train_loader = DataLoader(
            input_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
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

        dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, tags_idx_dict)
        dev_loss_results.append(dev_loss)
        dev_acc_results.append(dev_acc)
        dev_acc_no_o_results.append(dev_acc_clean)

        print(
            f"Epoch {j + 1}/{epochs}, Loss: {train_loss / i}, Dev Loss: {dev_loss}, Dev Acc: {dev_acc}"
            f" Dev Acc Without O:{dev_acc_clean}"
        )
        # scheduler.step()
    return dev_loss_results, dev_acc_results, dev_acc_no_o_results


def test_model(model, input_data, tags_idx_dict):

    BATCH_SIZE = 256

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
        running_val_loss / (k + 1),
        count / len(input_data),
        count_no_o / (len(input_data) - to_remove),
    )


def create_test_predictions(model, input_data, task, idx_tags_dict, all_test_words, embed, sub_word_method):

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

    with open(f"test1_{embed}_{sub_word_method}.{task}", "w") as f:
        for pred in predictions:
            if task == "ner":
                f.write(f"{pred[0]}\t{pred[1]}\n")
            else:
                f.write(f"{pred[0]} {pred[1]}\n")


def read_tagged_file(file_name, window_size=2, words_vocabulary=None, tags_vocabulary=None, file_type="train",
                     pretrained_words_vocab=None, embedding_vecs=None, pre_vocab=None, suf_vocab=None,
                     is_pretrained=False, subword_method="all_word", char_vocab=None):
    all_words = []
    all_tags = []
    all_pre_words = []
    all_suf_words = []
    longest_word_len = 0

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
                if file_type == "dev" and is_pretrained:
                    word = word.strip().lower()
                else:
                    word = word.strip()
                tag = tag.strip()
                all_tags.append(tag)
            else:
                word = line.strip()
                tag = ""
            if any(char.isdigit() for char in word) and tag == "O":
                word = "NUM"
            longest_word_len = max(longest_word_len, len(word))
            all_words.append(word)
            if subword_method == "sub_word":
                if len(word) < 3:
                    all_pre_words.append(word)
                    all_suf_words.append(word)
                else:
                    all_pre_words.append(word[:3])
                    all_suf_words.append(word[-3:])
            words.append(word)
            tags.append(tag)

    if not words_vocabulary:  # if in train case

        words_vocabulary = set(all_words)
        words_vocabulary.add("PAD")  # add a padding token
        words_vocabulary.add("UNK")  # add an unknown token

        if is_pretrained:  # if tagger2 case, loading from embedding matrix
            all_pre_words.extend([word[:3] for word in pretrained_words_vocab])
            all_suf_words.extend([word[-3:] for word in pretrained_words_vocab])
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

        if subword_method == "sub_word":
            pre_vocab = set(all_pre_words)
            suf_vocab = set(all_suf_words)
            pre_vocab.add("PAD")  # add a padding token
            pre_vocab.add("UNK")  # add an unknown token
            suf_vocab.add("PAD")  # add a padding token
            suf_vocab.add("UNK")  # add an unknown token
            # no need to add NUM to pre and suf because it was already generated

    if not tags_vocabulary:  # if in train case
        tags_vocabulary = set(all_tags)
        if subword_method == "char_word":
            # Create a vocabulary of unique characters for character embeddings
            ascii_chars = set(string.ascii_letters + string.digits + string.punctuation + string.whitespace)
            char_vocab = ascii_chars.union(set([char for word in all_words for char in word])) # shouldn't be additions

    # Map words to their corresponding index in the vocabulary (word:idx)
    # this adjusted to the new vocabulary if we use pretrained too.
    words_idx_dict = {word: i for i, word in enumerate(words_vocabulary)}
    tags_idx_dict = {tag: i for i, tag in enumerate(tags_vocabulary)}

    if subword_method == "sub_word":
        pre_words_idx_dict = {pre_word: i for i, pre_word in enumerate(pre_vocab)}
        suf_words_idx_dict = {suf_word: i for i, suf_word in enumerate(suf_vocab)}
    elif subword_method == "char_word":
        char_to_idx = {char: i for i, char in enumerate(char_vocab)}

    # tokens_idx = torch.from_numpy(tokens_idx)
    tags_idx = [tags_idx_dict[tag] for tag in all_tags]

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)
    windows = []

    if subword_method == "sub_word":
        for sentence in sentences:
            words, tags = sentence
            for i in range(len(words)):
                window = []
                if i < window_size:
                    for j in range(window_size - i):
                        window.append((pre_words_idx_dict["PAD"], words_idx_dict["PAD"], suf_words_idx_dict["PAD"]))
                extra_words = words[max(0, i - window_size):min(len(words), i + window_size + 1)]
                window_tuples = []
                for word in extra_words:
                    pre_in_tuple, word_in_tuple, suf_in_tuple = pre_words_idx_dict["UNK"], words_idx_dict["UNK"], \
                        suf_words_idx_dict["UNK"]

                    if is_pretrained:
                        if word.lower() in words_idx_dict.keys():
                            pre_in_tuple, word_in_tuple, suf_in_tuple = pre_words_idx_dict[word[:3].lower()], \
                                words_idx_dict[word.lower()], \
                                suf_words_idx_dict[word[-3:].lower()]
                            window_tuples.append((pre_in_tuple, word_in_tuple, suf_in_tuple))
                            continue
                        if word[:3].lower() in pre_words_idx_dict:
                            pre_in_tuple = pre_words_idx_dict[word[:3].lower()]
                        if word[-3:].lower() in suf_words_idx_dict:
                            suf_in_tuple = suf_words_idx_dict[word[-3:].lower()]
                    else:
                        if word in words_idx_dict.keys():
                            pre_in_tuple, word_in_tuple, suf_in_tuple = pre_words_idx_dict[word[:3]], \
                                words_idx_dict[word], \
                                suf_words_idx_dict[word[-3:]]
                            window_tuples.append((pre_in_tuple, word_in_tuple, suf_in_tuple))
                            continue
                        if word[:3] in pre_words_idx_dict:
                            pre_in_tuple = pre_words_idx_dict[word[:3]]
                        if word[-3:] in suf_words_idx_dict:
                            suf_in_tuple = suf_words_idx_dict[word[-3:]]

                    window_tuples.append((pre_in_tuple, word_in_tuple, suf_in_tuple))
                window.extend(window_tuples)
                if i > len(words) - window_size - 1:
                    for j in range(i - (len(words) - window_size - 1)):
                        window.append((pre_words_idx_dict["PAD"], words_idx_dict["PAD"], suf_words_idx_dict["PAD"]))
                if file_type == "test":
                    windows.append((window, 0))
                else:
                    windows.append((window, tags_idx_dict[tags[i]]))
    elif subword_method == "char_word":
        for sentence in sentences:
            words, tags = sentence
            for i in range(len(words)):
                window = []
                if i < window_size:
                    for j in range(window_size - i):
                        indices_chars = [char_to_idx[char] for char in "PAD"]
                        while len(indices_chars) <= longest_word_len:
                            indices_chars.append(len(char_vocab))
                        indices_chars.append(len("PAD"))
                        indices_chars.append(words_idx_dict["PAD"])
                        window.append(indices_chars)
                extra_words = words[max(0, i - window_size):min(len(words), i + window_size + 1)]
                for word in extra_words:
                    if is_pretrained:
                        if word.lower() in words_idx_dict.keys():
                            word_in_tuple = words_idx_dict[word.lower()]
                            indices_chars = [char_to_idx[char] for char in word.lower()]
                            while len(indices_chars) <= longest_word_len:
                                indices_chars.append(len(char_vocab))
                            indices_chars.append(len(word))
                            indices_chars.append(word_in_tuple)
                            window.append(indices_chars)
                        else:
                            word_in_tuple = words_idx_dict["UNK"]
                            indices_chars = [char_to_idx[char] for char in "UNK"]
                            while len(indices_chars) <= longest_word_len:
                                indices_chars.append(len(char_vocab))
                            indices_chars.append(len("UNK"))
                            indices_chars.append(word_in_tuple)
                            window.append(indices_chars)
                    else:
                        if word in words_idx_dict.keys():
                            word_in_tuple = words_idx_dict[word]
                            indices_chars = [char_to_idx[char] for char in word]
                            while len(indices_chars) <= longest_word_len:
                                indices_chars.append(len(char_vocab))
                            indices_chars.append(len(word))
                            indices_chars.append(word_in_tuple)
                            window.append(indices_chars)
                        else:
                            word_in_tuple = words_idx_dict["UNK"]
                            indices_chars = [char_to_idx[char] for char in "UNK"]
                            while len(indices_chars) <= longest_word_len:
                                indices_chars.append(len(char_vocab))
                            indices_chars.append(len("UNK"))
                            indices_chars.append(word_in_tuple)
                            window.append(indices_chars)
                if i > len(words) - window_size - 1:
                    for j in range(i - (len(words) - window_size - 1)):
                        indices_chars = [char_to_idx[char] for char in "PAD"]
                        while len(indices_chars) <= longest_word_len:
                            indices_chars.append(len(char_vocab))
                        indices_chars.append(len("PAD"))
                        indices_chars.append(words_idx_dict["PAD"])
                        window.append(indices_chars)
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
                if is_pretrained:
                    window.extend([words_idx_dict[word.lower()] if word.lower()
                                                                   in words_idx_dict.keys() else words_idx_dict["UNK"]
                                   for word in extra_words])
                else:
                    window.extend([words_idx_dict[word] if word
                                                           in words_idx_dict.keys() else words_idx_dict["UNK"]
                                   for word in extra_words])
                if i > len(words) - window_size - 1:
                    for j in range(i - (len(words) - window_size - 1)):
                        window.append(words_idx_dict["PAD"])
                if file_type == "test":
                    windows.append((window, 0))
                else:
                    windows.append((window, tags_idx_dict[tags[i]]))

    if subword_method == "sub_word":
        if embedding_vecs != None:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, \
                embedding_vecs, pre_vocab, suf_vocab, None, None
        else:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, None, \
                pre_vocab, suf_vocab, None, None

    elif subword_method == "char_word":
        if embedding_vecs != None:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, \
                embedding_vecs, pre_vocab, suf_vocab, char_vocab, char_to_idx
        else:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, None, \
                pre_vocab, suf_vocab, char_vocab, char_to_idx
    else:
        if embedding_vecs != None:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, \
                embedding_vecs, None, None, None, None
        else:
            return torch.tensor(tags_idx), windows, words_vocabulary, tags_vocabulary, tags_idx_dict, all_words, None, \
                None, None, None, None


def plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task, embed, sub_word_method):
    # Plot the dev loss, and save
    plt.figure()  # Create a new figure
    plt.plot(dev_loss, label=f"{task} {embed} {sub_word_method} dev loss")
    plt.title(f"{task} {embed} {sub_word_method} task")
    plt.savefig(f"{task}_{embed}_{sub_word_method}_loss.jpg")
    plt.show()

    # Plot the dev accuracy, and save
    plt.figure()  # Create a new figure
    plt.plot(dev_accuracy, label=f"{task} {embed} {sub_word_method} dev accuracy")
    plt.title(f"{task} {embed} {sub_word_method} task")
    plt.savefig(f"{task}_{embed}_{sub_word_method}_accuracy.jpg")
    plt.show()

    if task == "ner":
        # Plot the dev accuracy no O, and save
        plt.figure()  # Create a new figure
        plt.plot(dev_accuracy_no_o, label=f"{task} {embed} {sub_word_method} dev accuracy without O")
        plt.title(f"{task} {embed} {sub_word_method} task")
        plt.savefig(f"{task}_{embed}_{sub_word_method}_without_O_accuracy.jpg")
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
    pre_embedding_matrix = None
    suf_embedding_matrix = None

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("GPU is available!")
    else:
        print("GPU is not available, using CPU instead.")

    if sub_word_method == "sub_word":
        if embed == "pre":
            words_embedding_vocabulary, embedding_vecs = load_pretrained_embedding("vocab.txt", "wordVectors.txt")
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, \
                embedding_vecs, pre_vocab, suf_vocab, _, _ = \
                read_tagged_file(f"./{task}/train", file_type="train",
                                 pretrained_words_vocab=words_embedding_vocabulary,
                                 embedding_vecs=embedding_vecs, is_pretrained=True, subword_method=sub_word_method)

            tags_idx_dev, windows_dev, _, _, _, _, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, pre_vocab=pre_vocab,
                suf_vocab=suf_vocab, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=True, subword_method=sub_word_method)

            embedding_matrix = nn.Embedding.from_pretrained(embedding_vecs, freeze=False)
        else:
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, _, \
                pre_vocab, suf_vocab, _, _ = read_tagged_file(
                f"./{task}/train", file_type="train", is_pretrained=False, subword_method=sub_word_method)

            tags_idx_dev, windows_dev, _, _, _, _, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                pre_vocab=pre_vocab, suf_vocab=suf_vocab, is_pretrained=False, subword_method=sub_word_method)
            # initialize model and embedding matrix and dataset
            embedding_matrix = nn.Embedding(len(words_vocabulary), EMBEDDING_DIMS)
            nn.init.xavier_uniform_(embedding_matrix.weight)

        pre_embedding_matrix = nn.Embedding(len(pre_vocab), EMBEDDING_DIMS)
        nn.init.xavier_uniform_(pre_embedding_matrix.weight)
        suf_embedding_matrix = nn.Embedding(len(suf_vocab), EMBEDDING_DIMS)
        nn.init.xavier_uniform_(suf_embedding_matrix.weight)

    elif sub_word_method == "char_word":
        if embed == "pre":
            words_embedding_vocabulary, embedding_vecs = load_pretrained_embedding("vocab.txt", "wordVectors.txt")
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, \
                embedding_vecs, _, _, char_vocab, char_to_idx = \
                read_tagged_file(f"./{task}/train", file_type="train",
                                 pretrained_words_vocab=words_embedding_vocabulary,
                                 embedding_vecs=embedding_vecs, is_pretrained=True, subword_method=sub_word_method)

            tags_idx_dev, windows_dev, _, _, _, _, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=True, subword_method=sub_word_method, char_vocab=char_vocab)

            embedding_matrix = nn.Embedding.from_pretrained(embedding_vecs, freeze=False)
        else:
            tags_idx, windows, words_vocabulary, tags_vocabulary, \
                tags_idx_dict, _, _, _, _, char_vocab, char_to_idx = read_tagged_file(
                f"./{task}/train", file_type="train", is_pretrained=False, subword_method=sub_word_method)

            tags_idx_dev, windows_dev, _, _, _, _, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=False, subword_method=sub_word_method, char_vocab=char_vocab)

            # initialize model and embedding matrix and dataset
            embedding_matrix = nn.Embedding(len(words_vocabulary), EMBEDDING_DIMS)
            nn.init.xavier_uniform_(embedding_matrix.weight)

    else:
        if embed == "pre":
            words_embedding_vocabulary, embedding_vecs = load_pretrained_embedding("vocab.txt", "wordVectors.txt")
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, embedding_vecs, _, _, _, _ = \
                read_tagged_file(f"./{task}/train", file_type="train",
                                 pretrained_words_vocab=words_embedding_vocabulary,
                                 embedding_vecs=embedding_vecs, is_pretrained=True, subword_method=sub_word_method)

            tags_idx_dev, windows_dev, _, _, _, _, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=True, subword_method=sub_word_method)

            embedding_matrix = nn.Embedding.from_pretrained(embedding_vecs, freeze=False)
        else:
            tags_idx, windows, words_vocabulary, tags_vocabulary, tags_idx_dict, _, _, _, _, _, _ = read_tagged_file(
                f"./{task}/train", file_type="train", is_pretrained=False, subword_method=sub_word_method)

            tags_idx_dev, windows_dev, _, _, _, _, _, _, _, _, _ = read_tagged_file(
                f"./{task}/dev", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary, file_type="dev",
                is_pretrained=False, subword_method=sub_word_method)

            # initialize model and embedding matrix and dataset
            embedding_matrix = nn.Embedding(len(words_vocabulary), EMBEDDING_DIMS)
            nn.init.xavier_uniform_(embedding_matrix.weight)

    char_embedding = None
    if sub_word_method == "char_word":
        # initialize model and embedding matrix and dataset
        embedding = torch.FloatTensor(len(char_vocab), 30).uniform_(-math.sqrt(3 / 30), math.sqrt(3 / 30))
        embedding = torch.cat((embedding, torch.zeros(1, 30)), dim=0)
        char_embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
    model = Tagger(task, tags_vocabulary, embedding_matrix,
                   pre_embedding_matrix=pre_embedding_matrix, suf_embedding_matrix=suf_embedding_matrix,
                   char_embedding=char_embedding)
    word_window_idx = torch.tensor([window for window, tag in windows])

    # the windows were generated according to the new pretrained vocab (plus missing from train) if we use pretrained.
    word_window_idx_dev = torch.tensor([window for window, tag in windows_dev])
    dataset = TensorDataset(word_window_idx, tags_idx)
    dev_dataset = TensorDataset(word_window_idx_dev, tags_idx_dev)

    if task == "ner":
        if embed == "pre":
            if sub_word_method == "all_word":
                epochs = 10
                lr = 0.0003
            elif sub_word_method == "char_word":
                epochs = 10
                lr = 0.001
            else:
                epochs = 10
                lr = 0.0002
        else:
            if sub_word_method == "all_word":
                epochs = 10
                lr = 0.00015
            elif sub_word_method == "char_word":
                epochs = 10
                lr = 0.0001
            else:
                epochs = 10
                lr = 0.00009
    else:
        if embed == "pre":
            if sub_word_method == "all_word":
                lr = 0.00035
                epochs = 8
            elif sub_word_method == "char_word":
                epochs = 10
                lr = 0.0003
            else:
                lr = 0.0003
                epochs = 8
        else:
            if sub_word_method == "all_word":
                lr = 0.0002
                epochs = 6
            elif sub_word_method == "char_word":
                epochs = 10
                lr = 0.0001
            else:
                lr = 0.00007
                epochs = 10

    # train model
    dev_loss, dev_accuracy, dev_accuracy_no_o = train_model(model, input_data=dataset,
                                                            dev_data=dev_dataset, tags_idx_dict=tags_idx_dict,
                                                            epochs=epochs, lr=lr)

    # plot the results
    plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task, embed, sub_word_method)

    print("Test")

    if sub_word_method == "sub_word":
        _, windows_test, _, _, _, all_test_words, _, _, _, _, _ = \
            read_tagged_file(f"./{task}/test", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary,
                             pre_vocab=pre_vocab, suf_vocab=suf_vocab, file_type="test", subword_method=sub_word_method)
    elif sub_word_method == "char_word":
        _, windows_test, _, _, _, all_test_words, _, _, _, _, _ = \
            read_tagged_file(f"./{task}/test", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary,
                             file_type="test", subword_method=sub_word_method, char_vocab=char_vocab)
    else:
        _, windows_test, _, _, _, all_test_words, _, _, _, _, _ = \
            read_tagged_file(f"./{task}/test", words_vocabulary=words_vocabulary, tags_vocabulary=tags_vocabulary,
                             file_type="test")

    word_window_idx_test = torch.tensor([window for window, tag in windows_test])
    test_dataset = TensorDataset(word_window_idx_test, torch.tensor([0] * len(word_window_idx_test)))
    idx_tags_dict = {i: tag for i, tag in enumerate(tags_vocabulary)}

    create_test_predictions(model, test_dataset, task, idx_tags_dict, all_test_words, embed, sub_word_method)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] in ["ner", "pos"] and sys.argv[2] in ["pre", "rand"] \
            and sys.argv[3] in ["sub_word", "all_word", "char_word"]:
        run(sys.argv[1], sys.argv[2], sys.argv[3])

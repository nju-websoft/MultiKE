import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

from pytorch.kgs import read_kgs_from_folder
from pytorch.utils import l2_normalize, generate_unlisted_word2vec, get_optimizer
from pytorch.utils import load_args, read_local_name, clear_attribute_triples, read_word2vec


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=None, activ='', normalize=False):
        super(AutoEncoder, self).__init__()
        self.normalize = normalize

        if hidden_dims is None:
            hidden_dims = [1024, 512]
        cfg = [input_dim] + hidden_dims + [output_dim]

        self.encoder = self._make_layers(cfg, activ)
        self.decoder = self._make_layers(cfg[::-1], activ)

        self._init_parameters()

    def _init_parameters(self):
        for param in self.parameters():
            nn.init.normal_(param, std=1.0)

    @staticmethod
    def _make_layers(cfg, activ):
        layers = []
        for i in range(len(cfg) - 1):
            layers += [nn.Linear(cfg[i], cfg[i + 1], bias=True)]
            if activ == 'sigmoid':
                layers += [nn.Sigmoid()]
            elif activ == 'tanh':
                layers += [nn.Tanh()]
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        if self.normalize is not None:
            z = l2_normalize(z, dim=1)
        x = self.decoder(z)
        return x


def encode_literals(args, literal_list, word2vec, tokens_max_len=5, word2vec_dim=300):
    word2vec_unlisted = generate_unlisted_word2vec(word2vec, literal_list)
    literal_vector_list = []
    for literal in literal_list:
        vectors = np.zeros((tokens_max_len, word2vec_dim), dtype=np.float32)
        words = literal.split(' ')
        for i in range(min(tokens_max_len, len(words))):
            if words[i] in word2vec_unlisted:
                vectors[i] = word2vec_unlisted[words[i]]
        literal_vector_list.append(vectors)
    literal_vector_list = np.stack(literal_vector_list).reshape(len(literal_vector_list), -1)
    assert len(literal_list) == len(literal_vector_list)

    word_vec_norm_list = preprocessing.normalize(literal_vector_list, norm='l2', axis=1, copy=False)
    dataset = TensorDataset(torch.from_numpy(word_vec_norm_list))
    # FIXME: Check with the original implementation that training data is not shuffled.
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)

    device = torch.device(args.device)
    model = AutoEncoder(literal_vector_list.shape[1], args.embed_dim, activ=args.encoder_activ, normalize=args.encoder_normalize)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    total = 0
    running_loss = 0.0

    model.train()
    for i in range(args.encoder_epochs):
        start_time = time.time()
        for inputs in dataloader:
            inputs = inputs[0].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

        end_time = time.time()
        print('epoch {} of literal encoder, loss: {:.4f}, time: {:.4f}s'.format(i + 1, running_loss / total, end_time - start_time))

    print('encode literal embeddings...', len(dataset))
    # FIXME: Check with the original implementation that does not perform normalization for encoding.
    # dataset = TensorDataset(torch.from_numpy(literal_vector_list))
    dataloader = DataLoader(dataset, 2 * args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    encoded_literal_vector = []

    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0].to(device)

            outputs = model.encoder(inputs)
            encoded_literal_vector.append(outputs.cpu())

    return torch.cat(encoded_literal_vector, dim=0).numpy()


def save_literal_vectors(folder, literal_list, literal_vectors):
    np.save(os.path.join(folder, 'literal_vectors.npy'), literal_vectors)
    assert len(literal_list) == len(literal_vectors)
    with open(os.path.join(folder, 'literals.txt'), 'w', encoding='utf-8') as file:
        for literal in literal_list:
            file.write(literal + '\n')
    print('literals and embeddings are saved in', folder)


def literal_vectors_exists(folder):
    literal_vectors_path = os.path.join(folder, 'literal_vectors.npy')
    literal_path = os.path.join(folder, 'literals.txt')
    return os.path.exists(literal_vectors_path) and os.path.exists(literal_path)


def load_literal_vectors(folder):
    print('load literal embeddings from', folder)
    literal_list = []
    literal_vectors = np.load(os.path.join(folder, 'literal_vectors.npy'))
    with open(os.path.join(folder, 'literals.txt'), 'r', encoding='utf-8') as file:
        for line in file:
            literal = line.strip('\n')
            literal_list.append(literal)
    return literal_list, literal_vectors

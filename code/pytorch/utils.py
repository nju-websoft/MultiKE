import os
import time
import json

import numpy as np
import torch
from gensim.models.word2vec import Word2Vec


def load_args(file_path):
    with open(file_path, 'r') as f:
        args_dict = json.load(f)
        f.close()
    print("load arguments:", args_dict)
    args = ARGs(args_dict)
    return args


class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    path = params[-1]
    folder = out_folder + method_name + '/' + path + "/" + div_path + str(time.strftime("%Y%m%d%H%M%S")) + "/"
    print("results output folder:", folder)
    return folder


def dict2file(file, dic):
    if dic is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in dic.items():
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()
    print(file, "saved.")


def save_embeddings(folder, kgs, ent_embeds, nv_ent_embeds, rv_ent_embeds, av_ent_embeds, rel_embeds, attr_embeds):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if ent_embeds is not None:
        np.save(folder + 'ent_embeds.npy', ent_embeds)
    if ent_embeds is not None:
        np.save(folder + 'nv_ent_embeds.npy', nv_ent_embeds)
    if ent_embeds is not None:
        np.save(folder + 'rv_ent_embeds.npy', rv_ent_embeds)
    if ent_embeds is not None:
        np.save(folder + 'av_ent_embeds.npy', av_ent_embeds)
    if rel_embeds is not None:
        np.save(folder + 'rel_embeds.npy', rel_embeds)
    if attr_embeds is not None:
        np.save(folder + 'attr_embeds.npy', attr_embeds)
    dict2file(folder + 'kg1_ent_ids', kgs.kg1.entities_id_dict)
    dict2file(folder + 'kg2_ent_ids', kgs.kg2.entities_id_dict)
    dict2file(folder + 'kg1_rel_ids', kgs.kg1.relations_id_dict)
    dict2file(folder + 'kg2_rel_ids', kgs.kg2.relations_id_dict)
    dict2file(folder + 'kg1_attr_ids', kgs.kg1.attributes_id_dict)
    dict2file(folder + 'kg2_attr_ids', kgs.kg2.attributes_id_dict)
    print("Embeddings saved!")


def read_word2vec(file_path, vector_dimension=300):
    print('\n', file_path)
    word2vec = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split(' ')
            if len(line) != vector_dimension + 1:
                continue
            v = np.array(list(map(float, line[1:])), dtype=np.float32)
            word2vec[line[0]] = v
    file.close()
    return word2vec


def read_local_name(folder_path, entities_set_1, entities_set_2):
    entity_local_name_1 = read_local_name_file(folder_path + 'entity_local_name_1', entities_set_1)
    entity_local_name_2 = read_local_name_file(folder_path + 'entity_local_name_2', entities_set_2)
    entity_local_name = entity_local_name_1
    entity_local_name.update(entity_local_name_2)
    print("total local names:", len(entity_local_name))
    return entity_local_name


def read_local_name_file(file_path, entities_set):
    print('read local names from', file_path)
    entity_local_name = dict()
    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            assert len(line) == 2
            if line[1] == '':
                cnt += 1
            ln = line[1]
            if ln.endswith(')'):
                ln = ln.split('(')[0]
            entity_local_name[line[0]] = ln.replace('_', ' ')
    file.close()

    for e in entities_set:
        if e not in entity_local_name:
            entity_local_name[e] = ''
    assert len(entity_local_name) == len(entities_set)
    return entity_local_name


def generate_word2vec_by_character_embedding(word_list, vector_dimension=300):
    character_vectors = {}
    alphabet = ''
    ch_num = {}
    for word in word_list:
        for ch in word:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n
    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            alphabet += ch_num[i][0]
    # print(alphabet)
    # print('len(alphabet):', len(alphabet), '\n')
    char_sequences = [list(word) for word in word_list]
    model = Word2Vec(char_sequences, size=vector_dimension, window=5, min_count=1)
    # model.save('char_embeddings.vec')
    for ch in alphabet:
        assert ch in model
        character_vectors[ch] = model[ch]

    word2vec = {}
    for word in word_list:
        vec = np.zeros(vector_dimension, dtype=np.float32)
        for ch in word:
            if ch in alphabet:
                vec += character_vectors[ch]
        if len(word) != 0:
            word2vec[word] = vec / len(word)
    return word2vec


def generate_unlisted_word2vec(word2vec, literal_list):
    unlisted_words = []
    for literal in literal_list:
        words = literal.split(' ')
        for word in words:
            if word not in word2vec:
                unlisted_words.append(word)
    word2vec_char = generate_word2vec_by_character_embedding(unlisted_words)
    word2vec.update(word2vec_char)
    return word2vec


def look_up_word2vec(id_tokens_dict, word2vec, tokens2vec_mode='add', keep_unlist=False, vector_dimension=300, tokens_max_len=5):
    if tokens2vec_mode == 'add':
        return tokens2vec_add(id_tokens_dict, word2vec, vector_dimension, keep_unlist)
    else:
        return tokens2vec_encoder(id_tokens_dict, word2vec, vector_dimension, tokens_max_len, keep_unlist)


def tokens2vec_encoder(id_tokens_dict, word2vec, vector_dimension, tokens_max_len, keep_unlist):
    tokens_vectors_dict = {}
    for v_id, tokens in id_tokens_dict.items():
        words = tokens.split(' ')
        vectors = np.zeros((tokens_max_len, vector_dimension), dtype=np.float32)
        flag = False
        for i in range(min(tokens_max_len, len(words))):
            if words[i] in word2vec:
                vectors[i] = word2vec[words[i]]
                flag = True
        if flag:
            tokens_vectors_dict[v_id] = vectors
    if keep_unlist:
        for v_id, _ in id_tokens_dict.items():
            if v_id not in tokens_vectors_dict:
                tokens_vectors_dict[v_id] = np.zeros((tokens_max_len, vector_dimension), dtype=np.float32)
    return tokens_vectors_dict


def tokens2vec_add(id_tokens_dict, word2vec, vector_dimension, keep_unlist):
    tokens_vectors_dict = {}
    cnt = 0
    for e_id, local_name in id_tokens_dict.items():
        words = local_name.split(' ')
        vec_sum = np.zeros(vector_dimension, dtype=np.float32)
        for word in words:
            if word in word2vec:
                vec_sum += word2vec[word]
        if sum(vec_sum) != 0:
            vec_sum = vec_sum / np.linalg(vec_sum)
        elif not keep_unlist:
            cnt += 1
            continue
        tokens_vectors_dict[e_id] = vec_sum
    # print('clear_unlisted_value:', cnt)
    return tokens_vectors_dict


def look_up_char2vec(id_tokens_dict, character_vectors, vector_dimension=300):
    tokens_vectors_dict = {}
    for e_id, ln in id_tokens_dict.items():
        vec_sum = np.zeros(vector_dimension, dtype=np.float32)
        for ch in ln:
            if ch in character_vectors:
                vec_sum += character_vectors[ch]
        if sum(vec_sum) != 0:
            vec_sum = vec_sum / np.linalg(vec_sum)
        tokens_vectors_dict[e_id] = vec_sum
    return tokens_vectors_dict


def clear_attribute_triples(attribute_triples):
    print('\nbefore clear:', len(attribute_triples))
    # step 1
    attribute_triples_new = set()
    attr_num = {}
    for (e, a, _) in attribute_triples:
        ent_num = 1
        if a in attr_num:
            ent_num += attr_num[a]
        attr_num[a] = ent_num
    attr_set = set(attr_num.keys())
    attr_set_new = set()
    for a in attr_set:
        if attr_num[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 1:', len(attribute_triples))

    # step 2
    attribute_triples_new = []
    literals_number, literals_string = [], []
    for (e, a, v) in attribute_triples:
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if is_number(v):
            literals_number.append(v)
        else:
            literals_string.append(v)
        v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue
        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 2:', len(attribute_triples))
    return attribute_triples, literals_number, literals_string


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


class L2Normalize(torch.nn.Module):

    def __init__(self, eps=1e-12):
        super(L2Normalize, self).__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, x, dim=None):
        if dim is None:
            norm = torch.sqrt(torch.max(torch.sum(x ** 2), self.eps)).expand_as(x)
        else:
            norm = torch.sqrt(torch.max(torch.sum(x ** 2, dim), self.eps)).unsqueeze(dim)
        return x / norm


def get_optimizer(opt, model, learning_rate):
    if opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return optimizer

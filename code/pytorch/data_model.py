import gc
import time
import math
import random
import multiprocessing

import numpy as np
import Levenshtein
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from base.kgs import read_kgs_from_folder
from pytorch.literal_encoder import encode_literals, literal_vectors_exists, load_literal_vectors, save_literal_vectors
from pytorch.utils import task_divide, merge_dic, read_local_name, clear_attribute_triples, read_word2vec, l2_normalize, load_args


def generate_sup_attribute_triples(sup_links, av_dict1, av_dict2):
    def generate_sup_attribute_triples_one_link(e1, e2, av_dict):
        new_triples = set()
        for a, v in av_dict.get(e1, set()):
            new_triples.add((e2, a, v))
        return new_triples
    new_triples1, new_triples2 = set(), set()
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_attribute_triples_one_link(ent1, ent2, av_dict1))
        new_triples2 |= (generate_sup_attribute_triples_one_link(ent2, ent1, av_dict2))
    print("supervised attribute triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2


def generate_dict(literal_list, literal_vectors_list):
    assert len(literal_list) == len(literal_vectors_list)
    dic = {}
    for i in range(len(literal_list)):
        dic[literal_list[i]] = literal_vectors_list[i]
    return dic


def generate_literal_id_dic(literal_list):
    literal_id_dic = {}
    print('literal id', len(literal_list), len(set(literal_list)))
    for i in range(len(literal_list)):
        literal_id_dic[literal_list[i]] = i
    assert len(literal_list) == len(literal_id_dic)
    return literal_id_dic


def link2dic(links):
    dic1, dic2 = {}, {}
    for i, j, w in links:
        dic1[i] = (j, w)
        dic2[j] = (i, w)
    assert len(dic1) == len(dic2)
    return dic1, dic2


def generate_sup_predicate_triples(predicate_links, triples1, triples2):
    link_dic1, link_dic2 = link2dic(predicate_links)
    sup_triples1, sup_triples2 = set(), set()
    for s, p, o in triples1:
        if p in link_dic1:
            sup_triples1.add((s, link_dic1.get(p)[0], o, link_dic1.get(p)[1]))
    for s, p, o in triples2:
        if p in link_dic2:
            sup_triples2.add((s, link_dic2.get(p)[0], o, link_dic2.get(p)[1]))
    return list(sup_triples1), list(sup_triples2)


def add_weights(predicate_links, triples1, triples2, min_w_before):
    link_dic1, link_dic2 = link2dic(predicate_links)
    weighted_triples1, weighted_triples2 = set(), set()
    w = 0.2
    for s, p, o in triples1:
        if p in link_dic1:
            weighted_triples1.add((s, p, o, zoom_weight(link_dic1.get(p)[1], min_w_before)))
        else:
            weighted_triples1.add((s, p, o, w))
    for s, p, o in triples2:
        if p in link_dic2:
            weighted_triples2.add((s, p, o, zoom_weight(link_dic2.get(p)[1], min_w_before)))
        else:
            weighted_triples2.add((s, p, o, w))
    assert len(triples1) == len(weighted_triples1)
    assert len(triples2) == len(weighted_triples2)
    return list(weighted_triples1), list(weighted_triples2), weighted_triples1, weighted_triples2


def init_predicate_alignment(predicate_local_name_dict_1, predicate_local_name_dict_2, predicate_init_sim):
    def get_predicate_match_dict(p_ln_dict_1, p_ln_dict_2):
        predicate_match_dict, sim_dict = {}, {}
        for p1, ln1 in p_ln_dict_1.items():
            match_p2 = ''
            max_sim = 0
            for p2, ln2 in p_ln_dict_2.items():
                sim_p2 = Levenshtein.ratio(ln1, ln2)
                if sim_p2 > max_sim:
                    match_p2 = p2
                    max_sim = sim_p2
            predicate_match_dict[p1] = match_p2
            sim_dict[p1] = max_sim
        return predicate_match_dict, sim_dict

    match_dict_1_2, sim_dict_1 = get_predicate_match_dict(predicate_local_name_dict_1, predicate_local_name_dict_2)
    match_dict_2_1, sim_dict_2 = get_predicate_match_dict(predicate_local_name_dict_2, predicate_local_name_dict_1)

    predicate_match_pairs_set = set()
    predicate_latent_match_pairs_similarity_dict = {}
    for p1, p2 in match_dict_1_2.items():
        if match_dict_2_1[p2] == p1:
            predicate_latent_match_pairs_similarity_dict[(p1, p2)] = sim_dict_1[p1]
            if sim_dict_1[p1] > predicate_init_sim:
                predicate_match_pairs_set.add((p1, p2, sim_dict_1[p1]))
                # print(p1, p2, sim_dict_1[p1], sim_dict_2[p2])
    return predicate_match_pairs_set, predicate_latent_match_pairs_similarity_dict


def read_predicate_local_name_file(file_path, relation_set):
    relation_local_name_dict, attribute_local_name_dict = {}, {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            assert len(line) == 2
            if line[0] in relation_set:
                relation_local_name_dict[line[0]] = line[1]
            else:
                attribute_local_name_dict[line[0]] = line[1]
    file.close()
    return relation_local_name_dict, attribute_local_name_dict


def predicate2id_matched_pairs(predicate_match_pairs_set, predicate_id_dict_1, predicate_id_dict_2):
    id_match_pairs_set = set()
    for (p1, p2, w) in predicate_match_pairs_set:
        if p1 in predicate_id_dict_1 and p2 in predicate_id_dict_2:
            id_match_pairs_set.add((predicate_id_dict_1[p1], predicate_id_dict_2[p2], w))
    return id_match_pairs_set


def find_predicate_alignment_by_embedding(embed, predicate_list1, predicate_list2, predicate_id_dict1, predicate_id_dict2):
    embed = preprocessing.normalize(embed)
    sim_mat = np.matmul(embed, embed.T)

    matched_1, matched_2 = {}, {}
    for i in predicate_list1:
        sorted_sim = (-sim_mat[i, :]).argsort()
        for j in sorted_sim:
            if j in predicate_list2:
                matched_1[i] = j
                break
    for j in predicate_list2:
        sorted_sim = (-sim_mat[j, :]).argsort()
        for i in sorted_sim:
            if i in predicate_list1:
                matched_2[j] = i
                break

    id_attr_dict1, id_attr_dict2 = {}, {}
    for a, i in predicate_id_dict1.items():
        id_attr_dict1[i] = a
    for a, i in predicate_id_dict2.items():
        id_attr_dict2[i] = a

    predicate_latent_match_pairs_similarity_dict = {}
    for i, j in matched_1.items():
        if matched_2[j] == i:
            predicate_latent_match_pairs_similarity_dict[(i, j)] = sim_mat[i, j]
    return predicate_latent_match_pairs_similarity_dict


def zoom_weight(weight, min_w_before, min_w_after=0.5):
    weight_new = 1.0 - (1.0 - weight) * (1.0 - min_w_after) / (1.0 - min_w_before)
    return weight_new


def _generate_neighbours(entity_embeds, entity_list, neighbors_num, threads_num):
    ent_frags = task_divide(entity_list, threads_num)
    ent_frag_indexes = task_divide(np.arange(0, entity_list.shape[0]), threads_num)

    pool = multiprocessing.Pool(processes=len(ent_frags))
    results = []
    for i in range(len(ent_frags)):
        results.append(pool.apply_async(find_neighbours, args=(ent_frags[i], entity_list, entity_embeds[ent_frag_indexes[i], :], entity_embeds, neighbors_num)))
    pool.close()
    pool.join()

    dic = {}
    for res in results:
        dic = merge_dic(dic, res.get())

    del results
    gc.collect()
    return dic


def find_neighbours(frags, entity_list, sub_embed, embed, k):
    dic = {}
    sim_mat = np.matmul(sub_embed, embed.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        neighbors_index = sort_index[0:k]
        neighbors = entity_list[neighbors_index].tolist()
        dic[frags[i]] = neighbors
    return dic


class DataModel:
    def __init__(self, args):
        self.args = args
        self.kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, False)
        self.word2vec_path = args.word2vec_path
        self.entity_local_name_dict = read_local_name(args.training_data, set(self.kgs.kg1.entities_id_dict.keys()),
                                                      set(self.kgs.kg2.entities_id_dict.keys()))
        self._generate_literal_vectors()
        self._generate_name_vectors()
        self._generate_attribute_value_vectors()

        self.relation_name_dict1, self.attribute_name_dict1 = read_predicate_local_name_file(
            args.training_data + 'predicate_local_name_1', set(self.kgs.kg1.relations_id_dict.keys()))
        self.relation_name_dict2, self.attribute_name_dict2 = read_predicate_local_name_file(
            args.training_data + 'predicate_local_name_2', set(self.kgs.kg2.relations_id_dict.keys()))

        self.relation_id_alignment_set = None
        # self.train_relations1, self.train_relations2 = None, None

        self.attribute_id_alignment_set = None
        # self.train_attributes1, self.train_attributes2 = None, None

        self.relation_alignment_set, self.relation_latent_match_pairs_similarity_dict_init = \
            init_predicate_alignment(self.relation_name_dict1, self.relation_name_dict2, args.predicate_init_sim)
        self.attribute_alignment_set, self.attribute_latent_match_pairs_similarity_dict_init = \
            init_predicate_alignment(self.attribute_name_dict1, self.attribute_name_dict2, args.predicate_init_sim)
        self.relation_alignment_set_init = self.relation_alignment_set
        self.attribute_alignment_set_init = self.attribute_alignment_set
        self.update_relation_triples(self.relation_alignment_set)
        self.update_attribute_triples(self.attribute_alignment_set)
        self.neighbors1, self.neighbors2 = {}, {}
        self.dataloader1 = DataLoader(TensorDataset(torch.from_numpy(self.kgs.all_entities[:, 0])), self.args.batch_size,
                                shuffle=False, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        self.dataloader2 = DataLoader(TensorDataset(torch.from_numpy(self.kgs.all_entities[:, 1])), self.args.batch_size,
                                shuffle=False, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)

    def _generate_literal_vectors(self):
        if not self.args.retrain_literal_embeds and literal_vectors_exists(self.args.training_data):
            self.literal_list, self.literal_vectors = load_literal_vectors(self.args.training_data)
        else:
            cleaned_attribute_triples_list1, _, _ = clear_attribute_triples(self.kgs.kg1.local_attribute_triples_list)
            cleaned_attribute_triples_list2, _, _ = clear_attribute_triples(self.kgs.kg2.local_attribute_triples_list)
            value_list = [v for (_, _, v) in cleaned_attribute_triples_list1 + cleaned_attribute_triples_list2]
            local_name_list = list(self.entity_local_name_dict.values())
            self.literal_list = list(set(value_list + local_name_list))
            print('literal num:', len(local_name_list), len(value_list), len(self.literal_list))
            word2vec = read_word2vec(self.args.word2vec_path)
            self.literal_vectors = encode_literals(self.args, self.literal_list, word2vec)
            save_literal_vectors(self.args.training_data, self.literal_list, self.literal_vectors)
            assert self.literal_vectors.shape[0] == len(self.literal_list)
        self.literal_id_dic = generate_literal_id_dic(self.literal_list)

    def _generate_name_vectors(self):
        name_ordered_list = []
        print("total entities:", self.kgs.num_entities)
        entity_id_uris_dic = dict(zip(self.kgs.kg1.entities_id_dict.values(), self.kgs.kg1.entities_id_dict.keys()))
        entity_id_uris_dic2 = dict(zip(self.kgs.kg2.entities_id_dict.values(), self.kgs.kg2.entities_id_dict.keys()))
        entity_id_uris_dic.update(entity_id_uris_dic2)
        print('total entities ids:', len(entity_id_uris_dic))
        assert len(entity_id_uris_dic) == self.kgs.num_entities
        for i in range(self.kgs.num_entities):
            assert i in entity_id_uris_dic
            entity_uri = entity_id_uris_dic.get(i)
            assert entity_uri in self.entity_local_name_dict
            entity_name = self.entity_local_name_dict.get(entity_uri)
            entity_name_index = self.literal_id_dic.get(entity_name)
            name_ordered_list.append(entity_name_index)
        print('name_ordered_list', len(name_ordered_list))
        name_vectors = self.literal_vectors[name_ordered_list, ]
        print("entity name embeddings:", type(name_vectors), name_vectors.shape)
        if self.args.literal_normalize:
            name_vectors = preprocessing.normalize(name_vectors)
        self.local_name_vectors = name_vectors

    def _generate_attribute_value_vectors(self):
        self.literal_set = set(self.literal_list)
        values_set = set()
        cleaned_attribute_triples_list1, _, _ = clear_attribute_triples(self.kgs.kg1.local_attribute_triples_list)
        cleaned_attribute_triples_list2, _, _ = clear_attribute_triples(self.kgs.kg2.local_attribute_triples_list)
        attribute_triples_list1, attribute_triples_list2 = set(), set()
        for h, a, v in cleaned_attribute_triples_list1:
            if v in self.literal_set:
                values_set.add(v)
                attribute_triples_list1.add((h, a, v))

        for h, a, v in cleaned_attribute_triples_list2:
            if v in self.literal_set:
                values_set.add(v)
                attribute_triples_list2.add((h, a, v))
        print("selected attribute triples", len(attribute_triples_list1), len(attribute_triples_list2))
        values_id_dic = {}
        values_list = list(values_set)
        num = len(values_list)
        for i in range(num):
            values_id_dic[values_list[i]] = i
        id_attribute_triples1 = set([(h, a, int(values_id_dic[v])) for (h, a, v) in attribute_triples_list1])
        id_attribute_triples2 = set([(h, a, int(values_id_dic[v])) for (h, a, v) in attribute_triples_list2])
        self.kgs.kg1.set_attributes(id_attribute_triples1)
        self.kgs.kg2.set_attributes(id_attribute_triples2)
        sup_triples1, sup_triples2 = generate_sup_attribute_triples(self.kgs.train_links, self.kgs.kg1.av_dict, self.kgs.kg2.av_dict)
        self.kgs.kg1.add_sup_attribute_triples(sup_triples1)
        self.kgs.kg2.add_sup_attribute_triples(sup_triples2)
        num = len(values_id_dic)
        value_ordered_list = []
        for i in range(num):
            value = values_list[i]
            value_index = self.literal_id_dic.get(value)
            value_ordered_list.append(value_index)
        print('value_ordered_list', len(value_ordered_list))
        value_vectors = self.literal_vectors[value_ordered_list, ]
        print("value embeddings:", type(value_vectors), value_vectors.shape)
        if self.args.literal_normalize:
            value_vectors = preprocessing.normalize(value_vectors)
        self.value_vectors = value_vectors

    def update_attribute_triples(self, attribute_alignment_set):
        self.attribute_id_alignment_set = predicate2id_matched_pairs(attribute_alignment_set,
                                                                     self.kgs.kg1.attributes_id_dict,
                                                                     self.kgs.kg2.attributes_id_dict)
        # self.train_attributes1 = [a for (a, _, _) in self.attribute_id_alignment_set]
        # self.train_attributes2 = [a for (_, a, _) in self.attribute_id_alignment_set]
        self.kgs.kg1.sup_attribute_alignment_triples, self.kgs.kg2.sup_attribute_alignment_triples = \
            generate_sup_predicate_triples(self.attribute_id_alignment_set, self.kgs.kg1.local_attribute_triples_list,
                                           self.kgs.kg2.local_attribute_triples_list)
        self.kgs.kg1.attribute_triples_w_weights, self.kgs.kg2.attribute_triples_w_weights, self.kgs.kg1.attribute_triples_w_weights_set, \
        self.kgs.kg2.attribute_triples_w_weights_set = add_weights(self.attribute_id_alignment_set,
                                                            self.kgs.kg1.local_attribute_triples_list,
                                                            self.kgs.kg2.local_attribute_triples_list,
                                                            self.args.predicate_soft_sim)

    def update_relation_triples(self, relation_alignment_set):
        self.relation_id_alignment_set = predicate2id_matched_pairs(relation_alignment_set,
                                                                    self.kgs.kg1.relations_id_dict,
                                                                    self.kgs.kg2.relations_id_dict)
        # self.train_relations1 = [a for (a, _, _) in self.relation_id_alignment_set]
        # self.train_relations2 = [a for (_, a, _) in self.relation_id_alignment_set]
        self.kgs.kg1.sup_relation_alignment_triples, self.kgs.kg2.sup_relation_alignment_triples = \
            generate_sup_predicate_triples(self.relation_id_alignment_set, self.kgs.kg1.local_relation_triples_list,
                                           self.kgs.kg2.local_relation_triples_list)
        self.kgs.kg1.relation_triples_w_weights, self.kgs.kg2.relation_triples_w_weights, self.kgs.kg1.relation_triples_w_weights_set, \
        self.kgs.kg1.relation_triples_w_weights_set = add_weights(self.relation_id_alignment_set,
                                                           self.kgs.kg1.local_relation_triples_list,
                                                           self.kgs.kg2.local_relation_triples_list,
                                                           self.args.predicate_soft_sim)

    def _update_predicate_alignment(self, embed, predicate_type='relation', w=0.7):
        if predicate_type == 'relation':
            predicate_list1, predicate_list2 = self.kgs.kg1.relations_list, self.kgs.kg2.relations_list
            predicate_id_dict1, predicate_id_dict2 = self.kgs.kg1.relations_id_dict, self.kgs.kg2.relations_id_dict
            predicate_alignment_set_init = self.relation_alignment_set_init
        else:
            predicate_list1, predicate_list2 = self.kgs.kg1.attributes_list, self.kgs.kg2.attributes_list
            predicate_id_dict1, predicate_id_dict2 = self.kgs.kg1.attributes_id_dict, self.kgs.kg2.attributes_id_dict
            predicate_alignment_set_init = self.attribute_alignment_set_init

        predicate_latent_match_pairs_similarity_dict = \
            find_predicate_alignment_by_embedding(embed, predicate_list1, predicate_list2, predicate_id_dict1,
                                                  predicate_id_dict2)

        predicate_alignment_set = set()
        for (p1, p2, sim_init) in predicate_alignment_set_init:
            p_id_1 = predicate_id_dict1[p1]
            p_id_2 = predicate_id_dict2[p2]
            sim = sim_init
            if (p_id_1, p_id_2) in predicate_latent_match_pairs_similarity_dict:
                sim = w * sim + (1 - w) * predicate_latent_match_pairs_similarity_dict[(p_id_1, p_id_2)]
            if sim > self.args.predicate_soft_sim:
                predicate_alignment_set.add((p1, p2, sim))
        print('update ' + predicate_type + ' alignment:', len(predicate_alignment_set))

        if predicate_type == 'relation':
            self.relation_alignment_set = predicate_alignment_set
            self.update_relation_triples(predicate_alignment_set)
        else:
            self.attribute_alignment_set = predicate_alignment_set
            self.update_attribute_triples(predicate_alignment_set)

    def update_predicate_alignment(self, model):
        rel_embeds = l2_normalize(model.rel_embeds.detach()).cpu().numpy()
        self._update_predicate_alignment(rel_embeds)
        attr_embeds = model.attr_embeds.detach().cpu().numpy()
        self._update_predicate_alignment(attr_embeds, predicate_type='attribute')

    def generate_neighbours(self, model, truncated_epsilon):
        start_time = time.time()
        neighbors_num1 = int((1 - truncated_epsilon) * self.kgs.kg1.entities_num)
        neighbors_num2 = int((1 - truncated_epsilon) * self.kgs.kg2.entities_num)

        entity_embeds1 = model.embeds(model, self.dataloader1)
        self.neighbors1 = _generate_neighbours(entity_embeds1, self.kgs.all_entities[:, 0], neighbors_num1, self.args.batch_threads_num)
        entity_embeds2 = model.embeds(model, self.dataloader2)
        self.neighbors2 = _generate_neighbours(entity_embeds2, self.kgs.all_entities[:, 1], neighbors_num2, self.args.batch_threads_num)
        ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
        end_time = time.time()
        print('neighbor dict:', len(self.neighbors1), type(self.neighbors2))
        print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, end_time - start_time))


class TrainDataset(Dataset):

    def __init__(self, data_model, batch_size, view, num_neg_triples=0):
        super(TrainDataset, self).__init__()
        self.data_model = data_model
        self.batch_size = batch_size
        self.view = view
        self.num_neg_triples = num_neg_triples
        if view == 'ckgatv':
            self.kg1 = self.data_model.kgs.kg1.sup_attribute_triples_list
            self.kg2 = self.data_model.kgs.kg2.sup_attribute_triples_list
        elif view == 'ckgrtv':
            self.kg1 = self.data_model.kgs.kg1.sup_relation_triples_list
            self.kg2 = self.data_model.kgs.kg2.sup_relation_triples_list
        elif view in ['cnv', 'mv']:
            self.kg1 = self.data_model.kgs.kg1.entities_list
            self.kg2 = self.data_model.kgs.kg2.entities_list
        elif view in ['ckgrrv', 'ckgarv', 'rv', 'av']:
            self.kg1 = self.data_model.kgs.kg1
            self.kg2 = self.data_model.kgs.kg2

        self.regenerate()

    def regenerate(self):
        if self.view not in ['rv', 'av']:
            if self.view == 'ckgrrv':
                total = len(self.kg1.sup_relation_alignment_triples) + len(self.kg2.sup_relation_alignment_triples)
            elif self.view == 'ckgarv':
                total = len(self.kg1.sup_attribute_alignment_triples) + len(self.kg2.sup_attribute_alignment_triples)
            else:
                total = len(self.kg1) + len(self.kg2)
            steps = int(math.ceil(total / self.batch_size))
            batch_size = self.batch_size if steps > 1 else total
            self.indices = [idx for _ in range(steps) for idx in random.sample(range(total), batch_size)]
            # map(l.extend, [random.sample(range(total), batch_size) for _ in range(steps)])
            # itertools.chain.from_iterable([random.sample(range(total), batch_size) for _ in range(steps)])
        else:
            if self.view == 'rv':
                kg1_len = len(self.kg1.local_relation_triples_list)
                kg2_len = len(self.kg2.local_relation_triples_list)
            else:  # 'av'
                kg1_len = len(self.kg1.attribute_triples_w_weights)
                kg2_len = len(self.kg2.attribute_triples_w_weights)
            total = kg1_len + kg2_len
            steps = int(math.ceil(total / self.batch_size))
            batch_size1 = int(kg1_len / total * self.batch_size)
            batch_size2 = self.batch_size - batch_size1
            kg1_indices = list(range(kg1_len))
            kg2_indices = list(range(kg1_len, kg1_len + kg2_len))
            random.shuffle(kg1_indices)
            random.shuffle(kg2_indices)
            self.indices = []
            for i in range(steps):
                self.indices += kg1_indices[i * batch_size1:(i + 1) * batch_size1]
                self.indices += kg2_indices[i * batch_size2:(i + 1) * batch_size2]

    def get_negs_fast(self, pos, triples_set, entities, neighbors, max_try=10):
        head, relation, tail = pos
        neg_triples = []
        nums_to_sample = self.num_neg_triples
        head_candidates = neighbors.get(head, entities)
        tail_candidates = neighbors.get(tail, entities)
        for i in range(max_try):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            if i == max_try - 1:
                neg_triples += list(i_neg_triples)
                break
            else:
                i_neg_triples = list(i_neg_triples - triples_set)
                neg_triples += i_neg_triples
            if len(neg_triples) == self.num_neg_triples:
                break
            else:
                nums_to_sample = self.num_neg_triples - len(neg_triples)
        assert len(neg_triples) == self.num_neg_triples
        return neg_triples

    def get_negs(self, pos, triples_set, entities, neighbors, max_try=10):
        neg_triples = []
        head, relation, tail  = pos
        head_candidates = neighbors.get(head, entities)
        tail_candidates = neighbors.get(tail, entities)
        for i in range(self.num_neg_triples):
            n = 0
            while True:
                corrupt_head_prob = np.random.binomial(1, 0.5)
                neg_head = head
                neg_tail = tail
                if corrupt_head_prob:
                    neg_head = random.choice(head_candidates)
                else:
                    neg_tail = random.choice(tail_candidates)
                if (neg_head, relation, neg_tail) not in triples_set:
                    neg_triples.append((neg_head, relation, neg_tail))
                    break
                n += 1
                if n == max_try:
                    neg_tail = random.choice(entities)
                    neg_triples.append((head, relation, neg_tail))
                    break
        assert len(neg_triples) == self.num_neg_triples
        return neg_triples

    def gen_negs_attr(self, pos, triples_set, entities, neighbors):
        neg_triples = []
        head, attribute, value, w = pos
        for i in range(self.num_neg_triples):
            while True:
                neg_head = random.choice(neighbors.get(head, entities))
                if (neg_head, attribute, value, w) not in triples_set:
                    break
            neg_triples.append((neg_head, attribute, value, w))
        assert len(neg_triples) == self.num_neg_triples
        return neg_triples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        if self.view not in ['rv', 'av']:
            if self.view not in ['ckgrrv', 'ckgarv']:
                kg1_len = len(self.kg1)
                pos = self.kg2[idx - kg1_len] if idx >= kg1_len else self.kg1[idx]
            else:
                kg1_len = len(self.kg1.sup_relation_alignment_triples if self.view == 'ckgrrv' else self.kg1.sup_attribute_alignment_triples)
                kg = self.kg2 if idx >= kg1_len else self.kg1
                if self.view == 'ckgrrv':
                    pos = kg.sup_relation_alignment_triples[idx - kg1_len if idx >= kg1_len else idx]
                else:
                    pos = kg.sup_attribute_alignment_triples[idx - kg1_len if idx >= kg1_len else idx]
        else:
            kg1_len = len(self.kg1.local_relation_triples_list if self.view == 'rv' else self.kg1.attribute_triples_w_weights)
            kg = self.kg2 if idx >= kg1_len else self.kg1
            neighbors = self.data_model.neighbors2 if idx >= kg1_len else self.data_model.neighbors1
            if self.view == 'rv':
                pos = kg.local_relation_triples_list[idx - kg1_len if idx >= kg1_len else idx]
                negs = self.get_negs_fast(pos, kg.local_relation_triples_set, kg.entities_list, neighbors)
                nhs = [x[0] for x in negs]
                nrs = [x[1] for x in negs]
                nts = [x[2] for x in negs]
                return list(pos) + [nhs, nrs, nts], []
            else:
                pos = kg.attribute_triples_w_weights[idx - kg1_len if idx >= kg1_len else idx]
                # negs = self.gen_negs_attr(pos, kg.attribute_triples_w_weights_set, kg.entities_list, neighbors)

        inputs = pos[:3] if self.view not in ['cnv', 'mv'] else [pos]
        weights = []
        if self.view in ['ckgarv', 'ckgrrv', 'av']:
            weights.append(pos[3])
        return inputs, weights


class TestDataset(Dataset):

    def __init__(self, kg1_entities, kg2_entities):
        super(TestDataset, self).__init__()
        self.kg1 = kg1_entities
        self.kg2 = kg2_entities

    def __len__(self):
        return len(self.kg1) + len(self.kg2)

    def __getitem__(self, index):
        kg1_len = len(self.kg1)
        inputs = self.kg2[index - kg1_len] if index >= kg1_len else self.kg1[index]
        return inputs


if __name__ == '__main__':
    args = load_args('./pytorch/args.json')
    data = DataModel(args)

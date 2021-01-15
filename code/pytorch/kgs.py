import os
import numpy as np

from base.kg import KG
from base.read import *


class KGs:
    def __init__(self, kg1: KG, kg2: KG, train_links, valid_links, test_links=None, mode='mapping', ordered=True):
        if mode == "sharing":
            ent_ids1, ent_ids2 = generate_sharing_id(train_links, kg1.relation_triples_set, kg1.entities_set,
                                                     kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_sharing_id([], kg1.relation_triples_set, kg1.relations_set,
                                                     kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_sharing_id([], kg1.attribute_triples_set, kg1.attributes_set,
                                                       kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)
        else:
            ent_ids1, ent_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.entities_set,
                                                     kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.relations_set,
                                                     kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_mapping_id(kg1.attribute_triples_set, kg1.attributes_set,
                                                       kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)

        id_relation_triples1 = uris_relation_triple_2ids(kg1.relation_triples_set, ent_ids1, rel_ids1)
        id_relation_triples2 = uris_relation_triple_2ids(kg2.relation_triples_set, ent_ids2, rel_ids2)

        id_attribute_triples1 = uris_attribute_triple_2ids(kg1.attribute_triples_set, ent_ids1, attr_ids1)
        id_attribute_triples2 = uris_attribute_triple_2ids(kg2.attribute_triples_set, ent_ids2, attr_ids2)

        kg1 = KG(id_relation_triples1, id_attribute_triples1)
        kg2 = KG(id_relation_triples2, id_attribute_triples2)
        kg1.set_id_dict(ent_ids1, rel_ids1, attr_ids1)
        kg2.set_id_dict(ent_ids2, rel_ids2, attr_ids2)

        train_links = uris_pair_2ids(train_links, ent_ids1, ent_ids2)
        valid_links = uris_pair_2ids(valid_links, ent_ids1, ent_ids2)
        test_links = uris_pair_2ids(test_links, ent_ids1, ent_ids2) if test_links is not None else []

        if mode == 'swapping':
            sup_triples1, sup_triples2 = generate_sup_relation_triples(train_links,
                                                                       kg1.rt_dict, kg1.hr_dict,
                                                                       kg2.rt_dict, kg2.hr_dict)
            kg1.add_sup_relation_triples(sup_triples1)
            kg2.add_sup_relation_triples(sup_triples2)

            sup_triples1, sup_triples2 = generate_sup_attribute_triples(train_links, kg1.av_dict, kg2.av_dict)
            kg1.add_sup_attribute_triples(sup_triples1)
            kg2.add_sup_attribute_triples(sup_triples2)

        self.kg1 = kg1
        self.kg2 = kg2

        self.num_train_entities = len(train_links)
        self.num_valid_entities = len(valid_links)
        self.num_test_entities = len(test_links)
        self.all_entities = np.array(train_links + valid_links + test_links, dtype=np.int32)

        self.num_entities = len(self.kg1.entities_set | self.kg2.entities_set)
        self.num_relations = len(self.kg1.relations_set | self.kg2.relations_set)
        self.num_attributes = len(self.kg1.attributes_set | self.kg2.attributes_set)

    def get_entities(self, split, kg=None):
        if split == 'train':
            entities = self.all_entities[:self.num_train_entities]
        elif split == 'valid':
            entities = self.all_entities[self.num_train_entities:self.num_train_entities + self.num_valid_entities]
        elif split == 'test':
            entities = self.all_entities[-self.num_test_entities:]
        else:  # valid and test
            entities = self.all_entities[self.num_train_entities:]
        return entities if kg is None else entities[:, kg - 1]

def read_kgs_from_folder(training_data_folder, division, mode, ordered):
    kg1_relation_triples, _, _ = read_relation_triples(os.path.join(training_data_folder, 'rel_triples_1'))
    kg2_relation_triples, _, _ = read_relation_triples(os.path.join(training_data_folder, 'rel_triples_2'))
    kg1_attribute_triples, _, _ = read_attribute_triples(os.path.join(training_data_folder, 'attr_triples_1'))
    kg2_attribute_triples, _, _ = read_attribute_triples(os.path.join(training_data_folder, 'attr_triples_2'))

    train_links = read_links(os.path.join(training_data_folder, division, 'train_links'))
    valid_links = read_links(os.path.join(training_data_folder, division, 'valid_links'))
    test_links = read_links(os.path.join(training_data_folder, division, 'test_links'))

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, valid_links, test_links=test_links, mode=mode, ordered=ordered)
    return kgs


def read_kgs_from_files(kg1_relation_triples, kg2_relation_triples, kg1_attribute_triples, kg2_attribute_triples,
                        train_links, valid_links, test_links, mode):
    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, valid_links, test_links=test_links, mode=mode)
    return kgs

from base.kgs import read_kgs_from_folder
from literal_encoder import *
from utils import *
from sklearn import preprocessing


SPLIT = '\t'
LITERAL_EMBEDDINGS_FILE = 'literal_vectors.npy'
LITERAL_FILE = 'literals.txt'


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


def save_literal_vectors(folder, literal_list, literal_vectors):
    np.save(folder + LITERAL_EMBEDDINGS_FILE, literal_vectors)
    assert len(literal_list) == len(literal_vectors)
    with open(folder + LITERAL_FILE, 'w', encoding='utf-8') as file:
        for l in literal_list:
            file.write(l + '\n')
    print('literals and embeddings are saved in', folder)
    file.close()


def load_literal_vectors(folder):
    print('load literal embeddings from', folder)
    literal_list = list()
    mat = np.load(folder + LITERAL_EMBEDDINGS_FILE)
    with open(folder + LITERAL_FILE, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            literal_list.append(line)
    file.close()
    return literal_list, np.matrix(mat)


def generate_dict(literal_list, literal_vectors_list):
    assert len(literal_list) == len(literal_vectors_list)
    dic = dict()
    list()
    for i in range(len(literal_list)):
        dic[literal_list[i]] = literal_vectors_list[i]
    return dic


def generate_literal_id_dic(literal_list):
    literal_id_dic = dict()
    print('literal id', len(literal_list), len(set(literal_list)))
    for i in range(len(literal_list)):
        literal_id_dic[literal_list[i]] = i
    assert len(literal_list) == len(literal_id_dic)
    return literal_id_dic


class DataModel:
    def __init__(self, args):
        self.args = args
        self.session = load_session()
        self.kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, False)
        self.entities = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        self.word2vec_path = args.word2vec_path
        self.entity_local_name_dict = read_local_name(args.training_data, set(self.kgs.kg1.entities_id_dict.keys()),
                                                      set(self.kgs.kg2.entities_id_dict.keys()))
        self._generate_literal_vectors()
        self._generate_name_vectors_mat()
        self._generate_attribute_value_vectors()

    def _generate_literal_vectors(self):
        file_path = self.args.training_data + LITERAL_EMBEDDINGS_FILE
        if not self.args.retrain_literal_embeds and os.path.exists(file_path):
            self.literal_list, self.literal_vectors_mat = load_literal_vectors(self.args.training_data)
        else:
            cleaned_attribute_triples_list1, _, _ = clear_attribute_triples(self.kgs.kg1.local_attribute_triples_list)
            cleaned_attribute_triples_list2, _, _ = clear_attribute_triples(self.kgs.kg2.local_attribute_triples_list)
            value_list = [v for (_, _, v) in cleaned_attribute_triples_list1 + cleaned_attribute_triples_list2]
            local_name_list = list(self.entity_local_name_dict.values())
            self.literal_list = list(set(value_list + local_name_list))
            print('literal num:', len(local_name_list), len(value_list), len(self.literal_list))
            word2vec = read_word2vec(self.word2vec_path)
            literal_encoder = LiteralEncoder(self.literal_list, word2vec, self.args)
            self.literal_vectors_mat = literal_encoder.encoded_literal_vector
            save_literal_vectors(self.args.training_data, self.literal_list, self.literal_vectors_mat)
            assert self.literal_vectors_mat.shape[0] == len(self.literal_list)
        self.literal_id_dic = generate_literal_id_dic(self.literal_list)

    def _generate_name_vectors_mat(self):
        name_ordered_list = list()
        num = len(self.entities)
        print("total entities:", num)
        entity_id_uris_dic = dict(zip(self.kgs.kg1.entities_id_dict.values(), self.kgs.kg1.entities_id_dict.keys()))
        entity_id_uris_dic2 = dict(zip(self.kgs.kg2.entities_id_dict.values(), self.kgs.kg2.entities_id_dict.keys()))
        entity_id_uris_dic.update(entity_id_uris_dic2)
        print('total entities ids:', len(entity_id_uris_dic))
        assert len(entity_id_uris_dic) == num
        for i in range(num):
            assert i in entity_id_uris_dic
            entity_uri = entity_id_uris_dic.get(i)
            assert entity_uri in self.entity_local_name_dict
            entity_name = self.entity_local_name_dict.get(entity_uri)
            entity_name_index = self.literal_id_dic.get(entity_name)
            name_ordered_list.append(entity_name_index)
        print('name_ordered_list', len(name_ordered_list))
        name_mat = self.literal_vectors_mat[name_ordered_list, ]
        print("entity name embeddings mat:", type(name_mat), name_mat.shape)
        if self.args.literal_normalize:
            name_mat = preprocessing.normalize(name_mat)
        self.local_name_vectors = name_mat

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
        values_id_dic = dict()
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
        value_ordered_list = list()
        for i in range(num):
            value = values_list[i]
            value_index = self.literal_id_dic.get(value)
            value_ordered_list.append(value_index)
        print('value_ordered_list', len(value_ordered_list))
        value_vectors = self.literal_vectors_mat[value_ordered_list, ]
        print("value embeddings mat:", type(value_vectors), value_vectors.shape)
        if self.args.literal_normalize:
            value_vectors = preprocessing.normalize(value_vectors)
        self.value_vectors = value_vectors





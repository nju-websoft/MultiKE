import math
import gc
import multiprocessing as mp
from sklearn import preprocessing

import base.batch as bat
from utils import *
import base.evaluation as eva
from data_model import DataModel
from MultiKE_model import MultiKE
from predicate_alignment import PredicateAlignModel


def valid(model, embed_choice='avg', w=(1, 1, 1)):
    if embed_choice == 'nv':
        ent_embeds = model.name_embeds.eval(session=model.session)
    elif embed_choice == 'rv':
        ent_embeds = model.rv_ent_embeds.eval(session=model.session)
    elif embed_choice == 'av':
        ent_embeds = model.av_ent_embeds.eval(session=model.session)
    elif embed_choice == 'final':
        ent_embeds = model.ent_embeds.eval(session=model.session)
    elif embed_choice == 'avg':
        ent_embeds = w[0] * model.name_embeds.eval(session=model.session) + \
                     w[1] * model.rv_ent_embeds.eval(session=model.session) + \
                     w[2] * model.av_ent_embeds.eval(session=model.session)
    else:  # 'final'
        ent_embeds = model.ent_embeds
    print(embed_choice, 'valid results:')
    embeds1 = ent_embeds[model.kgs.valid_entities1,]
    embeds2 = ent_embeds[model.kgs.valid_entities2 + model.kgs.test_entities2,]
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)
    del embeds1, embeds2
    gc.collect()
    return mrr_12


def test(model, embed_choice='avg', w=(1, 1, 1)):
    if embed_choice == 'nv':
        ent_embeds = model.name_embeds.eval(session=model.session)
    elif embed_choice == 'rv':
        ent_embeds = model.rv_ent_embeds.eval(session=model.session)
    elif embed_choice == 'av':
        ent_embeds = model.av_ent_embeds.eval(session=model.session)
    elif embed_choice == 'final':
        ent_embeds = model.ent_embeds.eval(session=model.session)
    elif embed_choice == 'avg':
        ent_embeds = w[0] * model.name_embeds.eval(session=model.session) + \
                     w[1] * model.rv_ent_embeds.eval(session=model.session) + \
                     w[2] * model.av_ent_embeds.eval(session=model.session)
    else:  # wavg
        ent_embeds = model.ent_embeds
    print(embed_choice, 'test results:')
    embeds1 = ent_embeds[model.kgs.test_entities1,]
    embeds2 = ent_embeds[model.kgs.test_entities2,]
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)
    del embeds1, embeds2
    gc.collect()
    return mrr_12


def _compute_weight(embeds1, embeds2, embeds3):
    def min_max_normalization(mat):
        min_ = np.min(mat)
        max_ = np.max(mat)
        return (mat - min_) / (max_ - min_)

    other_embeds = (embeds1 + embeds2 + embeds3) / 3
    # other_embeds = (embeds2 + embeds3) / 2
    other_embeds = preprocessing.normalize(other_embeds)
    embeds1 = preprocessing.normalize(embeds1)
    # sim_mat = sim(embeds1, other_embeds, metric='cosine')
    sim_mat = np.matmul(embeds1, other_embeds.T)
    # sim_mat = 1 - euclidean_distances(embeds1, other_embeds)
    weights = np.diag(sim_mat)
    # print(weights.shape, np.mean(weights))
    # weights = min_max_normalization(weights)
    print(weights.shape, np.mean(weights))
    return np.mean(weights)


def wva(embeds1, embeds2, embeds3):
    weight1 = _compute_weight(embeds1, embeds2, embeds3)
    weight2 = _compute_weight(embeds2, embeds1, embeds3)
    weight3 = _compute_weight(embeds3, embeds1, embeds2)
    return weight1, weight2, weight3
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight
    print('final weights', weight1, weight2, weight3)
    ent_embeds = weight1 * embeds1 + \
                 weight2 * embeds2 + \
                 weight3 * embeds3
    return ent_embeds


def valid_WVA(model):
    nv_ent_embeds1 = tf.nn.embedding_lookup(model.name_embeds, model.kgs.valid_entities1).eval(session=model.session)
    rv_ent_embeds1 = tf.nn.embedding_lookup(model.rv_ent_embeds, model.kgs.valid_entities1).eval(session=model.session)
    av_ent_embeds1 = tf.nn.embedding_lookup(model.av_ent_embeds, model.kgs.valid_entities1).eval(session=model.session)
    weight11, weight21, weight31 = wva(nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1)

    test_list = model.kgs.valid_entities2 + model.kgs.test_entities2
    nv_ent_embeds2 = tf.nn.embedding_lookup(model.name_embeds, test_list).eval(session=model.session)
    rv_ent_embeds2 = tf.nn.embedding_lookup(model.rv_ent_embeds, test_list).eval(session=model.session)
    av_ent_embeds2 = tf.nn.embedding_lookup(model.av_ent_embeds, test_list).eval(session=model.session)
    weight12, weight22, weight32 = wva(nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2)

    weight1 = weight11 + weight12
    weight2 = weight21 + weight22
    weight3 = weight31 + weight32
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight

    print('weights', weight1, weight2, weight3)

    embeds1 = weight1 * nv_ent_embeds1 + \
              weight2 * rv_ent_embeds1 + \
              weight3 * av_ent_embeds1
    embeds2 = weight1 * nv_ent_embeds2 + \
              weight2 * rv_ent_embeds2 + \
              weight3 * av_ent_embeds2
    print('wvag valid results:')
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)

    del nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1
    del nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2
    del embeds1, embeds2
    gc.collect()

    return mrr_12


def test_WVA(model):
    nv_ent_embeds1 = tf.nn.embedding_lookup(model.name_embeds, model.kgs.test_entities1).eval(session=model.session)
    rv_ent_embeds1 = tf.nn.embedding_lookup(model.rv_ent_embeds, model.kgs.test_entities1).eval(session=model.session)
    av_ent_embeds1 = tf.nn.embedding_lookup(model.av_ent_embeds, model.kgs.test_entities1).eval(session=model.session)
    weight11, weight21, weight31 = wva(nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1)

    test_list = model.kgs.test_entities2
    nv_ent_embeds2 = tf.nn.embedding_lookup(model.name_embeds, test_list).eval(session=model.session)
    rv_ent_embeds2 = tf.nn.embedding_lookup(model.rv_ent_embeds, test_list).eval(session=model.session)
    av_ent_embeds2 = tf.nn.embedding_lookup(model.av_ent_embeds, test_list).eval(session=model.session)
    weight12, weight22, weight32 = wva(nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2)

    weight1 = weight11 + weight12
    weight2 = weight21 + weight22
    weight3 = weight31 + weight32
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight

    print('weights', weight1, weight2, weight3)

    embeds1 = weight1 * nv_ent_embeds1 + \
              weight2 * rv_ent_embeds1 + \
              weight3 * av_ent_embeds1
    embeds2 = weight1 * nv_ent_embeds2 + \
              weight2 * rv_ent_embeds2 + \
              weight3 * av_ent_embeds2
    print('wvag test results:')
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)
    del embeds1, embeds2
    gc.collect()
    return mrr_12


class MultiKE_Late(MultiKE):
    def __init__(self, data, args, attr_align_model):
        super().__init__(data, args, attr_align_model)

        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = False

        self._define_variables()

        self._define_name_view_graph()
        self._define_relation_view_graph()
        self._define_attribute_view_graph()

        self._define_cross_kg_entity_reference_relation_view_graph()
        self._define_cross_kg_entity_reference_attribute_view_graph()
        self._define_cross_kg_relation_reference_graph()
        self._define_cross_kg_attribute_reference_graph()

        self._define_common_space_learning_graph()
        self._define_space_mapping_graph()

        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def run(self):
        t = time.time()
        relation_triples_num = self.kgs.kg1.local_relation_triples_num + self.kgs.kg2.local_relation_triples_num
        attribute_triples_num = self.kgs.kg1.local_attribute_triples_num + self.kgs.kg2.local_attribute_triples_num
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        attribute_triple_steps = int(math.ceil(attribute_triples_num / self.args.batch_size))
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        attribute_step_tasks = task_divide(list(range(attribute_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        relation_batch_queue = manager.Queue()
        attribute_batch_queue = manager.Queue()
        cross_kg_relation_triples = self.kgs.kg1.sup_relation_triples_list + self.kgs.kg2.sup_relation_triples_list
        cross_kg_entity_inference_in_attribute_triples = self.kgs.kg1.sup_attribute_triples_list + \
                                                         self.kgs.kg2.sup_attribute_triples_list
        cross_kg_relation_inference = self.predicate_align_model.sup_relation_alignment_triples1 + \
                                      self.predicate_align_model.sup_relation_alignment_triples2
        cross_kg_attribute_inference = self.predicate_align_model.sup_attribute_alignment_triples1 + \
                                       self.predicate_align_model.sup_attribute_alignment_triples2
        neighbors1, neighbors2 = None, None

        entity_list = self.kgs.kg1.entities_list + self.kgs.kg2.entities_list

        valid(self, embed_choice='nv')
        valid(self, embed_choice='avg')
        for i in range(1, self.args.max_epoch + 1):
            print('epoch {}:'.format(i))
            self.train_relation_view_1epo(i, relation_triple_steps, relation_step_tasks,
                                          relation_batch_queue, neighbors1, neighbors2)
            self.train_cross_kg_entity_inference_relation_view_1epo(i, cross_kg_relation_triples)
            if i > self.args.start_predicate_soft_alignment:
                self.train_cross_kg_relation_inference_1epo(i, cross_kg_relation_inference)

            self.train_attribute_view_1epo(i, attribute_triple_steps, attribute_step_tasks, attribute_batch_queue,
                                           neighbors1, neighbors2)
            self.train_cross_kg_entity_inference_attribute_view_1epo(i, cross_kg_entity_inference_in_attribute_triples)
            if i > self.args.start_predicate_soft_alignment:
                self.train_cross_kg_attribute_inference_1epo(i, cross_kg_attribute_inference)

            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                valid(self, embed_choice='rv')
                valid(self, embed_choice='av')
                valid(self, embed_choice='avg')
                valid_WVA(self)
                if i >= self.args.start_predicate_soft_alignment:
                    self.predicate_align_model.update_predicate_alignment(self.rel_embeds.eval(session=self.session))
                    self.predicate_align_model.update_predicate_alignment(self.attr_embeds.eval(session=self.session),
                                                                          predicate_type='attribute')
                    cross_kg_relation_inference = self.predicate_align_model.sup_relation_alignment_triples1 + \
                                                  self.predicate_align_model.sup_relation_alignment_triples2
                    cross_kg_attribute_inference = self.predicate_align_model.sup_attribute_alignment_triples1 + \
                                                   self.predicate_align_model.sup_attribute_alignment_triples2

            if self.early_stop or i == self.args.max_epoch:
                break

            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
                t1 = time.time()
                assert 0.0 < self.args.truncated_epsilon < 1.0
                neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
                neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
                neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list1,
                                                     neighbors_num1, self.args.batch_threads_num)
                neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list2,
                                                     neighbors_num2, self.args.batch_threads_num)
                ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
                print('neighbor dict:', len(neighbors1), type(neighbors2))
                print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
        for i in range(1, self.args.shared_learning_max_epoch + 1):
            self.train_shared_space_mapping_1epo(i, entity_list)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                valid(self, embed_choice='final')
        self.save()
        test(self, embed_choice='nv')
        test(self, embed_choice='rv')
        test(self, embed_choice='av')
        test(self, embed_choice='avg')
        test_WVA(self)
        test(self, embed_choice='final')


if __name__ == '__main__':
    args = load_args('args.json')
    data = DataModel(args)
    attr_align_model = PredicateAlignModel(data.kgs, args)
    model = MultiKE_Late(data, args, attr_align_model)
    model.run()

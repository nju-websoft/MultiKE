import math
import time
import multiprocessing as mp
import tensorflow as tf

import openea.modules.train.batch as bat
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session
from openea.modules.utils.util import task_divide
from openea.modules.args.args_hander import load_args
from data_model import DataModel
from MultiKE_model import MultiKE
from predicate_alignment import PredicateAlignModel
from MultiKE_Late import valid, test


class MultiKE_CV(MultiKE):
    def __init__(self, data, args, predicate_align_model):
        super().__init__(data, args, predicate_align_model)
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, '', self.__class__.__name__)

        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = False

        self._define_variables()

        self._define_name_view_graph()
        self._define_relation_view_graph()
        self._define_attribute_view_graph()

        self._define_cross_kg_entity_reference_relation_view_graph()
        self._define_cross_kg_entity_reference_attribute_view_graph()
        self._define_cross_kg_attribute_reference_graph()
        self._define_cross_kg_relation_reference_graph()
        self._define_common_space_learning_graph()

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

        for i in range(1, self.args.max_epoch + 1):
            print('epoch {}:'.format(i))
            self.train_relation_view_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue,
                                          neighbors1, neighbors2)
            self.train_common_space_learning_1epo(i, entity_list)
            self.train_cross_kg_entity_inference_relation_view_1epo(i, cross_kg_relation_triples)
            if i > self.args.start_predicate_soft_alignment:
                self.train_cross_kg_relation_inference_1epo(i, cross_kg_relation_inference)

            self.train_attribute_view_1epo(i, attribute_triple_steps, attribute_step_tasks, attribute_batch_queue,
                                           neighbors1, neighbors2)
            self.train_common_space_learning_1epo(i, entity_list)
            self.train_cross_kg_entity_inference_attribute_view_1epo(i, cross_kg_entity_inference_in_attribute_triples)
            if i > self.args.start_predicate_soft_alignment:
                self.train_cross_kg_attribute_inference_1epo(i, cross_kg_attribute_inference)

            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                valid(self, embed_choice='rv')
                valid(self, embed_choice='av')
                valid(self, embed_choice='final')

                if self.early_stop or i == self.args.max_epoch:
                    break

            if i >= self.args.start_predicate_soft_alignment and i % 10 == 0:
                self.predicate_align_model.update_predicate_alignment(self.rel_embeds.eval(session=self.session))
                self.predicate_align_model.update_predicate_alignment(self.attr_embeds.eval(session=self.session),
                                                                      predicate_type='attribute')
                cross_kg_relation_inference = self.predicate_align_model.sup_relation_alignment_triples1 + \
                                              self.predicate_align_model.sup_relation_alignment_triples2
                cross_kg_attribute_inference = self.predicate_align_model.sup_attribute_alignment_triples1 + \
                                               self.predicate_align_model.sup_attribute_alignment_triples2

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
        self.save()
        test(self, embed_choice='nv')
        test(self, embed_choice='rv')
        test(self, embed_choice='av')
        test(self, embed_choice='final')


if __name__ == '__main__':
    args = load_args('args.json')
    data = DataModel(args)
    predicate_align_model = PredicateAlignModel(data.kgs, args)
    model = MultiKE_CV(data, args, predicate_align_model)
    model.run()

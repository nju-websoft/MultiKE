import math
import random
import multiprocessing as mp

import base.batch as bat
from utils import *
from base.initializers import xavier_init
from attr_batch import generate_attribute_triple_batch_queue
from utils import save_embeddings

from losses import relation_logistic_loss, attribute_logistic_loss, relation_logistic_loss_wo_negs, \
    attribute_logistic_loss_wo_negs, space_mapping_loss, alignment_loss, logistic_loss_wo_negs, orthogonal_loss


def get_optimizer(opt, learning_rate):
    if opt == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif opt == 'Adadelta':
        # To match the exact form in the original paper use 1.0.
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:  # opt == 'SGD'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer


def generate_optimizer(loss, learning_rate, var_list=None, opt='SGD'):
    optimizer = get_optimizer(opt, learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    return optimizer.apply_gradients(grads_and_vars)


def conv(attr_hs, attr_as, attr_vs, dim, feature_map_size=2, kernel_size=[2, 4], activation=tf.nn.tanh, layer_num=2):
    # print("feature map size", feature_map_size)
    # print("kernel size", kernel_size)
    # print("layer_num", layer_num)
    attr_as = tf.reshape(attr_as, [-1, 1, dim])
    attr_vs = tf.reshape(attr_vs, [-1, 1, dim])

    input_avs = tf.concat([attr_as, attr_vs], 1)
    input_shape = input_avs.shape.as_list()
    input_layer = tf.reshape(input_avs, [-1, input_shape[1], input_shape[2], 1])
    # print("input_layer", input_layer.shape)
    _conv = input_layer
    _conv = tf.layers.batch_normalization(_conv, 2)
    for i in range(layer_num):
        _conv = tf.layers.conv2d(inputs=_conv,
                                 filters=feature_map_size,
                                 kernel_size=kernel_size,
                                 strides=[1, 1],
                                 padding="same",
                                 activation=activation)
        # print("conv" + str(i + 1), _conv.shape)
    _conv = tf.nn.l2_normalize(_conv, 2)
    _shape = _conv.shape.as_list()
    _flat = tf.reshape(_conv, [-1, _shape[1] * _shape[2] * _shape[3]])
    # print("_flat", _flat.shape)
    dense = tf.layers.dense(inputs=_flat, units=dim, activation=activation)
    dense = tf.nn.l2_normalize(dense)  # important!!
    # print("dense", dense.shape)
    score = -tf.reduce_sum(tf.square(attr_hs - dense), 1)
    return score


class MultiKE:

    def __check_args(self):
        assert self.args.alignment_module == 'swapping'  # for cross-KG inference

    def __init__(self, data, args, attr_align_model):
        self.predicate_align_model = attr_align_model

        self.args = args
        self.__check_args()

        self.data = data
        self.kgs = kgs = data.kgs
        self.kg1 = kgs.kg1
        self.kg2 = kgs.kg2

        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, '', self.__class__.__name__)

        self.session = None

    def _define_variables(self):
        with tf.variable_scope('literal' + 'embeddings'):
            self.literal_embeds = tf.constant(self.data.value_vectors, dtype=tf.float32)
        with tf.variable_scope('name_view' + 'embeddings'):
            self.name_embeds = tf.constant(self.data.local_name_vectors, dtype=tf.float32)
        with tf.variable_scope('relation_view' + 'embeddings'):
            self.rv_ent_embeds = xavier_init([self.kgs.entities_num, self.args.dim], 'rv_ent_embeds', True)
            self.rel_embeds = xavier_init([self.kgs.relations_num, self.args.dim], 'rel_embeds', True)
        with tf.variable_scope('attribute_view' + 'embeddings'):
            self.av_ent_embeds = xavier_init([self.kgs.entities_num, self.args.dim], 'av_ent_embeds', True)
            # False important!
            self.attr_embeds = xavier_init([self.kgs.attributes_num, self.args.dim], 'attr_embeds', False)
        with tf.variable_scope('shared' + 'embeddings'):
            self.ent_embeds = xavier_init([self.kgs.entities_num, self.args.dim], 'ent_embeds', True)
        with tf.variable_scope('shared' + 'combination'):
            self.nv_mapping = tf.get_variable('nv_mapping', shape=[self.args.dim, self.args.dim],
                                              initializer=tf.initializers.orthogonal())
            self.rv_mapping = tf.get_variable('rv_mapping', shape=[self.args.dim, self.args.dim],
                                              initializer=tf.initializers.orthogonal())
            self.av_mapping = tf.get_variable('av_mapping', shape=[self.args.dim, self.args.dim],
                                              initializer=tf.initializers.orthogonal())
            self.eye_mat = tf.constant(np.eye(self.args.dim), dtype=tf.float32, name='eye')

    # --- The followings are view-specific embedding models --- #

    def _define_name_view_graph(self):
        pass

    def _define_relation_view_graph(self):
        with tf.name_scope('relation_triple_placeholder'):
            self.rel_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.rel_pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.rel_pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.rel_neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.rel_neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.rel_neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('relation_triple_lookup'):
            rel_phs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_pos_hs)
            rel_prs = tf.nn.embedding_lookup(self.rel_embeds, self.rel_pos_rs)
            rel_pts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_pos_ts)
            rel_nhs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_neg_hs)
            rel_nrs = tf.nn.embedding_lookup(self.rel_embeds, self.rel_neg_rs)
            rel_nts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_neg_ts)
        with tf.name_scope('relation_triple_loss'):
            self.relation_loss = relation_logistic_loss(rel_phs, rel_prs, rel_pts, rel_nhs, rel_nrs, rel_nts)
            self.relation_optimizer = generate_optimizer(self.relation_loss, self.args.learning_rate,
                                                         opt=self.args.optimizer)

    def _define_attribute_view_graph(self):
        with tf.name_scope('attribute_triple_placeholder'):
            self.attr_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.attr_pos_as = tf.placeholder(tf.int32, shape=[None])
            self.attr_pos_vs = tf.placeholder(tf.int32, shape=[None])
            self.attr_pos_ws = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('attribute_triple_lookup'):
            attr_phs = tf.nn.embedding_lookup(self.av_ent_embeds, self.attr_pos_hs)
            attr_pas = tf.nn.embedding_lookup(self.attr_embeds, self.attr_pos_as)
            attr_pvs = tf.nn.embedding_lookup(self.literal_embeds, self.attr_pos_vs)
        with tf.variable_scope('cnn'):
            pos_score = conv(attr_phs, attr_pas, attr_pvs, self.args.dim)
            pos_score = tf.log(1 + tf.exp(-pos_score))
            pos_score = tf.multiply(pos_score, self.attr_pos_ws)
            pos_loss = tf.reduce_sum(pos_score)
            self.attribute_loss = pos_loss
            self.attribute_optimizer = generate_optimizer(self.attribute_loss, self.args.learning_rate,
                                                          opt=self.args.optimizer)

    # --- The followings are cross-kg identity inference --- #

    def _define_cross_kg_name_view_graph(self):
        pass

    def _define_cross_kg_entity_reference_relation_view_graph(self):
        with tf.name_scope('cross_kg_relation_triple_placeholder'):
            self.ckge_rel_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.ckge_rel_pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.ckge_rel_pos_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('cross_kg_relation_triple_lookup'):
            ckge_rel_phs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.ckge_rel_pos_hs)
            ckge_rel_prs = tf.nn.embedding_lookup(self.rel_embeds, self.ckge_rel_pos_rs)
            ckge_rel_pts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.ckge_rel_pos_ts)
        with tf.name_scope('cross_kg_relation_triple_loss'):
            self.ckge_relation_loss = 2 * relation_logistic_loss_wo_negs(ckge_rel_phs, ckge_rel_prs, ckge_rel_pts)
            self.ckge_relation_optimizer = generate_optimizer(self.ckge_relation_loss, self.args.learning_rate,
                                                              opt=self.args.optimizer)

    def _define_cross_kg_entity_reference_attribute_view_graph(self):
        with tf.name_scope('cross_kg_attribute_triple_placeholder'):
            self.ckge_attr_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.ckge_attr_pos_as = tf.placeholder(tf.int32, shape=[None])
            self.ckge_attr_pos_vs = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('cross_kg_attribute_triple_lookup'):
            ckge_attr_phs = tf.nn.embedding_lookup(self.av_ent_embeds, self.ckge_attr_pos_hs)
            ckge_attr_pas = tf.nn.embedding_lookup(self.attr_embeds, self.ckge_attr_pos_as)
            ckge_attr_pvs = tf.nn.embedding_lookup(self.literal_embeds, self.ckge_attr_pos_vs)
        with tf.name_scope('cross_kg_attribute_triple_loss'):
            pos_score = conv(ckge_attr_phs, ckge_attr_pas, ckge_attr_pvs, self.args.dim)
            self.ckge_attribute_loss = 2 * tf.reduce_sum(tf.log(1 + tf.exp(-pos_score)))
            self.ckge_attribute_optimizer = generate_optimizer(self.ckge_attribute_loss, self.args.learning_rate,
                                                               opt=self.args.optimizer)

    def _define_cross_kg_relation_reference_graph(self):
        with tf.name_scope('cross_kg_relation_reference_placeholder'):
            self.ckgp_rel_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.ckgp_rel_pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.ckgp_rel_pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.ckgp_rel_pos_ws = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('cross_kg_relation_reference_lookup'):
            ckgp_rel_phs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.ckgp_rel_pos_hs)
            ckgp_rel_prs = tf.nn.embedding_lookup(self.rel_embeds, self.ckgp_rel_pos_rs)
            ckgp_rel_pts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.ckgp_rel_pos_ts)
        with tf.name_scope('cross_kg_relation_reference_loss'):
            self.ckgp_relation_loss = 2 * logistic_loss_wo_negs(ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts,
                                                                  self.ckgp_rel_pos_ws)
            self.ckgp_relation_optimizer = generate_optimizer(self.ckgp_relation_loss, self.args.learning_rate,
                                                              opt=self.args.optimizer)

    def _define_cross_kg_attribute_reference_graph(self):
        with tf.name_scope('cross_kg_attribute_reference_placeholder'):
            self.ckga_attr_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.ckga_attr_pos_as = tf.placeholder(tf.int32, shape=[None])
            self.ckga_attr_pos_vs = tf.placeholder(tf.int32, shape=[None])
            self.ckga_attr_pos_ws = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('cross_kg_attribute_reference_lookup'):
            ckga_attr_phs = tf.nn.embedding_lookup(self.av_ent_embeds, self.ckga_attr_pos_hs)
            ckga_attr_pas = tf.nn.embedding_lookup(self.attr_embeds, self.ckga_attr_pos_as)
            ckga_attr_pvs = tf.nn.embedding_lookup(self.literal_embeds, self.ckga_attr_pos_vs)
        with tf.name_scope('cross_kg_attribute_reference_loss'):
            pos_score = conv(ckga_attr_phs, ckga_attr_pas, ckga_attr_pvs, self.args.dim)
            pos_score = tf.log(1 + tf.exp(-pos_score))
            pos_score = tf.multiply(pos_score, self.ckga_attr_pos_ws)
            pos_loss = tf.reduce_sum(pos_score)
            self.ckga_attribute_loss = pos_loss
            # self.ckga_attribute_loss = tf.reduce_sum(tf.log(1 + tf.exp(-pos_score)))
            self.ckga_attribute_optimizer = generate_optimizer(self.ckga_attribute_loss, self.args.learning_rate,
                                                               opt=self.args.optimizer)

    # --- The followings are intermediate combination --- #

    def _define_common_space_learning_graph(self):
        with tf.name_scope('cross_name_view_placeholder'):
            self.cn_hs = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('cross_name_view_lookup'):
            final_cn_phs = tf.nn.embedding_lookup(self.ent_embeds, self.cn_hs)
            cn_hs_names = tf.nn.embedding_lookup(self.name_embeds, self.cn_hs)
            cr_hs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.cn_hs)
            ca_hs = tf.nn.embedding_lookup(self.av_ent_embeds, self.cn_hs)
        with tf.name_scope('cross_name_view_loss'):
            self.cross_name_loss = self.args.cv_name_weight * alignment_loss(final_cn_phs, cn_hs_names)
            self.cross_name_loss += alignment_loss(final_cn_phs, cr_hs)
            self.cross_name_loss += alignment_loss(final_cn_phs, ca_hs)
            self.cross_name_optimizer = generate_optimizer(self.args.cv_weight * self.cross_name_loss,
                                                           self.args.ITC_learning_rate,
                                                           opt=self.args.optimizer)

    def _define_space_mapping_graph(self):
        with tf.name_scope('final_entities_placeholder'):
            self.entities = tf.placeholder(tf.int32, shape=[self.args.entity_batch_size, ])
        with tf.name_scope('multi_view_entities_lookup'):
            final_ents = tf.nn.embedding_lookup(self.ent_embeds, self.entities)
            nv_ents = tf.nn.embedding_lookup(self.name_embeds, self.entities)
            rv_ents = tf.nn.embedding_lookup(self.rv_ent_embeds, self.entities)
            av_ents = tf.nn.embedding_lookup(self.av_ent_embeds, self.entities)
        with tf.name_scope('mapping_loss'):
            nv_space_mapping_loss = space_mapping_loss(nv_ents, final_ents, self.nv_mapping, self.eye_mat,
                                                       self.args.orthogonal_weight)
            rv_space_mapping_loss = space_mapping_loss(rv_ents, final_ents, self.rv_mapping, self.eye_mat,
                                                       self.args.orthogonal_weight)
            av_space_mapping_loss = space_mapping_loss(av_ents, final_ents, self.av_mapping, self.eye_mat,
                                                       self.args.orthogonal_weight)
            self.shared_comb_loss = nv_space_mapping_loss + rv_space_mapping_loss + av_space_mapping_loss
            opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("shared")]
            self.shared_comb_optimizer = generate_optimizer(self.shared_comb_loss,
                                                            self.args.learning_rate,
                                                            var_list=opt_vars,
                                                            opt=self.args.optimizer)

    def eval_kg1_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.rv_ent_embeds, self.kgs.kg1.entities_list)
        return embeds.eval(session=self.session)

    def eval_kg2_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.rv_ent_embeds, self.kgs.kg2.entities_list)
        return embeds.eval(session=self.session)

    def eval_kg1_useful_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.rv_ent_embeds, self.kgs.useful_entities_list1)
        return embeds.eval(session=self.session)

    def eval_kg2_useful_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.rv_ent_embeds, self.kgs.useful_entities_list2)
        return embeds.eval(session=self.session)

    def save(self):
        ent_embeds = self.ent_embeds.eval(session=self.session)
        nv_ent_embeds = self.name_embeds.eval(session=self.session)
        rv_ent_embeds = self.rv_ent_embeds.eval(session=self.session)
        av_ent_embeds = self.av_ent_embeds.eval(session=self.session)
        rel_embeds = self.rel_embeds.eval(session=self.session)
        att_embeds = self.rel_embeds.eval(session=self.session)
        save_embeddings(self.out_folder, self.kgs, ent_embeds, nv_ent_embeds, rv_ent_embeds, av_ent_embeds,
                        rel_embeds, att_embeds)

    # --- The followings are training for multi-view embeddings --- #

    def train_relation_view_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.local_relation_triples_list, self.kgs.kg2.local_relation_triples_list,
                             self.kgs.kg1.local_relation_triples_set, self.kgs.kg2.local_relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.relation_loss, self.relation_optimizer],
                                             feed_dict={self.rel_pos_hs: [x[0] for x in batch_pos],
                                                        self.rel_pos_rs: [x[1] for x in batch_pos],
                                                        self.rel_pos_ts: [x[2] for x in batch_pos],
                                                        self.rel_neg_hs: [x[0] for x in batch_neg],
                                                        self.rel_neg_rs: [x[1] for x in batch_neg],
                                                        self.rel_neg_ts: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.local_relation_triples_list)
        random.shuffle(self.kgs.kg2.local_relation_triples_list)
        end = time.time()
        print('epoch {} of rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))

    def train_attribute_view_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for steps_task in steps_tasks:
            mp.Process(target=generate_attribute_triple_batch_queue,
                       args=(self.predicate_align_model.attribute_triples_w_weights1,
                             self.predicate_align_model.attribute_triples_w_weights2,
                             self.predicate_align_model.attribute_triples_w_weights_set1,
                             self.predicate_align_model.attribute_triples_w_weights_set2,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.attribute_batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, 0)).start()
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.attribute_loss, self.attribute_optimizer],
                                             feed_dict={self.attr_pos_hs: [x[0] for x in batch_pos],
                                                        self.attr_pos_as: [x[1] for x in batch_pos],
                                                        self.attr_pos_vs: [x[2] for x in batch_pos],
                                                        self.attr_pos_ws: [x[3] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.predicate_align_model.attribute_triples_w_weights1)
        random.shuffle(self.predicate_align_model.attribute_triples_w_weights2)
        end = time.time()
        print('epoch {} of att. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))

    # --- The followings are training for cross-kg identity inference --- #

    def train_cross_kg_entity_inference_relation_view_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.batch_size))
        batch_size = self.args.batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.ckge_relation_loss, self.ckge_relation_optimizer],
                                             feed_dict={self.ckge_rel_pos_hs: [x[0] for x in batch_pos],
                                                        self.ckge_rel_pos_rs: [x[1] for x in batch_pos],
                                                        self.ckge_rel_pos_ts: [x[2] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg entity inference in rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                            epoch_loss,
                                                                                                            end - start))

    def train_cross_kg_entity_inference_attribute_view_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.attribute_batch_size))
        batch_size = self.args.attribute_batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.ckge_attribute_loss, self.ckge_attribute_optimizer],
                                             feed_dict={self.ckge_attr_pos_hs: [x[0] for x in batch_pos],
                                                        self.ckge_attr_pos_as: [x[1] for x in batch_pos],
                                                        self.ckge_attr_pos_vs: [x[2] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg entity inference in attr. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                             epoch_loss,
                                                                                                             end - start))

    def train_cross_kg_relation_inference_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.batch_size))
        batch_size = self.args.batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.ckgp_relation_loss, self.ckgp_relation_optimizer],
                                             feed_dict={self.ckgp_rel_pos_hs: [x[0] for x in batch_pos],
                                                        self.ckgp_rel_pos_rs: [x[1] for x in batch_pos],
                                                        self.ckgp_rel_pos_ts: [x[2] for x in batch_pos],
                                                        self.ckgp_rel_pos_ws: [x[3] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg relation inference in rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                              epoch_loss,
                                                                                                              end - start))

    def train_cross_kg_attribute_inference_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.attribute_batch_size))
        batch_size = self.args.attribute_batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.ckga_attribute_loss, self.ckga_attribute_optimizer],
                                             feed_dict={self.ckga_attr_pos_hs: [x[0] for x in batch_pos],
                                                        self.ckga_attr_pos_as: [x[1] for x in batch_pos],
                                                        self.ckga_attr_pos_vs: [x[2] for x in batch_pos],
                                                        self.ckga_attr_pos_ws: [x[3] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg attribute inference in attr. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                                epoch_loss,
                                                                                                                end - start))

    def train_shared_space_mapping_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.entity_batch_size))
        batch_size = self.args.entity_batch_size if steps > 1 else len(entities)
        for i in range(steps):
            batch_pos = random.sample(entities, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.shared_comb_loss, self.shared_comb_optimizer],
                                             feed_dict={self.entities: batch_pos})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of shared space learning, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                           end - start))

    # --- The followings are training for cross-view inference --- #

    def train_common_space_learning_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.entity_batch_size))
        batch_size = self.args.entity_batch_size if steps > 1 else len(entities)
        for i in range(steps):
            batch_pos = random.sample(entities, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.cross_name_loss, self.cross_name_optimizer],
                                             feed_dict={self.cn_hs: batch_pos})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of common space learning, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                           end - start))

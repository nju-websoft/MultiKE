import tensorflow as tf


def relation_logistic_loss(phs, prs, pts, nhs, nrs, nts):
    pos_distance = phs + prs - pts
    neg_distance = nhs + nrs - nts
    pos_score = -tf.reduce_sum(tf.square(pos_distance), axis=1)
    neg_score = -tf.reduce_sum(tf.square(neg_distance), axis=1)
    pos_loss = tf.reduce_sum(tf.log(1 + tf.exp(-pos_score)))
    neg_loss = tf.reduce_sum(tf.log(1 + tf.exp(neg_score)))
    loss = tf.add(pos_loss, neg_loss)
    return loss


def attribute_logistic_loss(phs, pas, pvs, pws, nhs, nas, nvs, nws):
    pos_distance = phs + pas - pvs
    neg_distance = nhs + nas - nvs
    pos_score = -tf.reduce_sum(tf.square(pos_distance), axis=1)
    neg_score = -tf.reduce_sum(tf.square(neg_distance), axis=1)
    pos_score = tf.log(1 + tf.exp(-pos_score))
    neg_score = tf.log(1 + tf.exp(neg_score))
    pos_score = tf.multiply(pos_score, pws)
    neg_score = tf.multiply(neg_score, nws)
    pos_loss = tf.reduce_sum(pos_score)
    neg_loss = tf.reduce_sum(neg_score)
    loss = tf.add(pos_loss, neg_loss)
    return loss


def relation_logistic_loss_wo_negs(phs, prs, pts):
    pos_distance = phs + prs - pts
    pos_score = -tf.reduce_sum(tf.square(pos_distance), axis=1)
    pos_loss = tf.reduce_sum(tf.log(1 + tf.exp(-pos_score)))
    return pos_loss


def attribute_logistic_loss_wo_negs(phs, pas, pvs):
    pos_distance = phs + pas - pvs
    pos_score = -tf.reduce_sum(tf.square(pos_distance), axis=1)
    pos_loss = tf.reduce_sum(tf.log(1 + tf.exp(-pos_score)))
    return pos_loss


def logistic_loss_wo_negs(phs, pas, pvs, pws):
    pos_distance = phs + pas - pvs
    pos_score = -tf.reduce_sum(tf.square(pos_distance), axis=1)
    pos_score = tf.log(1 + tf.exp(-pos_score))
    pos_score = tf.multiply(pos_score, pws)
    pos_loss = tf.reduce_sum(pos_score)
    return pos_loss


def space_mapping_loss(view_embeds, shared_embeds, mapping, eye, orthogonal_weight, norm_w=0.0001):
    mapped_ents2 = tf.matmul(view_embeds, mapping)
    mapped_ents2 = tf.nn.l2_normalize(mapped_ents2)
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.square(shared_embeds - mapped_ents2), 1))
    norm_loss = tf.reduce_sum(tf.reduce_sum(tf.square(mapping), 1))
    return map_loss + orthogonal_weight * orthogonal_loss(mapping, eye) + norm_w * norm_loss


def orthogonal_loss(mapping, eye):
    loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return loss


def alignment_loss(ents1, ents2):
    distance = ents1 - ents2
    loss = tf.reduce_sum(tf.reduce_sum(tf.square(distance), axis=1))
    return loss

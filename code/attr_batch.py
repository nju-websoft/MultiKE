import random


def generate_pos_triples(triples, batch_size, step):
    start = step * batch_size
    end = start + batch_size
    if end > len(triples):
        end = len(triples)
    pos_batch = triples[start: end]
    return pos_batch


def generate_neg_attribute_triples(pos_batch, all_triples_set, entity_list, neg_triples_num, neighbor=None):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, attribute, value, w in pos_batch:
        for i in range(neg_triples_num):
            while True:
                neg_head = random.choice(neighbor.get(head, entity_list))
                if (neg_head, attribute, value, w) not in all_triples_set:
                    break
            neg_batch.append((neg_head, attribute, value, w))
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch


def generate_attribute_triple_batch_queue(triple_list1, triple_list2, triple_set1, triple_set2,
                                          entity_list1, entity_list2, batch_size,
                                          steps, out_queue, neighbor1, neighbor2, neg_triples_num):
    for step in steps:
        pos_batch, neg_batch = generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                                               entity_list1, entity_list2, batch_size,
                                                               step, neighbor1, neighbor2, neg_triples_num)
        out_queue.put((pos_batch, neg_batch))
    exit(0)


def generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                    entity_list1, entity_list2, batch_size,
                                    step, neighbor1, neighbor2, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = generate_pos_triples(triple_list1, batch_size1, step)
    pos_batch2 = generate_pos_triples(triple_list2, batch_size2, step)
    neg_batch1 = generate_neg_attribute_triples(pos_batch1, triple_set1, entity_list1,
                                                neg_triples_num, neighbor=neighbor1)
    neg_batch2 = generate_neg_attribute_triples(pos_batch2, triple_set2, entity_list2,
                                                neg_triples_num, neighbor=neighbor2)
    return pos_batch1 + pos_batch2, neg_batch1 + neg_batch2

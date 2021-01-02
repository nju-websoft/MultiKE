import numpy as np

from pytorch.multike_model import MultiKENet


class MultiKELateNet(MultiKENet):
    def __init__(self, num_entities, num_relations, num_attributes, embed_dim, value_vectors, local_name_vectors):
        super(MultiKELateNet, self).__init__(num_entities, num_relations, num_attributes, embed_dim, value_vectors, local_name_vectors, True)


if __name__ == '__main__':
    model = MultiKELateNet(200000, 550, 1086, 75, np.zeros((909911, 75), dtype=np.float32), np.zeros((200000, 75), dtype=np.float32))
    # model.test(model, kgs)

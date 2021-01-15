import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from pytorch.utils import load_args, get_optimizer, generate_out_folder
from pytorch.data_model import DataModel, TrainDataset, TestDataset
from pytorch.multike_late import MultiKELateNet
from pytorch.losses import MultiKELoss


parser = argparse.ArgumentParser(description='run')
parser.add_argument('--training_data', type=str, default='')
parser_args = parser.parse_args()


if __name__ == '__main__':
    args = load_args('./pytorch/args.json')
    # args.training_data = parser_args.training_data
    data = DataModel(args)

    views = ['rv', 'ckgrtv', 'ckgrrv', 'av', 'ckgatv', 'ckgarv', 'mv']

    batch_sizes = [args.batch_size] * len(views)
    batch_sizes[3:6] = [args.attribute_batch_size] * 3
    batch_sizes[-1] = args.entity_batch_size
    train_datasets = [TrainDataset(data, bs, v) for bs, v in zip(batch_sizes, views)]
    train_datasets[0].num_neg_triples = args.neg_triple_num
    train_dataloaders = [DataLoader(ds, bs, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) for ds, bs in zip(train_datasets, batch_sizes)]
    valid_dataset = TestDataset(data.kgs.get_entities('valid', 1), data.kgs.get_entities('validtest', 2))
    test_dataset = TestDataset(data.kgs.get_entities('test', 1), data.kgs.get_entities('test', 2))
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    device = torch.device(args.device)
    model = MultiKELateNet(data.kgs.entities_num, data.kgs.relations_num, data.kgs.attributes_num, args.dim, data.value_vectors, data.local_name_vectors)
    model.to(device)

    lrs = [args.learning_rate] * len(views)
    criterion = MultiKELoss(args.cv_name_weight, args.cv_weight, args.orthogonal_weight, model.eye)
    optimizers = [get_optimizer(args.optimizer, model.parameters(v), lr) for v, lr in zip(views, lrs)]

    out_folder = generate_out_folder(args.output, args.training_data, '', model.__class__.__name__)
    early_stop = False

    # model.test(args, model, test_dataloader, embed_choice='nv')
    # model.test(args, model, test_dataloader, embed_choice='avg')

    for i in range(1, args.max_epoch + 1):
        print('epoch {}:'.format(i))

        for idx, view in enumerate(views[:-1]):
            if view in ['ckgrrv', 'ckgarv'] and i <= args.start_predicate_soft_alignment:
                continue

            total = 0
            running_loss = 0.0
            optimizer = optimizers[idx]
            train_dataloader = train_dataloaders[idx]
            model.train()
            start_time = time.time()
            for inputs, weights in train_dataloader:
                if view == 'rv':
                    inputs_pos = list(map(lambda x: x.to(device), inputs[:3]))
                    inputs_negs = list(map(lambda x: torch.cat(x, dim=0).to(device), inputs[3:]))
                    inputs = inputs_pos + inputs_negs
                else:
                    inputs = list(map(lambda x: x.long().to(device), inputs))
                weights = list(map(lambda x: x.float().to(device), weights))

                optimizer.zero_grad()
                outputs = model(inputs, view)
                loss = criterion(outputs, weights, view)
                loss.backward()
                optimizer.step()

                total += inputs[0].size(0)
                running_loss += loss.item() * inputs[0].size(0)

            end_time = time.time()
            print('epoch {} of {}, avg. loss: {:.4f}, time: {:.4f}s'.format(i, view, running_loss / total, end_time - start_time))

        # save checkpoint
        torch.save({
            'epoch': i,
            'model': model.state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers]
        }, os.path.join(out_folder, 'checkpoint.pth'))

        if i >= args.start_valid and i % args.eval_freq == 0:
            # model.test(args, model, valid_dataloader, embed_choice='rv')
            # model.test(args, model, valid_dataloader, embed_choice='av')
            # model.test(args, model, valid_dataloader, embed_choice='avg')
            # model.test_wva(args, model, valid_dataloader)
            if i >= args.start_predicate_soft_alignment and i % 1 == 0:
                data.update_predicate_alignment(model)

        if early_stop or i == args.max_epoch:
            break

        # if args.neg_sampling == 'truncated' and i % args.truncated_freq == 0:
        #     assert 0.0 < args.truncated_epsilon < 1.0
        #     data.generate_neighbours(model, args.truncated_epsilon)

        for ds in train_datasets[:-1]:
            ds.regenerate()

    view = views[-1]
    optimizer = optimizers[-1]
    train_dataloader = train_dataloaders[-1]
    train_dataset = train_datasets[-1]
    for i in range(1, args.shared_learning_max_epoch + 1):
        total = 0
        running_loss = 0.0
        model.train()
        start_time = time.time()
        for inputs, weights in train_dataloader:
            inputs = list(map(lambda x: x.long().to(device), inputs))
            weights = list(map(lambda x: x.float().to(device), weights))

            optimizer.zero_grad()
            outputs = model(inputs, view)
            loss = criterion(outputs, weights, view)
            loss.backward()
            optimizer.step()

            total += inputs[0].size(0)
            running_loss += loss.item() * inputs[0].size(0)

        end_time = time.time()
        print('epoch {} of {}, avg. loss: {:.4f}, time: {:.4f}s'.format(i, view, running_loss / total, end_time - start_time))

        # if i >= args.start_valid and i % args.eval_freq == 0:
        #     model.test(args, model, valid_dataloader, embed_choice='final')

        train_dataset.regenerate()

    # model.test(args, model, test_dataloader, embed_choice='nv')
    # model.test(args, model, test_dataloader, embed_choice='rv')
    # model.test(args, model, test_dataloader, embed_choice='av')
    # model.test(args, model, test_dataloader, embed_choice='avg')
    # model.test_wva(args, model, test_dataloader)
    # model.test(args, model, test_dataloader, embed_choice='final')

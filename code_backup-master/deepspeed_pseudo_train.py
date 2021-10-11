import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from block_dataloader import generate_dataloader
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import deepspeed
import random
from graphsage_model import SAGE

from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.local_rank >= 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True


def ttt(tic, str1):
    toc = time.time()
    print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
    return toc


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(data_loader, model_engine, val_g, val_nfeat, val_labels, val_nid, args, timers, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """

    acc_res = 0

    # Turn on evaluation mode which disables dropout.
    model_engine.eval()
    with torch.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(data_loader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(val_nfeat, val_labels, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            # Compute loss and prediction
            batch_pred = model_engine(blocks, batch_inputs)
            # print(batch_pred.device)
            # print(type(batch_pred))
            # print(batch_pred.shape)
            # print(batch_labels.device)
            # print(type(batch_labels))
            # print(batch_labels.shape)
            # print('cur_acc = compute_acc(batch_pred[val_nid], batch_labels[val_nid])')
            cur_acc = compute_acc(batch_pred, batch_labels)
            # print(cur_acc)
            acc_res += cur_acc

    # Move model back to the train mode.
    model_engine.train()

    return acc_res / len(data_loader)


# def evaluate(model, g, nfeat, labels, val_nid, device):
# 	"""
# 	Evaluate the model on the validation set specified by ``val_nid``.
# 	g : The entire graph.
# 	inputs : The features of all the nodes.
# 	labels : The labels of all the nodes.
# 	val_nid : the node Ids for validation.
# 	device : The GPU device to evaluate on.
# 	"""
# 	model.eval()
# 	with torch.no_grad():
# 		pred = model.inference(g, nfeat, device, args)
# 	model.train()
# 	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    print('batch_inputs device')
    print(batch_inputs.device)
    return batch_inputs, batch_labels


def load_blocks_subtensor(g, labels, blocks, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = g.ndata['features'][blocks[0].srcdata[dgl.NID].tolist()].to(device)
    batch_labels = blocks[-1].dstdata['labels'].to(device)
    # print('batch_inputs device')
    # print(batch_inputs.device)
    return batch_inputs, batch_labels


#### Entry point
def run(args, device, data, tic):
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data

    in_feats = train_nfeat.shape[1]
    train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    dataloader_device = torch.device('cpu')

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    full_batch_size = len(train_nid)
    full_batch_dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=dataloader_device,
        batch_size=full_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    net = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)


    parameters = filter(lambda p: p.requires_grad, net.parameters())
    model_engine, optimizer, _, __ = deepspeed.initialize(args=args, model=net, model_parameters=parameters)

    device = model_engine.local_rank

    # model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # print("in_feats " + str(in_feats))
    # # print("train_g.shape "+ str(train_g.shape))
    # print("train_labels.shape " + str(train_labels.shape))
    # # print("val_g.shape "+ str(val_g.shape))
    tic = time.time()
    step_time_list = []
    step_data_trans_time_list = []
    step_GPU_train_time_list = []
    see_memory_usage("-----------------------------------------before for epoch loop ")

    for epoch in range(args.num_epochs):
        print('Epoch ' + str(epoch))
        tic = time.time()

        # data loader sampling fan-out neighbor each new epoch
        for full_batch_step, (input_nodes, output_seeds, full_batch_blocks) in enumerate(full_batch_dataloader):
            print('full_batch_dataloader device')
            # print(len(full_batch_dataloader))
            print(full_batch_dataloader.device)
            # if full_batch_step==1: break    # now just test one full batch, later we can test OOM batch size training for big dataset

            see_memory_usage("-----------before full_batch_dataloader start")
            full_batch_inputs, full_batch_labels = load_subtensor(train_nfeat, train_labels, output_seeds, input_nodes,
                                                                  dataloader_device)
            see_memory_usage("-----------after full_batch_dataloader load_subtensor")
            # full_batch_blocks = [block.int().to(device) for block in full_batch_blocks]

            # Compute loss and prediction
            # batch_pred_full = model(full_batch_blocks, full_batch_inputs)
            # full_batch_loss = loss_fcn(batch_pred_full, full_batch_labels)

            # optimizer.zero_grad()
            # full_batch_loss.backward()
            # optimizer.step()
            # print('full batch loss ' + str(full_batch_loss.tolist()))

            # # # mini-batch start -------------------------------------------------------------------------------
            # full_batch_blocks = [block.int().to('cpu') for block in full_batch_blocks]
            OUTPUT_NID = full_batch_blocks[-1].dstdata[dgl.NID]
            # print('OUTPUT_NID')
            # print(OUTPUT_NID)
            # print('full_batch_blocks')
            # print(full_batch_blocks)
            # print('after full batch dataloader')

            # Create DataLoader for constructing blocks
            print('start generate block_dataloader')
            ssss_time = time.time()
            block_dataloader = generate_dataloader(train_g, full_batch_blocks, OUTPUT_NID, args)
            block_generate_time = time.time() - ssss_time
            print('total spend   ' + str(time.time() - ssss_time))
            print('after generate block_dataloader')
            # Define model and optimizer

            # Training loop
            avg = 0
            iter_tput = []
            avg_step_data_trans_time_list = []
            avg_step_GPU_train_time_list = []
            avg_step_time_list = []

            total_time = 0
            # CPU_mem("-----------------------------------------before start------------------------")

            if epoch == 4:
                total_time = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
            # torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # start.record()
            tic_step = time.time()

            torch.cuda.synchronize()

            batch_pred_compact = torch.tensor([], dtype=torch.long, device='cpu')
            batch_labels_compact = torch.tensor([], dtype=torch.long, device='cpu')
            model_engine.zero_grad()
            see_memory_usage("-----------------------------------------before block_dataloader loop ")
            for step, (input, seeds, blocks) in enumerate(block_dataloader):
                # print(
                # 	"\n   ***************************     step   " + str(step) + "   *************************************")

                torch.cuda.synchronize()
                start.record()
                # see_memory_usage("---before batch_inputs, batch_labels = load_blocks_subtensor(train_g, train_labels, blocks, device) ")

                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_blocks_subtensor(train_g, train_labels, blocks, device)

                blocks = [block.int().to(device) for block in blocks]

                torch.cuda.synchronize()  # wait for move to complete
                end.record()
                torch.cuda.synchronize()
                step_data_trans_time_list.append(start.elapsed_time(end))

                start1 = torch.cuda.Event(enable_timing=True)
                end1 = torch.cuda.Event(enable_timing=True)
                start1.record()

                see_memory_usage(
                    "-----------------------------------------before batch_pred = model(blocks, batch_inputs) ")

                # Compute loss and prediction
                batch_pred = model_engine(blocks, batch_inputs)
                see_memory_usage("-----------------------------------------after batch_pred = model(blocks, batch_inputs) ")
                # ---------------------------------------------------------------------------------------
                batch_pred = batch_pred.to(device='cpu')
                batch_labels = batch_labels.to(device='cpu')
                batch_pred_compact = torch.cat((batch_pred_compact, batch_pred))
                batch_labels_compact = torch.cat((batch_labels_compact, batch_labels))
                # ---------------------------------------------------------------------------------------
                print('batch_pred device')
                print(batch_pred.device)
                blocks = [block.int().to("cpu") for block in blocks]
                print('blocks device')
                print(blocks[0].device)

                torch.cuda.synchronize()  # wait for all training steps to complete
                end1.record()
                torch.cuda.synchronize()

                step_GPU_train_time_list.append(start1.elapsed_time(end1))

                torch.cuda.synchronize()
                step_time = time.time() - tic_step
                step_time_list.append(step_time)
                torch.cuda.empty_cache()
                # print(step_time)

                iter_tput.append(len(seeds) / (time.time() - tic_step))

                tic_step = time.time()

            final_loss = loss_fcn(batch_pred_compact.to(device), batch_labels_compact.to(device))
            optimizer.zero_grad()
        # final_loss.backward()
        model_engine.backward(final_loss)

        model_engine.step()
        see_memory_usage("-----------------------------------------after model_engine.step() ")

    # print('full batch loss '   + str(full_batch_loss.tolist()))
    # print('pseudo loss shape    '  + str(pseudo_loss) )
    print('final sum loss  ' + str(final_loss.tolist()))
    if epoch % args.log_every == 0:
        acc = compute_acc(batch_pred_compact, batch_labels_compact)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        print(
            'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                epoch, 0, final_loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

    #
    #
    if len(step_data_trans_time_list[5:]) > 0:
        avg_iteration_time = sum(step_data_trans_time_list[5:]) / len(step_data_trans_time_list[5:])
        print('avg iteration(step) data from cpu to GPU time:%.8f ms' % (avg_iteration_time))
        avg_step_data_trans_time_list.append(avg_iteration_time)

        avg_iteration_gpu_time = sum(step_GPU_train_time_list[5:]) / len(step_GPU_train_time_list[5:])
        print('avg iteration GPU training time:%.8f ms' % (avg_iteration_gpu_time))
        avg_step_GPU_train_time_list.append(avg_iteration_gpu_time)

        avg_step_time = sum(step_time_list[5:]) / len(step_time_list[5:])
        print('avg iteration (step) total cpu time:%.8f ms' % (avg_step_time * 1000))
        avg_step_time_list.append(avg_step_time)
    #
    toc = time.time()
    print('Epoch train cpu Time(s): {:.4f}'.format(toc - tic - block_generate_time))
    # avg += toc - tic
    if epoch >= 5:
        avg += toc - tic
    if epoch % args.eval_every == 0 and epoch != 0:
        eval_acc = evaluate(model_engine, val_g, val_nfeat, val_labels, val_nid, device)
        print('Eval Acc {:.4f}'.format(eval_acc))

    # print('Avg cpu epoch time: {} ms'.format(avg*1000 / (epoch - 4)))

    if len(avg_step_data_trans_time_list) > 0:
        total_avg_iteration_time = sum(avg_step_data_trans_time_list) / len(avg_step_data_trans_time_list)
        print('total avg iteration(step) data from cpu to GPU time:%.8f ms' % (total_avg_iteration_time))

        total_avg_iteration_gpu_time = sum(avg_step_GPU_train_time_list) / len(avg_step_GPU_train_time_list)
        print('total avg iteration GPU training time:%.8f ms' % (total_avg_iteration_gpu_time))

        total_avg_step_time_list = sum(avg_step_time_list) / len(avg_step_time_list)
        print('total avg iteration (step) total cpu time:%.8f ms' % (total_avg_step_time_list * 1000))

    test_acc = evaluate(model_engine, test_g, test_nfeat, test_labels, test_nid, device)
    print('Test Acc: {:.4f}'.format(test_acc))

if __name__ == '__main__':
    # get_memory("-----------------------------------------main_start***************************")
    tt = time.time()
    print("main start at this time " + str(tt))
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--local_rank', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--seed', type=int, default=1236)

    # argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--aggre', type=str, default='lstm')
    # argparser.add_argument('--dataset', type=str, default='cora')
    # argparser.add_argument('--dataset', type=str, default='karate')
    argparser.add_argument('--dataset', type=str, default='reddit')
    # argparser.add_argument('--aggre', type=str, default='mean')
    argparser.add_argument('--num-epochs', type=int, default=11)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=1)
    # argparser.add_argument('--fan-out', type=str, default='20')
    argparser.add_argument('--fan-out', type=str, default='10')
    argparser.add_argument('--batch-size', type=int, default=5000)
    # argparser.add_argument('--full-batch', type=int, default=6)
    argparser.add_argument('--log-every', type=int, default=5)
    argparser.add_argument('--eval-every', type=int, default=5)

    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    parser = deepspeed.add_config_arguments(argparser)
    args = parser.parse_args()

    if args.local_rank >= 0:
        device = torch.device('cuda:%d' % args.local_rank)
    else:
        device = torch.device('cpu')
    set_seed(args)

    # get_memory("-----------------------------------------before load_ogb***************************")
    t2 = ttt(tt, "before load_data")
    if args.dataset == 'karate':
        g, n_classes = load_karate()
    elif args.dataset == 'cora':
        g, n_classes = load_cora()
    elif args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb(args.dataset)
        print('#nodes:', g.number_of_nodes())
        print('#edges:', g.number_of_edges())
        print('#classes:', n_classes)
    # get_memory("-----------------------------------------after load_ogb***************************")

    # if args.dataset in ['arxiv', 'collab', 'citation', 'ddi', 'protein', 'ppa', 'reddit.dgl','products']:
    #     g, n_classes = load_data(args.dataset)
    else:
        raise Exception('unknown dataset')
    # see_memory_usage("-----------------------------------------after data to cpu------------------------")
    t3 = ttt(t2, "after load_dataset")
    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('feat')
        train_labels = val_labels = test_labels = g.ndata.pop('label')

    # get_memory("-----------------------------------------after inductive else***************************")
    t4 = ttt(t3, "after inductive else")
    print('args.data_cpu')
    print(args.data_cpu)

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)
    # get_memory("-----------------------------------------after label***************************")
    t5 = ttt(t4, "after label")
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    # get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
    val_g.create_formats_()
    # get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
    test_g.create_formats_()
    # get_memory("-----------------------------------------before pack data***************************")
    t6 = ttt(t5, "after train_g.create_formats_()")
    # see_memory_usage("-----------------------------------------after model to gpu------------------------")
    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels
    # get_memory("-----------------------------------------after pack data***************************")
    t7 = ttt(t6, "after pack data")
    run(args, device, data, t6)
import dgl
import numpy as np
import torch
import torch as th
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from memory_usage import see_memory_usage
from graphsage_model import SAGE
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.gpu >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True



def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	model.eval()
	with th.no_grad():
		pred = model.inference(g, nfeat, device, args)
	model.train()
	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
	# Unpack data
	n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
	val_nfeat, val_labels, test_nfeat, test_labels = data
	in_feats = train_nfeat.shape[1]
	train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
	val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
	test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

	dataloader_device = th.device('cpu')
	if args.sample_gpu:
		train_nid = train_nid.to(device)
		# copy only the csc to the GPU
		train_g = train_g.formats(['csc'])
		train_g = train_g.to(device)
		dataloader_device = device

	# Create PyTorch DataLoader for constructing blocks
	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	dataloader = dgl.dataloading.NodeDataLoader(
		train_g,
		train_nid,
		sampler,
		device=dataloader_device,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)

	# Define model and optimizer
	model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	# Training loop
	avg = 0
	iter_tput = []
	for epoch in range(args.num_epochs):
		tic = time.time()

		# Loop over the dataloader to sample the computation dependency graph as a list of
		# blocks.
		tic_step = time.time()
		loss_sum = 0
		acc_step =int(len(train_nid.tolist())/args.batch_size)
		
		# see_memory_usage("-----------------------------------------before dataloader loop ")
		for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
			print('-----------------------------step  '+ str(step)+' -------------------------------------')
			# see_memory_usage("-----------------------------------------start a new block ")
			# Load the input features as well as output labels
			batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
														seeds, input_nodes, device)
			blocks = [block.int().to(device) for block in blocks]

			# Compute loss and prediction
			# see_memory_usage("-----------------------------------------before batch_pred = model(blocks, batch_inputs) ")
			batch_pred = model(blocks, batch_inputs)
			# see_memory_usage("-----------------------------------------after batch_pred = model(blocks, batch_inputs) ")
			# loss = loss_fcn(batch_pred, batch_labels)
			# loss.backward()
			
			# optimizer.step()
			# optimizer.zero_grad()
			loss = loss_fcn(batch_pred, batch_labels)/acc_step
			loss_sum += loss
			
			# see_memory_usage("-----------------------------------------before loss.backward() ")
			loss.backward()
			# see_memory_usage("-----------------------------------------before optimizer.step() ")

			if step % acc_step == 0:
				optimizer.step()
				optimizer.zero_grad()
			see_memory_usage("-----------------------------------------after optimizer.step() ")

			iter_tput.append(len(seeds) / (time.time() - tic_step))
			# if step % args.log_every == 0:
			# 	acc = compute_acc(batch_pred, batch_labels)
			# 	gpu_mem_alloc = th.cuda.max_memory_allocated() / 1024/1024/1024 if th.cuda.is_available() else 0
			# 	print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.4f} GB'.format(
			# 		epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
			tic_step = time.time()

		print('------------------------------------------------------current epoch train accumulated loss     ' +str(loss_sum))
		

		toc = time.time()
		print('Epoch Time(s): {:.4f}'.format(toc - tic))
		if epoch >= 5:
			avg += toc - tic
		if epoch % args.eval_every == 0 and epoch != 0:
			eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
			print('Eval Acc {:.4f}'.format(eval_acc))

	train_acc = evaluate(model, train_g, train_nfeat, train_labels, train_nid, device)
	print('train Acc: {:.4f}'.format(train_acc))
	test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
	print('Test Acc: {:.4f}'.format(test_acc))

	print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__=='__main__':
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--gpu', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)

	argparser.add_argument('--dataset', type=str, default='ogbn-products')
	argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--selection-method', type=str, default='range')
	argparser.add_argument('--num-epochs', type=int, default=16)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='20')
	argparser.add_argument('--fan-out', type=str, default='10')


	argparser.add_argument('--batch-size', type=int, default=196571)
	# argparser.add_argument('--batch-size', type=int, default=98308)
	# argparser.add_argument('--batch-size', type=int, default=49154)
	# argparser.add_argument('--batch-size', type=int, default=24577)
	# argparser.add_argument('--batch-size', type=int, default=12289)
	# argparser.add_argument('--batch-size', type=int, default=6145)
	# argparser.add_argument('--batch-size', type=int, default=3000)
	# argparser.add_argument('--batch-size', type=int, default=1500)


	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)

	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument('--inductive', action='store_true',
		help="Inductive learning setting")
	argparser.add_argument('--sample-gpu', action='store_true',
        help="Perform the sampling process on the GPU. Must have 0 workers.")
	argparser.add_argument('--data-cpu', action='store_true',
		help="By default the script puts all node features and labels "
		     "on GPU when using it to save time for data copy. This may "
		     "be undesired if they cannot fit in GPU memory at once. "
		     "This flag disables that.")
	args = argparser.parse_args()
	args.sample_gpu = False

	if args.gpu >= 0:
		device = torch.device('cuda:%d' % args.gpu)
	else:
		device = torch.device('cpu')
	set_seed(args)

	# get_memory("-----------------------------------------before load_ogb***************************")
	# t2 = ttt(tt, "before load_data")
	if args.dataset=='karate':
		g, n_classes = load_karate()
	elif args.dataset=='cora':
		g, n_classes = load_cora()
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
	elif args.dataset=='ogbn-products':
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
	# t3 = ttt(t2, "after load_dataset")
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
	# t4 = ttt(t3, "after inductive else")
	print('args.data_cpu')
	print(args.data_cpu)

	if not args.data_cpu:
		train_nfeat = train_nfeat.to(device)
		train_labels = train_labels.to(device)
	# get_memory("-----------------------------------------after label***************************")
	# t5 = ttt(t4, "after label")
	# Create csr/coo/csc formats before launching training processes with multi-gpu.
	# This avoids creating certain formats in each sub-process, which saves momory and CPU.
	train_g.create_formats_()
	# get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
	val_g.create_formats_()
	# get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
	test_g.create_formats_()
	# get_memory("-----------------------------------------before pack data***************************")
	# t6 = ttt(t5, "after train_g.create_formats_()")
	# see_memory_usage("-----------------------------------------after model to gpu------------------------")
	tmp = (train_g.in_degrees()==0) & (train_g.out_degrees()==0)
	isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
	train_g.remove_nodes(isolated_nodes)
	# Pack data
	data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
	       val_nfeat, val_labels, test_nfeat, test_labels
	# get_memory("-----------------------------------------after pack data***************************")
	# t7 = ttt(t6, "after pack data")
	run(args, device, data)
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
	if args.gpu >= 0:
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
	with torch.no_grad():
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


def load_blocks_subtensor(g, labels, blocks, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = g.ndata['features'][blocks[0].srcdata[dgl.NID].tolist()].to(device)
	batch_labels = blocks[-1].dstdata['labels'].to(device)

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

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	full_batch_size = len(train_nid)
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		train_g,
		train_nid,
		sampler,
		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)

	model_full = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model_full = model_full.to(device)
	loss_fcn_full = nn.CrossEntropyLoss()
	optimizer_full = optim.Adam(model_full.parameters(), lr=args.lr)
#---------------------------------------------------------------------------------------------------------------------
	model_mini = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model_mini = model_mini.to(device)
	loss_fcn_mini = nn.CrossEntropyLoss()
	optimizer_mini = optim.Adam(model_mini.parameters(), lr=args.lr)
	# print("in_feats " + str(in_feats))
	# # print("train_g.shape "+ str(train_g.shape))
	# print("train_labels.shape " + str(train_labels.shape))
	# # print("val_g.shape "+ str(val_g.shape))

	#  ----------------------------full batch Training loop---------------------------------------------------------
	avg = 0
	iter_tput = []
	full_batch_loss_list_=[]
	# for epoch in range(args.num_epochs):
	# 	tic = time.time()
	#
	# 	# Loop over the dataloader to sample the computation dependency graph as a list of
	# 	# blocks.
	# 	tic_step = time.time()
	# 	for step, (input_nodes, seeds, full_batch_blocks) in enumerate(full_batch_dataloader):
	# 		# Load the input features as well as output labels
	# 		print('------------------step ' + str(step) + '-' * 30)
	# 		full_batch_inputs, full_batch_labels = load_subtensor(train_nfeat, train_labels, seeds, input_nodes, device)
	# 		full_batch_blocks = [block.int().to(device) for block in full_batch_blocks]
	#
	# 		# Compute loss and prediction
	# 		batch_pred = model_full(full_batch_blocks, full_batch_inputs)
	# 		full_batch_loss = loss_fcn(batch_pred, full_batch_labels)
	# 		full_batch_loss_list_.append(full_batch_loss)
	# 		optimizer_full.zero_grad()
	# 		full_batch_loss.backward()
	# 		optimizer_full.step()
	#
	# 		iter_tput.append(len(seeds) / (time.time() - tic_step))
	# 		if step % args.log_every==0:
	# 			acc = compute_acc(batch_pred, full_batch_labels)
	# 			gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
	# 			print(
	# 				'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
	# 					epoch, step, full_batch_loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
	# 		tic_step = time.time()
	#
	# 	toc = time.time()
	# 	print('Epoch Time(s): {:.4f}'.format(toc - tic))
	# 	if epoch >= 5:
	# 		avg += toc - tic
	# 	if epoch % args.eval_every==0 and epoch!=0:
	# 		eval_acc = evaluate(model_full, val_g, val_nfeat, val_labels, val_nid, device)
	# 		print('Eval Acc {:.4f}'.format(eval_acc))
	# 		test_acc = evaluate(model_full, test_g, test_nfeat, test_labels, test_nid, device)
	# 		print('Test Acc: {:.4f}'.format(test_acc))
	#
	# test_acc = evaluate(model_full, test_g, test_nfeat, test_labels, test_nid, device)
	# print('Test Acc: {:.4f}'.format(test_acc))

# -------------------------------------- pseudo - train -----------------------------------------------------------------------

	for full_batch_step, (input_nodes, output_seeds, full_batch_blocks) in enumerate(full_batch_dataloader):
		if full_batch_step == 1: break    # now just test one full batch, later we can test OOM batch size training for big dataset

		full_batch_inputs, full_batch_labels = load_subtensor(train_nfeat, train_labels, output_seeds, input_nodes, device)
		full_batch_blocks = [block.int().to(device) for block in full_batch_blocks]

		# Compute loss and prediction
		batch_pred = model_full(full_batch_blocks, full_batch_inputs)
		print('print(len(batch_pred))')
		print(len(batch_pred))
		full_batch_loss = loss_fcn_full(batch_pred, full_batch_labels)
		print('full_batch_loss')
		print(full_batch_loss.tolist())

		# mini-batch start -------------------------------------------------------------------------------
		full_batch_blocks = [block.int().to('cpu') for block in full_batch_blocks]
		OUTPUT_NID = full_batch_blocks[-1].dstdata[dgl.NID]
		print('OUTPUT_NID')
		print(OUTPUT_NID.tolist())
		print(sorted(OUTPUT_NID.tolist()))
		print('after full batch dataloader')

		# Create DataLoader for constructing blocks
		block_dataloader = generate_dataloader(train_g, full_batch_blocks, OUTPUT_NID, args)
		print('after generate block_dataloader')
		# Define model and optimizer

		# Training loop
		for epoch in range(args.num_epochs):

			# full_batch_inputs, full_batch_labels = load_subtensor(train_nfeat, train_labels, output_seeds, input_nodes,	device)
			# full_batch_blocks = [block.int().to(device) for block in full_batch_blocks]
			#
			# # Compute loss and prediction
			# batch_pred = model_full(full_batch_blocks, full_batch_inputs)
			#
			# full_batch_loss = loss_fcn_full(batch_pred, full_batch_labels)
			# print()
			# full_batch_loss_list_.append(full_batch_loss)
			# optimizer_full.zero_grad()
			# full_batch_loss.backward()
			# optimizer_full.step()
			#
			# if epoch % args.log_every==0:
			# 	acc = compute_acc(batch_pred, full_batch_labels)
			# 	gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
			# 	print(
			# 		'full       Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
			# 			epoch, 0, full_batch_loss.item(), acc.item(), 0, gpu_mem_alloc))
#------------------------------------------------------------------------------------------------------------

			batch_pred_compact = torch.tensor([], dtype=torch.long)
			batch_labels_compact = torch.tensor([], dtype=torch.long)
			# Loop over the dataloader to sample the computation dependency graph as a list of blocks.
			optimizer_mini.zero_grad()
			for step, (input, seeds, blocks_cur) in enumerate(block_dataloader):

				print(	"\n   ***************************     step   " + str(step) + "   *************************************")

				# Load the input features as well as output labels
				# print('total train_labels')
				# print(train_labels)
				print('current block ndata')
				print(blocks_cur[0].ndata)
				batch_inputs_cur, batch_labels_cur = load_blocks_subtensor(train_g, train_labels, blocks_cur, device)

				blocks_cur = [block.int().to(device) for block in blocks_cur]

				# Compute loss and prediction
				batch_pred_cur = model_mini(blocks_cur, batch_inputs_cur)
				print('batch_pred_cur')
				print(batch_pred_cur)
				pseudo_loss = loss_fcn_mini(batch_pred_cur, batch_labels_cur)
				optimizer_mini.zero_grad()
				pseudo_loss.backward()
				# print('batch_pred')
				# print(batch_pred)
				# print('batch_labels')
				# print(batch_labels)

				batch_pred_compact = torch.cat((batch_pred_compact, batch_pred_cur.to('cpu')))

				batch_labels_compact = torch.cat((batch_labels_compact, batch_labels_cur.to('cpu')))

				print('batch input')
				print(batch_inputs_cur)
				print('batch_labels')
				print((batch_labels_cur))

			optimizer_mini.step()

			print('batch_labels_compact')
			print(batch_labels_compact)

			final_loss = loss_fcn_mini(batch_pred_compact, batch_labels_compact)

			# optimizer_mini.zero_grad()
			# pseudo_loss.backward()
			# optimizer_mini.step()


			print('final loss')
			print(final_loss.tolist())
			print('full batch total loss----------------')
			# print(full_batch_loss_list_[epoch].tolist())
			print('full_batch_loss')
			print(full_batch_loss.tolist())


			if epoch % args.log_every == 0:
				acc = compute_acc(batch_pred_compact, batch_labels_compact)
				gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
				print(
					'mini     Epoch {:05d} | step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
						epoch, 0, pseudo_loss.item(), acc.item(), 0, gpu_mem_alloc))

			if epoch % args.eval_every==0 and epoch!=0:
				eval_acc = evaluate(model_mini, val_g, val_nfeat, val_labels, val_nid, device)
				print('Eval Acc {:.4f}'.format(eval_acc))



		test_acc_full = evaluate(model_full, test_g, test_nfeat, test_labels, test_nid, device)
		print('Test Acc full batch : {:.4f}'.format(test_acc_full))
		test_acc_mini = evaluate(model_mini, test_g, test_nfeat, test_labels, test_nid, device)
		print('Test Acc mini: {:.4f}'.format(test_acc_mini))




if __name__ == '__main__':
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--gpu', type=int, default=0,
						   help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)

	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--num-epochs', type=int, default=1)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='10')
	argparser.add_argument('--fan-out', type=str, default='2')
	argparser.add_argument('--batch-size', type=int, default=2)
	# argparser.add_argument('--full-batch', type=int, default=6)
	argparser.add_argument('--log-every', type=int, default=1)
	argparser.add_argument('--eval-every', type=int, default=1)

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
	args = argparser.parse_args()

	if args.gpu >= 0:
		device = torch.device('cuda:%d' % args.gpu)
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
	else :
		train_g = val_g = test_g = g
		train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('feat')
		train_labels = val_labels = test_labels = g.ndata.pop('label')

	# elif args.dataset == 'karate':
	# 	train_g = val_g = test_g = g
	# 	train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
	# 	train_labels = val_labels = test_labels = g.ndata.pop('labels')
	# elif args.dataset == 'cora':
	# 	train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('feat')
	# 	train_labels = val_labels = test_labels = g.ndata.pop('label')
	# get_memory("-----------------------------------------after inductive else***************************")
	t4 = ttt(t3, "after inductive else")

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
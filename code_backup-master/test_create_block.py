import torch
import dgl
import networkx as nx
import numpy as np
from collections.abc import Mapping
from block_dataloader import *
import argparse
from dgl.heterograph import DGLHeteroGraph, combine_frames, DGLBlock
from dataloading import Pseudo_DataLoader
from torch.utils import dlpack
from heterograph import DGLBlock
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def unique_tensor_item(combined):
	uniques, counts = combined.unique(return_counts=True)
	return uniques.type(torch.long)

def unique_edges(edges_list):

	temp = []
	for i in range(len(edges_list)):
		tt = edges_list[i]    # tt : [[],[]]
		for j in range(len(tt[0])):
			cur = (tt[0][j], tt[1][j])
			if cur not in temp:
				temp.append(cur)
	print(temp)   # [(),(),()...]
	res_ = list(map(list, zip(*temp))) # [],[]
	res=tuple(sub for sub in res_)
	return res


def merge_(list1, list2):
	merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
	return merged_list


def draw_graph(G):
	fig = plt.figure()
	black_edges = G.edges()
	black_edges = list(black_edges)
	print(black_edges[0])
	print(black_edges[1])
	# print('total eid number   '	)
	# print(len(black_edges[0]))
	# dd = int(len(black_edges[0]) / 2)
	dd = int(len(black_edges[0]))
	black_edges[0] = black_edges[0].tolist()
	black_edges[1] = black_edges[1].tolist()
	black_edges = merge_(black_edges[0][:dd], black_edges[1][:dd])
	# print('black_edges')
	# print(black_edges)
	nx_G = G.to_networkx()
	# nx_G = G.to_networkx().to_undirected()

	pos = nx.kamada_kawai_layout(nx_G)
	# pos = nx.spring_layout(nx_G)

	nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
	nx.draw_networkx_edge_labels(nx_G, pos, font_color='r', label_pos=0.7)
	# nx.draw_networkx_edges(nx_G, pos,  arrows=False)
	nx.draw_networkx_edges(nx_G, pos, edgelist=black_edges, arrows=True)
	ax = plt.gca()
	ax.margins(0.20)

	plt.axis("off")
	plt.show()

# def mapping_edges_id_from_local_to_global(local_graph_edges, sub_graph_global_nid, sub_graph_global_eid):
# 	'''
#
# 	Parameters
# 	----------
# 	local_graph_edges: (tensor,tensor)
# 	sub_graph_global_nid: tensor
# 	sub_graph_global_eid: tensor
#
# 	Returns
# 	-------
# 	global_graph_edges_ids: [[u],[v]]
# 	'''
# 	edges = list(local_graph_edges)
# 	edges[0] = edges[0].tolist()
# 	edges[1] = edges[1].tolist()
# 	print(edges[0])
# 	print(edges[1])
# 	print('sub_graph_global_eid')
# 	print(sub_graph_global_eid)
# 	sub_graph_global_nid=sub_graph_global_nid.tolist()
# 	ll= len(edges[0])
# 	for  i in range(len(sub_graph_global_nid)):
# 		nid = sub_graph_global_nid[i]
# 		edges[0] = [nid if x==i else x for x in edges[0]]
# 		edges[1] = [nid if x==i else x for x in edges[1]]
#
# 	global_graph_edges_ids = edges
# 	return global_graph_edges_ids

def mapping_(local_edges, global_nid):
	'''

	Parameters
	----------
	local_edges: (tensor,tensor)
	global_nid: tensor


	Returns
	-------
	global_graph_edges_ids: [[u],[v]]
	'''

	edges = list(local_edges)
	edges[0] = edges[0].tolist()
	edges[1] = edges[1].tolist()
	print(' the edges src and dst node ids')
	print(edges[0])
	print(edges[1])
	# print('sub_graph_global_eid')
	# print(sub_graph_global_eid)
	global_nid_list= global_nid.tolist()
	ll= len(edges[0])
	global_0=[]
	global_1=[]
	for local_eid in edges[0]:
		global_0.append(global_nid_list[local_eid])
	for local_eid in edges[1]:
		global_1.append(global_nid_list[local_eid])

	global_graph_edges_ids = [global_0, global_1]
	return global_graph_edges_ids


def check_connection(graph, nid, edges_with_global_nids):
	'''

	Parameters
	----------
	graph: dgl.graph
	nid : long integer
	edges_with_global_nids: [[],[]]

	Returns
	-------
	srcid: tensor
	dstid: tensor
	eid_list_: tensor
	edges: [[],[]]
	'''

	eid_list_src = []
	eid_list_dst = []
	nid_list_src = []
	nid_list_dst = []
	e = edges_with_global_nids
	src = e[0]
	dst = e[1]
	for i in range(len(dst)):

		if dst[i] == nid:
			eid_list_dst.append(dst[i])
			eid_list_src.append(src[i])
			if src[i] not in nid_list_src:   # keep original nid order
				nid_list_src.append(src[i])
			if dst[i] not in nid_list_dst:
				nid_list_dst.append(dst[i])

	global_edges = [eid_list_src, eid_list_dst]
	# re_edges = (torch.tensor(eid_list_src),torch.tensor(eid_list_dst))
	re_edges = global_edges
	eid_list_ = get_global_eid(graph, global_edges)
	srcid = torch.tensor(nid_list_src)

	dstid = torch.tensor(nid_list_dst)

	return srcid, dstid, eid_list_, re_edges


def get_global_eid(graph, global_edges):
	'''

	Parameters
	----------
	graph: dgl.graph
	global_edges: (tensor,tensor)

	Returns
	-------
	eid: tensor
	'''
	eid_list = []
	tmp = graph.edges(form='all')
	tmp_list = list(tmp)
	graph_e_0 = tmp_list[0].tolist()
	graph_e_1 = tmp_list[1].tolist()
	graph_e_id = tmp_list[2].tolist()
	graph_merged_list = [(graph_e_0[i], graph_e_1[i]) for i in range(0, len(graph_e_0))]

	e_0 = global_edges[0]
	e_1 = global_edges[1]
	merged_list = [(e_0[i], e_1[i]) for i in range(0, len(e_0))]
	for tup in merged_list:
		if tup in graph_merged_list:
			idd = graph_merged_list.index(tup)
			eid_list.append(graph_e_id[idd])


	eid = torch.tensor(eid_list)
	return eid

def create_block(G, data_dict):
	if not isinstance(data_dict, Mapping):
		data_dict = {('_N', '_E', '_N'): data_dict}
	node_tensor_dict = {}
	for (sty, ety, dty), data in data_dict.items():
		u, v, urange, vrange = dgl.utils.graphdata2tensors(data, idtype = G.idtype, bipartite=True)
		node_tensor_dict[(sty, ety, dty)] = (u, v)


	return

if __name__=='__main__':

	src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6])
	dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5])

	u = np.concatenate([src, dst])
	v = np.concatenate([dst, src])
	G = dgl.graph((u, v))
	draw_graph(G)

	fan_out = '2'
	full_batch = 3
	mini_batch = 2
	num_workers = 4
	# the sampler we can rewrite by our method instead of random fan-out neighbor sampling method.
	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in fan_out.split(',')])

	train_nid = torch.tensor([0, 1, 2, 3, 4, 5, 6])
	'''--------------------------------------------------------------------------------------------'''
	# full_batch_dataloader = dgl.dataloading.NodeDataLoader(
	# 	G,
	# 	train_nid,
	# 	sampler,
	# 	batch_size=full_batch,
	# 	shuffle=True,
	# 	drop_last=False,
	# 	num_workers=num_workers)
	# full_batch_src_nid = torch.tensor([], dtype=torch.long)
	# full_batch_dst_nid = torch.tensor([], dtype=torch.long)
	# full_batch_sub_graph = []
	#
	# for step, (input_nodes, output_seeds, blocks) in enumerate(full_batch_dataloader):
	# 	# print(blocks)
	#
	# 	if step==1: break
	# 	print(blocks)
	# 	# now, we have a full batch blocks
	# 	# (one(1-layer model) or multi-block(multi-layer model))
	# 	########################################################################################
	# 	OUTPUT_NID = blocks[-1].dstdata[dgl.NID]  # get the output node id from the last layer's output nid
	# 	# OUTPUT_NID = output_seeds  # get the output node id from the last layer's output nid
	# 	print()
	# 	print('OUTPUT_NID')
	# 	print(OUTPUT_NID)
	# 	#########----------------------------draw the full batch sub graph below-----------------------###############
	#
	#
	# 	for bb in blocks:
	# 		print('bb.edges()')
	# 		print(bb.edges())
	# 		# current_block_graph = dgl.block_to_graph(bb)
	# 		# current_block_graph = bb._graph
	# 		# now, only 1-layer, blocks length is 1
	# 		# draw_graph(current_block_graph)              # need modification for multi-layer model-=------???????//
	# 		# tt = bb.edata[dgl.EID]
	#
	# 		print()
	# 		tmp = bb.ndata
	# 		tmp_e = bb.edata[dgl.EID]
	# 		full_batch_src_nid = torch.cat((full_batch_src_nid, bb.srcdata[dgl.NID]))
	# 		full_batch_dst_nid = torch.cat((full_batch_dst_nid, bb.dstdata[dgl.NID]))
	# 		print('full_batch_src_nid')
	# 		print(full_batch_src_nid)
	# 		print('full_batch_dst_nid')
	# 		print(full_batch_dst_nid)
	#
	# 		tmp_edges = mapping_(bb.edges(), full_batch_src_nid)   # in current block, src_nid == total batch nid(includes index order)
	# 		print(tmp_edges)
	# 		eidx = get_global_eid(G, tmp_edges)
	# 		full_batch_sub_graph = dgl.edge_subgraph(G, eidx)
	# 		print('draw full batch sub_graph ------------------------------')
	# 		# print(full_batch_sub_graph.ndata)
	# 		draw_graph(full_batch_sub_graph)
	#
	#
	# combined = torch.cat((full_batch_src_nid, full_batch_dst_nid))
	# uniques, counts = combined.unique(return_counts=True)
	# sub_graph_total_nid = uniques.type(torch.long)     # only collect nodes id, not node index order
	# print(sub_graph_total_nid)

	'''-----------------------------------------------------------------------------------------------------------'''

	# print(total_nid.tolist())
	# idx = total_nid.tolist()
	# full_batch_sub_graph = dgl.node_subgraph(G, idx)
	# print('sub_graph ------------------------------')
	# print(full_batch_sub_graph.ndata)
	# draw_graph(full_batch_sub_graph)

		#########--------------------------draw full batch sub graph above-----------------------------################
	### TODO
	full_batch_sub_graph = dgl.node_subgraph(G, [0,2,4,5]) #	*************************************************************
	OUTPUT_NID = torch.tensor([0,2,4,5])
	### TODO
	#####888888-------------********************************************************
	argparser = argparse.ArgumentParser("multi-gpu training")
	args = argparser.parse_args()
	args.batch_size = 2

	batches_nid_list = generate_random_mini_batch_seeds_list(OUTPUT_NID, args)
	print('batches_nid_list')
	print(batches_nid_list)
	print()
	data_loader = generate_blocks(G, full_batch_sub_graph, batches_nid_list)
	for step, (input_nodes, output_nodes, blocks) in enumerate(data_loader):
		print('step ' + str(step)+'---------------------------------------------')
		print('input_nodes')
		print(input_nodes)
		print('output_nodes')
		print(output_nodes)
		print('blocks')
		print(blocks)
		print('new block ndata[dgl.NID] ---------------------- after change ')
		print(blocks[0].ndata[dgl.NID])
		print('new block edata[dgl.EID]')
		print(blocks[0].edata[dgl.EID])
		print('edges')
		print(blocks[0].edges())




#
#
#
# 	sub_graph_total_nid = full_batch_sub_graph.ndata[dgl.NID]
# 	sub_graph_total_eid = full_batch_sub_graph.edata[dgl.EID]
# 	print('sub_graph_total_eid                  999999999999999999999999999999999999999999')
# 	print(sub_graph_total_eid)
# 	print('print full batch subgraph edges')
# 	print(full_batch_sub_graph.edges())  # local nid for edges
# 	print('full_batch_sub_graph.edata')
# 	print(full_batch_sub_graph.edata)      # global nid for edges src and dst nodes
# 	print()
# 	edges_with_global_nids = mapping_(full_batch_sub_graph.edges(), sub_graph_total_nid)
# 	print('after mapping edges src and dst local to global nid')
# 	print(edges_with_global_nids[0])
# 	print(edges_with_global_nids[1])
# 	print()
#
# 	####################################################################################
# 	# generate random mini_batch output nodes list below
# 	full_len = len(OUTPUT_NID)           # get the total number of output nodes
# 	indices = torch.randperm(full_len)   # get a permutation of the index of output nid tensor (permutation of 0~n-1)
# 	### TODO
# 	indices = torch.tensor([1,2,3,0])
# 	### TODO
# 	batches_nid_list = []
# 	mod = int(full_len / mini_batch)      # mod = full_len / mini_batch
# 	for i in range(mod):
# 		idx = indices[i*mini_batch: (i + 1) * mini_batch]
# 		batches_nid_list.append(OUTPUT_NID[idx])
#
# 	print('batches_nid_list*******************************')
# 	print(batches_nid_list)
# 	print()
#
# 	if full_len % mini_batch != 0:           # tail_ = full_len % mini_batch
# 		tail_ = full_len % mini_batch
# 		idx = indices[-tail_:]
# 		batches_nid_list.append(OUTPUT_NID[idx])
# 		print(batches_nid_list[-1])
# 	######################################################################
# 	# based on batches_OUTPUT_NID_list[i], full_batch_graph_total_eid, generate all nid of this mini_batch subgraph
# 	# to generate a mini_batch_graph
# 	print('-0-'*50)
# 	print('print sub_graph_total_nid')
# 	print(sub_graph_total_nid)
# 	print('total OUTPUT_NID')
# 	print(OUTPUT_NID)
#
#
# #-------------------------------------------------------------------------------------------------------------------------
# 	mini_batch_subgraph_eids_list = [] # list of  tensor list; each mini_batch_subgraph_nids is a tensor in the end
# 	mini_batch_edges = []
# 	mini_batch_subgraph_src_nids_list = []
# 	mini_batch_subgraph_dst_nids_list = []
# 	for nids in batches_nid_list:   # nids is current batch output node ids
# 		cur_batch_subgraph_edges_list = []
# 		cur_batch_subgraph_eid = torch.tensor([], dtype=torch.long)
# 		cur_batch_subgraph_src_nid = torch.tensor([], dtype=torch.long)
# 		cur_batch_subgraph_dst_nid = torch.tensor([], dtype=torch.long)
#
# 		for nid in nids:          # each output nodes in current batch
# 			if nid not in sub_graph_total_nid:
# 				print('error, current nid is not in sub_graph_total_nid')
# 				break
# 			# nid in sub graph nodes' global nid, and return connected eid list
# 			srcnid, dstnid, connected_eid, edges = check_connection(G, nid, edges_with_global_nids)
#
# 			if len(connected_eid)==0:
# 				print('nid_list of current batch is empty. ERROR')
# 				break
# 			cur_batch_subgraph_edges_list.append(edges)
# 			cur_batch_subgraph_eid = torch.cat((cur_batch_subgraph_eid, connected_eid))
# 			cur_batch_subgraph_src_nid = unique_tensor_item(torch.cat((cur_batch_subgraph_src_nid, srcnid)))
# 			cur_batch_subgraph_dst_nid = unique_tensor_item(torch.cat((cur_batch_subgraph_dst_nid, dstnid)))
#
#
# 		cur_batch_subgraph_edges = unique_edges(cur_batch_subgraph_edges_list)
# 		# TODO
# 		# new_block = create_block(cur_batch_subgraph_edges, cur_batch_subgraph_eid, cur_batch_subgraph_src_nid, cur_batch_subgraph_dst_nid)
# 		# print('new block')
# 		# print(new_block)
# 		#### TODO
# 	# cur_batch_subgraph_edges : [[],[]]           input is[   [[],[]], [[],[]]   ]
# 		mini_batch_edges.append(cur_batch_subgraph_edges)
# 		mini_batch_subgraph_eids_list.append(cur_batch_subgraph_eid)
# 		mini_batch_subgraph_src_nids_list.append(cur_batch_subgraph_src_nid)
# 		mini_batch_subgraph_dst_nids_list.append(cur_batch_subgraph_dst_nid)
#
# 	print()
# 	print('-------------------------mini_batch_subgraph_eids_list---------------------')
# 	print('mini_batch_subgraph_eids_list')
# 	print(mini_batch_subgraph_eids_list)
# 	print()
# 	print('mini_batch_edges')
# 	print(mini_batch_edges)
# 	print()
# 	print('mini_batch_subgraph_src_nids_list')
# 	print(mini_batch_subgraph_src_nids_list)
# 	print()
# 	print('mini_batch_subgraph_dst_nids_list')
# 	print(mini_batch_subgraph_dst_nids_list)
# 	print()
#
#
#
# 	# #######################   1- layer example  #################################################
# 	# # after get nid needed, generate mini-batch dataloader
# 	# # dgl.dataloading.pytorch.NodeDataLoader(g, nids, block_sampler, device='cpu', **kwargs)
# 	print('full batch -------------------dgl.NID, dgl.EID')
# 	print(full_batch_sub_graph.ndata[dgl.NID])
# 	print(full_batch_sub_graph.edata[dgl.EID])
#
# 	# dataloader = Pseudo_DataLoader( G, mini_batch_subgraph_eids_list)
#
# 	# for input_nodes, output_nodes, blocks in dataloader:
# 	# for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
# 	# 	print('step ' + str(step))
# 	# 	print('input_nodes')
# 	# 	print(input_nodes)
# 	# 	print('output_nodes')
# 	# 	print(output_nodes)
# 	# 	print('blocks')
# 	# 	print(blocks)
#
# 	lenn = len(mini_batch_subgraph_eids_list)
# 	blocks = []
# 	for i in range(lenn):
# 		print('')
# 		eids = mini_batch_subgraph_eids_list[i]
# 		cur_block_sub_graph = dgl.edge_subgraph(G, eids)
# 		print('min_batch sub_graph ------------------------------ block ' + str(i))
# 		draw_graph(cur_block_sub_graph)
# 		print()
# 		print('print graph')
# 		print('dgl.NID, dgl.EID')
# 		print(cur_block_sub_graph.ndata[dgl.NID])
# 		print(cur_block_sub_graph.ndata)
# 		print(cur_block_sub_graph.edata[dgl.EID])
# 		print(cur_block_sub_graph.edata)
# 		print(cur_block_sub_graph.edges())
# 		print(cur_block_sub_graph.dstdata)
#
#
# 		edges_global_nid = mini_batch_edges[i]  # [[],[]]
# 		# new_block = create_block(G, eids, edges_global_nid)
# 		dst_local_nid_list = cur_block_sub_graph.edges()[1].tolist()
# 		dst_local_nid_list = list(set(dst_local_nid_list))
# 		new_block = dgl.to_block(cur_block_sub_graph, dst_nodes=torch.tensor(dst_local_nid_list))
# 		print()
# 		print('new block')
# 		print(new_block)
# 		print('new block ndata')
# 		print(new_block.ndata)
# 		print('new block edata')
# 		print(new_block.edata)
# 		print('new block ndata[dgl.NID]')
# 		print(new_block.ndata[dgl.NID])
# 		print('cur_block_sub_graph.ndata[dgl.NID]')
# 		print(cur_block_sub_graph.ndata[dgl.NID])
# 		global_nid_list = cur_block_sub_graph.ndata[dgl.NID].tolist()
# 		block_nid_list = new_block.ndata[dgl.NID]['_N'].tolist()
# 		final_nid_list = [global_nid_list[i] for i in block_nid_list]
# 		new_block.ndata[dgl.NID] ={'_N': torch.tensor(final_nid_list)}
# 		print('new block ndata ---------------after change')
# 		print(new_block.ndata)
#
#
#
#
# 		print('new block ndata[dgl.NID] ---------------------- after change ')
# 		print(new_block.ndata[dgl.NID])
# 		print('new block edata[dgl.EID]')
# 		print(new_block.edata[dgl.EID])
# 		print('edges')
# 		print(new_block.edges())
# 		blocks.append(new_block)
#
#
#
#
#
#
# 	print()
# 	print('blocks')
# 	print(blocks)
#
#
#
#
#
#
#
# 	# lenn = len(mini_batch_subgraph_eids_list)
# 	# blocks = []
# 	# for i in range(lenn):
# 	# 	print('')
# 	# 	eids = mini_batch_subgraph_eids_list[i]
# 	# 	cur_block_sub_graph = dgl.edge_subgraph(G, eids)
# 	# 	print('min_batch sub_graph ------------------------------ block ' + str(i))
# 	# 	draw_graph(cur_block_sub_graph)
# 	# 	print('dgl.NID, dgl.EID')
# 	# 	print(cur_block_sub_graph.ndata[dgl.NID])
# 	# 	print(cur_block_sub_graph.ndata)
# 	# 	print(cur_block_sub_graph.edata[dgl.EID])
# 	# 	print(cur_block_sub_graph.edata)
# 	#
# 	#
# 	# 	blocks.append(cur_block_sub_graph)
#
#
# 		# print()
# 		# new_block = dgl.to_block(cur_block_sub_graph)
# 		# print('new_block')
# 		# print(new_block)
# 		# print('new_block dgl.NID')
# 		# print(new_block.ndata[dgl.NID])
# 		# print('new_block dgl.EID')
# 		# print(new_block.edata[dgl.EID])
# 		#
# 		# new_block.ndata[dgl.NID] = cur_block_sub_graph.ndata[dgl.NID]
# 		# new_block.edata[dgl.EID] = cur_block_sub_graph.edata[dgl.EID]
# 		# print('after update -------------------------------------------------')
# 		# print('new_block dgl.NID')
# 		# print(new_block.ndata[dgl.NID])
# 		# print('new_block dgl.EID')
# 		# print(new_block.edata[dgl.EID])
#
#
#
# 		#
# 		# new_graph = DGLBlock(new_graph_index, new_ntypes, g.etypes)
# 		# assert new_graph.is_unibipartite  # sanity check
# 		#
# 		# src_node_ids = [F.from_dgl_nd(src) for src in src_nodes_nd]
# 		# edge_ids = [F.from_dgl_nd(eid) for eid in induced_edges_nd]
# 		#
# 		# node_frames = utils.extract_node_subframes_for_block(g, src_node_ids, dst_node_ids)
# 		# edge_frames = utils.extract_edge_subframes(g, edge_ids)
# 		# utils.set_new_frames(new_graph, node_frames=node_frames, edge_frames=edge_frames)
#
#
#
#
#
#
#
#
#
#
# 		# edges_global_eid = mini_batch_edges[i] # [[],[]]
# 		# print('edges_global_eid:     '+ str(edges_global_eid))
# 		#
# 		# # src_nodes = mini_batch_subgraph_src_nids_list[i]
# 		# # num_src_nodes = len(src_nodes)
# 		# # dst_nodes = mini_batch_subgraph_dst_nids_list[i]
# 		# # num_dst_nodes = len(dst_nodes)
# 		# # new_block = dgl.create_block(edges_global_eid, num_src_nodes=num_src_nodes, num_dst_nodes=num_dst_nodes)
#
# 		# for block_id in reversed(range(num_layers)):
# 		# frontier = sample_frontier(block_id, g, seed_nodes)
# 		# frontier = sample_frontier(0, g, seed_nodes)
#
#
#
#
#
#



















		# new_block = dgl.create_block(edges_global_eid)
		#
		# print(new_block)
		# new_g= dgl.block_to_graph(new_block)
		# print(new_g.srcdata[dgl.NID])
		# print(new_g.dstdata[dgl.NID])
		#
		# # print(new_block)
		# print()



		# mini_batch_dataloader = dgl.dataloading.DataLoader(
		# 	G,
		# 	idx,
		# 	sampler,
		# 	batch_size=mini_batch,
		# 	shuffle=True,
		# 	drop_last=False,
		# 	num_workers=num_workers)
		#
		# for step, (input_nodes, output_seeds, blocks) in enumerate(mini_batch_dataloader):
		#
		#



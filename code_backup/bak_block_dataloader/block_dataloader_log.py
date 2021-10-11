import torch
import dgl
import numpy
import time
from itertools import islice
from memory_usage import see_memory_usage

# from utils import draw_graph_global, draw_graph_local


def unique_tensor_item(combined):
	uniques, counts = combined.unique(return_counts=True)
	return uniques.type(torch.long)


# def unique_edges(edges_list):
# 	temp = []
# 	for i in range(len(edges_list)):
# 		tt = edges_list[i]  # tt : [[],[]]
# 		for j in range(len(tt[0])):
# 			cur = (tt[0][j], tt[1][j])
# 			if cur not in temp:
# 				temp.append(cur)
# 	# print(temp)   # [(),(),()...]
# 	res_ = list(map(list, zip(*temp)))  # [],[]
# 	res = tuple(sub for sub in res_)
# 	return res


def generate_random_mini_batch_seeds_list(OUTPUT_NID, args):
	'''

	Parameters
	----------
	OUTPUT_NID: final layer output nodes global nid (tensor)
	args : all given parameters collection

	Returns
	-------

	'''
	mini_batch = args.batch_size
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	indices = torch.randperm(full_len)  # get a permutation of the index of output nid tensor (permutation of 0~n-1)
	# ### TODO
	# indices = torch.tensor([1, 2, 3, 0])
	# ### TODO
	print('OUTPUT_NID.tolist()')
	print(OUTPUT_NID.tolist())
	map_output_list = list(numpy.array(OUTPUT_NID)[indices.tolist()])

	print('list for split')
	print(map_output_list)
	batches_nid_list = [torch.tensor(map_output_list[i:i + mini_batch], dtype=torch.long) for i in
	                    range(0, len(map_output_list), mini_batch)]

	# def chunk(iterable, n=1):
	# 	l = len(iterable)
	# 	for ndx in range(0, l, n):
	# 		yield iterable[ndx:min(ndx + n, l)]

	# mappped_output = torch.tensor(map_output_list, dtype=torch.long)
	# batches_nid_list = chunk(mappped_output, mini_batch)
	# print()

	# batches_nid_list= [torch.tensor([20, 6]), torch.tensor([14, 7]), torch.tensor([2, 8]), torch.tensor([21, 3]), torch.tensor([9, 10]), torch.tensor([1, 16]),\
	#  torch.tensor([0, 5]), torch.tensor([11, 17]), torch.tensor([19, 23]), torch.tensor([18, 15]), torch.tensor([12, 4]), torch.tensor([22, 13])]
	# print('batches_nid_list')
	for i in batches_nid_list:
		print(i)
	# print(batches_nid_list)
	return batches_nid_list


def check_connection_nids(given_output_nids, edges, eidx):
	# global nid to local graph nid
	# edges order is match with eidx based on (order = 'eid')

	given_nid_list_ = given_output_nids.tolist()
	print('given_nid_list_')
	print(given_nid_list_)
	edges = list(edges)
	raw_src = edges[0].tolist()
	raw_dst = edges[1].tolist()
	edix_list = eidx.tolist()

	dict_ = {raw_dst[i]: edix_list[i] for i in range(0, len(raw_dst))}
	eid_list = list(map(dict_.get, given_nid_list_))
	print('eid_list')
	print(eid_list)

	dict_src = {edix_list[i]: raw_src[i] for i in range(0, len(edix_list))}
	src_list = list(map(dict_src.get, eid_list))
	print('pure src')
	print(src_list)

	global_eid_tensor = torch.tensor(eid_list, dtype=torch.long)

	src_final_list =  given_nid_list_ + src_list

	srcid = unique_tensor_item(torch.tensor(src_final_list, dtype=torch.long))
	print('srcid')
	print(srcid)

	dstid = given_output_nids
	# see_memory_usage('after map local srcnid , dstnid,  eid to global')

	return srcid, dstid, global_eid_tensor


def get_global_graph_edges_ids(raw_graph, cur_block):
	'''

		Parameters
		----------
		raw_graph : graph
		cur_block: (local nids, local nids): (tensor,tensor)


		Returns
		-------
		global_graph_edges_ids: []                    current block edges global id list

		'''

	src, dst = cur_block.all_edges(order='eid')
	src = src.long()
	dst = dst.long()
	print(src.tolist())
	print(dst.tolist())
	raw_src, raw_dst = cur_block.srcdata[dgl.NID][src], cur_block.dstdata[dgl.NID][dst]
	print(raw_src.tolist())
	print(raw_dst.tolist())
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)


# print('---------------print cur block srcdata and dstdata--------------------------------------------------------------------')
# print(cur_block.srcdata[dgl.NID].tolist())
# print(cur_block.dstdata[dgl.NID].tolist())
#
# print('---------------print cur block local edges------------')
#
# edges = list(cur_block.edges())
# print(edges[0].tolist())
# print(edges[1].tolist())
#
# print('total edges in current full batch block ')
# print(len(edges[0]))
#
# # print(sub_graph_global_eid)
# global_nids = cur_block.ndata['_ID']['_N']
# global_nid_list = global_nids.tolist()
#
# global_src = list(numpy.array(global_nid_list)[edges[0]])
# global_dst = list(numpy.array(global_nid_list)[edges[1]])
# # global_src = [global_nid_list[local_src] for local_src in edges[0]]
# # global_dst = [global_nid_list[local_dst] for local_dst in edges[1]]
# print('global_src')
# print(global_src)
# print('global_dst')
# print(global_dst)
# print()
# src, dst = cur_block.all_edges(order='eid')
# src = src.long()
# dst = dst.long()
# print(src.tolist())
# print(dst.tolist())
# raw_src, raw_dst = cur_block.srcdata[dgl.NID][src], cur_block.dstdata[dgl.NID][dst]
# print('raw_src')
# print(raw_src.tolist())
# print('raw_dst')
# print(raw_dst.tolist())
# print()
# #
# global_graph_eids_raw = G.edge_ids(raw_src, raw_dst)
# print('raw ')
# print(global_graph_eids_raw.tolist())
#
# global_graph_eids = G.edge_ids(global_src, global_dst)
# print('total edges in current full batch block after mapping to global')
# print(len(global_graph_eids.tolist()))
# print(global_graph_eids.tolist())
#
# return global_graph_eids


def generate_one_block(raw_graph, eids):
	'''

	Parameters
	----------
	raw_graph    global graph                              DGLGraph
	eids         global eids to create a new_block         tensor long

	Returns
	-------

	'''
	graph = dgl.edge_subgraph(raw_graph, eids)
	print('batch graph')
	print(graph.ndata)
	print(graph.edata)
	print(graph.edges())

	dst_local_nid_list = graph.edges()[1].tolist()
	dst_local_nid_list = list(set(dst_local_nid_list))
	new_block = dgl.to_block(graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))
	print()
	print('batch graph to block')
	print(new_block.ndata)
	print(new_block.srcdata)
	print(new_block.dstdata)
	print(new_block.edata)
	print(new_block.edges())

	global_nid_list = graph.ndata[dgl.NID].tolist()
	block_nid_list = new_block.ndata[dgl.NID]['_N'].tolist()
	block_dst_nid_list = new_block.dstdata[dgl.NID].tolist()

	final_nid_list = [global_nid_list[i] for i in block_nid_list]  # mapping global graph nid <--- block local nid
	final_dst_nid_list = [global_nid_list[i] for i in block_dst_nid_list]

	new_block.ndata[dgl.NID] = {'_N': torch.tensor(final_nid_list, dtype=torch.long)}  # the final nid list is global nid, then assign it to block
	new_block.dstdata[dgl.NID] = torch.tensor(final_dst_nid_list, dtype=torch.long)
	print()
	print('after rename new block nodes id ')
	print(new_block.ndata)
	print(new_block.srcdata)
	print(new_block.dstdata)
	print(new_block.edata)
	print(new_block.edges())

	return new_block


def generate_blocks(raw_graph,  edges, eidx, batches_nid_list):
	print('generate_blocks ----------------------------------------------------')

	data_loader = []
	for step, nids in enumerate(batches_nid_list):
		print('-------------------------------------batch ' + str(step) + '-' * 30)

		input_nodes, seeds, connected_global_eid = check_connection_nids(nids, edges, eidx)
		# see_memory_usage('after check_connection_nids ')

		cur_block = generate_one_block(raw_graph, connected_global_eid)
		# see_memory_usage('after generate_one_block ')

		data_loader.append((input_nodes, seeds, [cur_block]))

	# see_memory_usage('after generate blocks')

	return data_loader


def generate_dataloader(G, full_batch_data_blocks, OUTPUT_NID, args):
	# full_batch_blocks_src_nid = torch.tensor([], dtype=torch.long)
	# full_batch_blocks_dst_nid = torch.tensor([], dtype=torch.long)

	for cur_block in full_batch_data_blocks:  # the for loop total times equals model layers

		print('block_dataloader: generate_dataloader: cur_block.edges()')  # local edges (local nodes index)
		# print(cur_block.edges())
		# current_graph is created as a comparision for current_block, useless

		current_graph = dgl.block_to_graph(cur_block)
		print('current_graph')
		print(current_graph.ndata)
		print(current_graph.srcdata['_ID'])
		print(current_graph.dstdata['_ID'])
		print(current_graph.edata)
		# print(current_graph.edges())
		# draw_graph_local(current_graph)
		#------------------------------------------------------------------------------------------------------------------------
		current_block_eidx, current_block_edges = get_global_graph_edges_ids(G, cur_block)
		current_graph.edata['_ID']= current_block_eidx
		#------------------------------------------------------------------------------------------------------------------------
		# based on current block and global training graph to get global eid in current block
		print('current_block_eidx = get_global_graph_edges_ids(G, cur_block)')
		print(current_block_eidx.tolist())

		
		print('cur_block.edata[dgl.EID]')
		print(current_graph.edata['_ID'])

		cur_b, _ = torch.sort(cur_block.ndata['_ID']['_N'])
		print(cur_b.tolist())
		# print(full_batch_sub_graph.ndata[dgl.NID].tolist())
		full, _ = torch.sort(current_graph.srcdata[dgl.NID])
		print(full.tolist())

		if torch.equal(cur_b, full):
			print()

			'''-----------------------------------------------------------------------------------------------------------'''
			# print('batch_size')
			# print(args.batch_size)

			batches_nid_list = generate_random_mini_batch_seeds_list(OUTPUT_NID, args)
			# see_memory_usage('**********************************************************************after batches_nid_list')
			print('after batch_nid_list')

			data_loader = generate_blocks(G, current_block_edges, current_block_eidx, batches_nid_list)
			# see_memory_usage('*******************************************generate_blocks(G, full_batch_sub_graph, batches_nid_list)')
			return data_loader

		else:

			print(len(cur_block.ndata['_ID']['_N']))
			print(len(current_graph.ndata[dgl.NID]))

			print('transformation is not correct')
			print('generated from eidx sub_graph   node length')
			print(len(current_graph.ndata[dgl.NID]))

			print('current block  total  node length')
			print(len(cur_block.ndata[dgl.NID]))

			return []


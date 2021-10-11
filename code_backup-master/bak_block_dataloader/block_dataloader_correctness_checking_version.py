import torch
import dgl
import numpy
import time
from itertools import islice


def unique_tensor_item(combined):
	uniques, counts = combined.unique(return_counts=True)
	return uniques.type(torch.long)


def unique_edges(edges_list):
	temp = []
	for i in range(len(edges_list)):
		tt = edges_list[i]  # tt : [[],[]]
		for j in range(len(tt[0])):
			cur = (tt[0][j], tt[1][j])
			if cur not in temp:
				temp.append(cur)
	# print(temp)   # [(),(),()...]
	res_ = list(map(list, zip(*temp)))  # [],[]
	res = tuple(sub for sub in res_)
	return res


def generate_random_mini_batch_seeds_list(OUTPUT_NID, args):
	'''

	Parameters
	----------
	OUTPUT_NID: final layer output nodes id (tensor)
	args : all given parameters collection

	Returns
	-------

	'''
	selection_method = args.selection_method
 	
	mini_batch = args.batch_size
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	if selection_method == 'random':
		indices = torch.randperm(full_len)  # get a permutation of the index of output nid tensor (permutation of 0~n-1)
	else: #selection_method == 'range'
		indices = torch.tensor(range(full_len))
	
	print('OUTPUT_NID.tolist()')
	print(len(OUTPUT_NID.tolist()))
	print()
	output_num = len(OUTPUT_NID.tolist())

	map_output_list = list(numpy.array(OUTPUT_NID)[indices.tolist()])
	print('list for split-------------------------------------------------------------------------------------------')
	print(len(map_output_list))
	print()

	batches_nid_list = [torch.tensor(map_output_list[i:i + mini_batch], dtype=torch.long) for i in range(0, len(map_output_list), mini_batch)]
	print('-------------------------batches_nid_list ----------------------------------------------------------------')
	print(len(batches_nid_list))
	# print(batches_nid_list)
	weights_list = []
	for i in batches_nid_list:
		temp = len(i)/output_num
		
		weights_list.append(len(i)/output_num)
	print('- -- -- --'*2 +'weights_list'+'-- -- -- -----'*2)
	print(weights_list)



	# def chunk(iterable, n=1):
	# 	l = len(iterable)
	# 	for ndx in range(0, l, n):
	# 		yield iterable[ndx:min(ndx + n, l)]

	# mappped_output = torch.tensor(map_output_list, dtype=torch.long)
	# batches_nid_list = chunk(mappped_output, mini_batch)
	# print()

	
	return batches_nid_list, weights_list


def check_connection_nids(given_output_nids, full_batch_block_graph, full_batch_block_edges, full_batch_block_eidx):
	# in block, only has src and dst nodes,
	# and src nodes includes dst nodes, src nodes equals dst nodes.
	given_nid_list_ = given_output_nids.tolist()
	# print('given_nid_list_')
	# print(given_nid_list_)
	# ------------------------------------------------------------------------------

	# edges = list(full_batch_block_edges)
	# raw_src = edges[0].tolist()
	# raw_dst = edges[1].tolist()
	# print('\n------------------------------------------- graph edges')
	# print(raw_src)
	# print(raw_dst)
	#
	# eidx_list = full_batch_block_eidx.tolist()
	# print('\n --------------------------------------------direct full batch edix_list')
	# # print(eidx_list)
	#
	# dict_ = {eidx_list[i]: raw_dst[i] for i in range(0, len(eidx_list))}
	#
	# new_dict = {v: x for v, x in dict_.items() if x in given_nid_list_}
	#
	# eid_list = list(new_dict.keys())
	# print('\n-------------------------------------------  mini-batch dict')
	# # print(dict_)
	# print('\n-------------------------------------------  mini-batch eid_list')
	# # print(eid_list)
	#
	# dict_src = {eidx_list[i]: raw_src[i] for i in range(0, len(eidx_list))}
	# src_list = list(map(dict_src.get, eid_list))
	# print('-------------------------------------------    min-batch pure src')
	# print(src_list)
	#
	# global_eid_tensor_ = torch.tensor(eid_list, dtype=torch.long)
	# src_final_list = given_nid_list_ + src_list
	# srcid_ = unique_tensor_item(torch.tensor(src_final_list, dtype=torch.long))

	# ---------------------------------------------------------------------------------

	block_src_nid_list = full_batch_block_graph.srcdata['_ID'].tolist()
	# print(
		# '\n *********************************************************************   src nid of full_batch_block_graph')
	# print(block_src_nid_list)

	dict_nid_2_local = {block_src_nid_list[i]: i for i in range(0, len(block_src_nid_list))}
	local_given_output_nids = list(map(dict_nid_2_local.get, given_nid_list_))

	local_in_edges_tensor = full_batch_block_graph.in_edges(local_given_output_nids, form='all')

	# get local srcnid and dstnid from subgraph
	mini_batch_srcid_local_list = list(local_in_edges_tensor)[0].tolist()

	srcid_list = list(numpy.array(block_src_nid_list)[mini_batch_srcid_local_list])

	# map local srcnid , dstnid,  eid to global
	eid_local_list = list(local_in_edges_tensor)[2]
	block_eids_global_list = full_batch_block_graph.edata['_ID'].tolist()
	# print('full batch block_eids_global_list')
	# print(block_eids_global_list)

	eid_list = list(numpy.array(block_eids_global_list)[eid_local_list.tolist()])

	global_eid_tensor = torch.tensor(eid_list, dtype=torch.long)

	srcid = torch.tensor(list(set(given_nid_list_+ srcid_list)), dtype=torch.long)
	# -------------------------------------------------------------------------
	# compare two methods results
	# print('srcid_')
	# print(srcid_)
	# print('\nsrcid')
	# print(srcid)
	# print('\nglobal_eid_tensor_')
	# print(global_eid_tensor_)
	# print('\nglobal_eid_tensor')
	# print(global_eid_tensor)
	# print()

	# -------------------------------------------------------------------------

	dstid = given_output_nids

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
	# print(src.tolist())
	# print(dst.tolist())
	raw_src, raw_dst = cur_block.srcdata[dgl.NID][src], cur_block.dstdata[dgl.NID][dst]
	# print(raw_src.tolist())
	# print(raw_dst.tolist())
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)


def generate_one_block(raw_graph, mini_batch_block_global_eids, mini_batch_block_global_srcnid):
	'''

	Parameters
	----------
	G    global graph                     DGLGraph
	eids  cur_batch_subgraph_global eid   tensor int64

	Returns
	-------

	'''
	mini_batch_graph = dgl.edge_subgraph(raw_graph, mini_batch_block_global_eids)
	# mini_batch_graph = dgl.edge_subgraph(raw_graph, mini_batch_block_global_eids, relabel_nodes=False)  # ?????????????????????
	# print('generated mini_batch_graph.ndata[dgl.NID]----------------------')
	# print(mini_batch_graph.ndata)
	# print('\n-----------------------------------------------------mini_batch_block_global_srcnid should be ')
	# print(mini_batch_block_global_srcnid)

	# gg, _ = torch.sort(mini_batch_graph.srcdata[dgl.NID])
	# oo, _ = torch.sort(mini_batch_block_global_srcnid)
	# isolated_nodes =[]

	# if not torch.equal(gg, oo):
	# 	# print('dgl.edge_subgraph in generate one block function failed. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	# 	if len(gg)<len(oo):
	# 		# print(' new generated graph nodes number is smaller than given edges srcnid  ' + str(len(oo)-len(gg)))
	# 		combined = torch.cat((gg, oo))
	# 		uniques, counts = combined.unique(return_counts=True)
	# 		difference = uniques[counts == 1]
	# 		# print('they are '+ str(difference))
	# 		isolated_nodes = isolated_nodes + difference.tolist()
	# 		# print('-------------------------------------------------------------------------------------------------\n')
	# 	else:
	# 		print('?'*50)

	edge_dst_list = mini_batch_graph.edges()[1].tolist()
	dst_local_nid_list = list(set(edge_dst_list))
	new_block = dgl.to_block(mini_batch_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))

	global_nid_list = mini_batch_graph.ndata[dgl.NID].tolist()
	block_nid_list = new_block.ndata[dgl.NID]['_N'].tolist()
	block_dst_nid_list = new_block.dstdata[dgl.NID].tolist()

	final_nid_list = [global_nid_list[i] for i in block_nid_list]  # mapping global graph nid <--- block local nid
	final_dst_nid_list = [global_nid_list[i] for i in block_dst_nid_list]

	new_block.ndata[dgl.NID] = {'_N': torch.tensor(final_nid_list, dtype=torch.long)}
	new_block.dstdata[dgl.NID] = torch.tensor(final_dst_nid_list, dtype=torch.long)
	# the final nid list is global nid, then assign it to block
	# print()
	# print(new_block.ndata)
	# print()
	# print(new_block.srcdata)
	# print()
	# print(new_block.dstdata)
	# print()
	# print(new_block.edata)
	# print()
	# return new_block, isolated_nodes
	return new_block


def generate_blocks(raw_graph, full_batch_block_2_graph, edges, eidx, batches_nid_list):
	data_loader = []
	# removed_isolated_nodes = []
	for step, nids in enumerate(batches_nid_list):
		# print('batch ' + str(step) + '-' * 30)

		srcnid, dstnid, current_block_global_eid = check_connection_nids(nids, full_batch_block_2_graph, edges, eidx)
		cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid)

		# cur_block, removed_nodes = generate_one_block(raw_graph, current_block_global_eid, srcnid)
		# removed_isolated_nodes = removed_isolated_nodes + removed_nodes

		data_loader.append((srcnid, dstnid, [cur_block]))
		# print()
		# print(len(removed_isolated_nodes))
		# removed_isolated_nodes.sort()
		# print('-----------------*&^%$##----------------------the total removed isolated nodes\n ')
		# print(removed_isolated_nodes)
		# print()
	return data_loader


def generate_dataloader(raw_graph, full_batch_data_blocks, args):
	for cur_block in full_batch_data_blocks:

		# print('bb.edges()')
		# print(bb.edges())

		block_to_graph = dgl.block_to_graph(cur_block)
		# print('block to graph.ndata')
		# # print(block_to_graph.ndata)
		# print('\nblock to graph.srcdata')
		# # print(block_to_graph.srcdata)
		# print('\nblock to graph.dstdata')
		# # print(block_to_graph.dstdata)
		# print('\nblock to graph.edata')
		# # print(block_to_graph.edata)
		# print()
		current_block_eidx, current_block_edges = get_global_graph_edges_ids(raw_graph, cur_block)
		block_to_graph.edata['_ID'] = current_block_eidx

		# current_block_graph = dgl.edge_subgraph(G, eidx)    # now, only 1-layer

		# print("bb.ndata[dgl.NID]['_N']")
		# # print(len(cur_block.ndata[dgl.NID]['_N']))
		# print()
		# print('block_to_graph.ndata[dgl.NID]')
		# # print(block_to_graph.ndata)
		# # print(len(current_block_graph.ndata[dgl.NID]))

		full, _ = torch.sort(block_to_graph.srcdata[dgl.NID])
		block_, _ = torch.sort(cur_block.ndata[dgl.NID]['_N'])

		if torch.equal(full, block_):  # make sure the new generated graph is exact the same with the current block

			'''-----------------------------------------------------------------------------------------------------------'''
			print()

			tt = time.time()
			# OUTPUT_NID = full_batch_data_blocks[-1].dstdata[dgl.NID]
			OUTPUT_NID, _ = torch.sort(full_batch_data_blocks[-1].dstdata[dgl.NID])
			print()
			temp = block_to_graph.dstdata

			print('full batch graph output nodes length')
			print(len(OUTPUT_NID.tolist()))
			print('block_to_graph.dstdata')
			# print(temp)

			batches_nid_list, weights_list = generate_random_mini_batch_seeds_list(OUTPUT_NID, args)
			t1 = time.time()
			print('after generate_random_mini_batch_seeds_list')
			print('time of batches_nid_list generation : ' + str(t1 - tt) + ' sec')

			data_loader = generate_blocks(raw_graph, block_to_graph, current_block_edges, current_block_eidx, batches_nid_list)
			return data_loader, weights_list

		else:
			print('*' * 40)
			print('transformation is not correct\n')

			print('\ncurrent block  to graph total  node length')
			print(len(block_to_graph.srcdata[dgl.NID]))
			print(block_to_graph.srcdata[dgl.NID])

			print('\ncurrent block  total  node length')
			print(len(cur_block.ndata[dgl.NID]['_N']))
			print(block_)

			return ()


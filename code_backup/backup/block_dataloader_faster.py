import torch
import dgl
import numpy


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
	# print(temp)   # [(),(),()...]
	res_ = list(map(list, zip(*temp))) # [],[]
	res=tuple(sub for sub in res_)
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
	mini_batch = args.batch_size
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	indices = torch.randperm(full_len)  # get a permutation of the index of output nid tensor (permutation of 0~n-1)
	# ### TODO
	# indices = torch.tensor([1, 2, 3, 0])
	# ### TODO
	batches_nid_list = []
	mod = int(full_len / mini_batch)  # mod = full_len / mini_batch
	for i in range(mod):
		idx = indices[i * mini_batch: (i + 1) * mini_batch]
		batches_nid_list.append(OUTPUT_NID[idx])

	if full_len % mini_batch!=0:  # tail_ = full_len % mini_batch
		tail_ = full_len % mini_batch
		idx = indices[-tail_:]
		batches_nid_list.append(OUTPUT_NID[idx])
		# print(batches_nid_list[-1])
	# batches_nid_list= [torch.tensor([20, 6]), torch.tensor([14, 7]), torch.tensor([2, 8]), torch.tensor([21, 3]), torch.tensor([9, 10]), torch.tensor([1, 16]),\
	#  torch.tensor([0, 5]), torch.tensor([11, 17]), torch.tensor([19, 23]), torch.tensor([18, 15]), torch.tensor([12, 4]), torch.tensor([22, 13])]
	# # print('batches_nid_list')
	# print(batches_nid_list)
	return batches_nid_list


def check_connection_nids(given_output_nids, full_batch_subgraph):

	f_g_nid_list = full_batch_subgraph.ndata['_ID'].tolist()
	given_nid_list = given_output_nids.tolist()
	local_given_output_nids = [f_g_nid_list.index(nid) for nid in given_nid_list]
	local_in_edges_tensor = full_batch_subgraph.in_edges(local_given_output_nids, form='all')

	srcid_local_list = list(local_in_edges_tensor)[0].tolist()
	nids_global_list = full_batch_subgraph.ndata['_ID'].tolist()
	srcid_list = list(numpy.array(nids_global_list)[srcid_local_list])

	eid_local_list = list(local_in_edges_tensor)[2].tolist()
	eids_global_list = full_batch_subgraph.edata['_ID'].tolist()
	eid_list = list(numpy.array(eids_global_list)[eid_local_list])

	global_eid_tensor = torch.tensor(eid_list, dtype=torch.long)

	srcid = torch.tensor(list(set(srcid_list)), dtype=torch.long)

	dstid = given_output_nids

	return srcid, dstid, global_eid_tensor




def check_connection_nids_bk(graph, given_output_nids, full_batch_subgraph):
	in_edges = graph.in_edges(given_output_nids, form='all')
	print('total in edges of given nids')
	print(in_edges)
	# in_edges

	f_g_nid_list = full_batch_subgraph.ndata['_ID'].tolist()
	given_nid_list = given_output_nids.tolist()
	print('global given nids')
	print(given_nid_list)
	local_given_output_nids = [f_g_nid_list.index(nid) for nid in given_nid_list]
	print('local output nids')
	print(local_given_output_nids)
	local_in_edges_tensor = full_batch_subgraph.in_edges(local_given_output_nids, form='all')
	# print('full_batch_subgraph.ndata')
	# print(full_batch_subgraph.ndata)
	print('local_in_edges_tensor')
	print(local_in_edges_tensor)

	srcid_local_list = list(local_in_edges_tensor)[0].tolist()
	print('--------------------------------------------------------------------------srcid_local_list')
	print(srcid_local_list)
	nids_global_list = full_batch_subgraph.ndata['_ID'].tolist()
	srcid_list = list(numpy.array(nids_global_list)[srcid_local_list])
	print('---------------------------------------------------------------------------global src nid list')
	print(srcid_list)



	eid_local_list = list(local_in_edges_tensor)[2].tolist()
	eids_global_list = full_batch_subgraph.edata['_ID'].tolist()
	eid_list = list(numpy.array(eids_global_list)[eid_local_list])
	print('---------------------------------------------------------------------------global src eid list')
	print(eid_list)

	global_eid_tensor = torch.tensor(eid_list, dtype=torch.long)

	# result_edges = graph.find_edges(global_eid_tensor)

	srcid = torch.tensor(list(set(srcid_list)), dtype=torch.long)

	dstid = given_output_nids

	print()
	print(srcid)
	print(dstid)
	print(global_eid_tensor)
	print()

	return srcid, dstid, global_eid_tensor


def check_connection_nids_2(graph, given_output_nids, subgraph_total_global_eids):
	'''

	Parameters
	----------
	graph: global graph                            dgl.graph
	given_output_nid : current output node id      long integer
	subgraph_total_global_eids:                     tensor()


	Returns
	-------
	srcid: tensor
	dstid: tensor
	eid_list_: tensor
	edges: [[],[]]
	'''

	global_eid_list_ = []
	in_edges = graph.in_edges(given_output_nids, form='all')
	print('total in edges of given nids')
	print(in_edges)
	# in_edges
	print('subgraph_total_global_eids len')
	print(len(subgraph_total_global_eids))
	print('len(in edges index [2])')
	print(len(list(in_edges)[2]))
	global_eid_list_ = [in_e for in_e in list(in_edges)[2] if in_e in subgraph_total_global_eids]
	# for in_e in list(in_edges)[2]:
	# 	# print('subgraph_total_global_eids len')
	# 	if in_e in subgraph_total_global_eids:
	# 		global_eid_list_.append(in_e)

	result_edges = graph.find_edges(global_eid_list_)
	global_eid_tensor = torch.tensor(global_eid_list_, dtype=torch.long)
	print('in edges in current subgraph')
	# print(global_eid_list_)
	print(global_eid_tensor)
	srcid = unique_tensor_item(list(result_edges)[0])
	dstid = unique_tensor_item(list(result_edges)[1])
	if not torch.equal(dstid.sort().values, given_output_nids.sort().values):
		print('error')
		print()
	dstid = unique_tensor_item(list(result_edges)[1])

	return srcid, dstid, global_eid_tensor, result_edges


def get_global_graph_edges_ids(local_edges, total_global_nids_of_subgraph):
	'''

		Parameters
		----------
		local_edges: (local nids, local nids): (tensor,tensor)

		total_global_nids_of_subgraph: tensor


		Returns
		-------
		global_graph_edges_ids: [[u],[v]]
		'''

	edges = list(local_edges)
	edges[0] = edges[0].tolist()
	edges[1] = edges[1].tolist()
	# print(' the edges src and dst node local ids')
	# print(edges[0])
	# print(edges[1])

	# print(sub_graph_global_eid)
	global_nid_list = total_global_nids_of_subgraph.tolist()
	ll = len(edges[0])
	global_0 = []
	global_1 = []
	for local_eid in edges[0]:
		global_0.append(global_nid_list[local_eid])
	for local_eid in edges[1]:
		global_1.append(global_nid_list[local_eid])

	global_graph_edges_list = [global_0, global_1]
	return global_graph_edges_list


def generate_blocks(G, full_batch_sub_graph, batches_nid_list):
	# sub_graph_total_nid = full_batch_sub_graph.ndata[dgl.NID]
	# sub_graph_total_eid = full_batch_sub_graph.edata[dgl.EID]
	# print('full batch subgraph EID')
	# print(sub_graph_total_eid)

	data_loader = []
	for step, nids in enumerate(batches_nid_list):
		# print('batch ' + str(step) + '-'*30)

		# srcnid, dstnid, connected_global_eid, connected_global_edges = check_connection_nids_2(G, nids, sub_graph_total_eid)
		srcnid, dstnid, connected_global_eid = check_connection_nids(nids, full_batch_sub_graph)
		cur_block = generate_one_block(G, connected_global_eid)
		# print('after generate_one_block')
		data_loader.append((srcnid, dstnid, [cur_block]))

	# print('length of data loader')
	# print(len(data_loader))
	return data_loader


def generate_one_block(G, eids):
	'''

	Parameters
	----------
	G    global graph                     DGLGraph
	eids  cur_batch_subgraph_global eid   tensor int64

	Returns
	-------

	'''
	cur_block_sub_graph = dgl.edge_subgraph(G, eids)
	dst_local_nid_list = cur_block_sub_graph.edges()[1].tolist()
	dst_local_nid_list = list(set(dst_local_nid_list))
	new_block = dgl.to_block(cur_block_sub_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))

	global_nid_list = cur_block_sub_graph.ndata[dgl.NID].tolist()
	block_nid_list = new_block.ndata[dgl.NID]['_N'].tolist()
	final_nid_list = [global_nid_list[i] for i in block_nid_list] # mapping global graph nid <--- block local nid
	new_block.ndata[dgl.NID] = {'_N': torch.tensor(final_nid_list, dtype=torch.long)} # the final nid list is global nid, then assign it to block
	# print(new_block.ndata)
	return new_block


def generate_dataloader( G, full_batch_data_blocks, OUTPUT_NID, args):
	full_batch_blocks_src_nid = torch.tensor([], dtype=torch.long)
	full_batch_blocks_dst_nid = torch.tensor([], dtype=torch.long)

	for bb in full_batch_data_blocks:

		# print('bb.edges()')
		# print(bb.edges())

		full_batch_blocks_src_nid = torch.cat((full_batch_blocks_src_nid, bb.srcdata[dgl.NID]))
		full_batch_blocks_dst_nid = torch.cat((full_batch_blocks_dst_nid, bb.dstdata[dgl.NID]))
		# print('full_batch_blocks_src_nid')
		# print(full_batch_blocks_src_nid.size())
		# print('full_batch_blocks_dst_nid')
		# print(full_batch_blocks_dst_nid.size())
		#
		print()
		# nidx = bb.ndata[dgl.NID]
		# print('current block ndis          777777777777777777777777777777777777777777777777777777777777')
		# print(nidx)
		# print('current block ndata')
		# print(bb.ndata)
		nid_list_edges = get_global_graph_edges_ids(bb.edges(), bb.ndata[dgl.NID]['_N'])
		eidx = G.edge_ids(nid_list_edges[0], nid_list_edges[1])   # https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids
		# print('current block eids           8 8 8 88 8888888888888888888888888888888888888888888888')
		# print(eidx)

		full_batch_sub_graph = dgl.edge_subgraph(G, eidx)    # now, only 1-layer
		# print('print full_batch_sub_graph ndata')
		# print('draw batch sub_graph ------------------------------')
		# print(full_batch_sub_graph.ndata)
		# draw_graph(full_batch_sub_graph)

		combined = torch.cat((full_batch_blocks_src_nid, full_batch_blocks_dst_nid))
		uniques, counts = combined.unique(return_counts=True)
		sub_graph_total_nid = uniques.type(torch.long)     # only collect nodes id, not node index order
		# print('sub_graph_total_nid.size()')
		# print(sub_graph_total_nid.size())

		if len(full_batch_sub_graph.ndata[dgl.NID]) != len(bb.ndata[dgl.NID]['_N']):
			print('transformation is not correct')
			print('generated from eidx sub_graph   node length')
			print(len(full_batch_sub_graph.ndata[dgl.NID]))

			print('current block  total  node length')
			print(len(bb.ndata[dgl.NID]))
		else:
			print()
			# print('compare bb dgl.NID v.s. full_batch sub graph dgl.NID')
			# print(bb.ndata[dgl.NID]['_N'].shape)
			# print(type(bb.ndata[dgl.NID]['_N']))
			# print(bb.ndata[dgl.NID]['_N'].tolist())
			# print(sorted(bb.ndata[dgl.NID]['_N'].tolist()))
			# print(full_batch_sub_graph.ndata[dgl.NID])
			# print(sorted(full_batch_sub_graph.ndata[dgl.NID].tolist()))
			# return
		'''-----------------------------------------------------------------------------------------------------------'''

		batches_nid_list = generate_random_mini_batch_seeds_list(OUTPUT_NID, args)
		# print('after generate_random_mini_batch_seeds_list')
		data_loader = generate_blocks(G, full_batch_sub_graph, batches_nid_list)
	return data_loader

import torch
import dgl
from dgl.distributed.dist_graph import DistGraph

from dgl.dataloading.dataloader import EdgeCollator, assign_block_eids
from dgl.dataloading import BlockSampler
from dgl.dataloading.pytorch import _pop_subgraph_storage, _pop_blocks_storage
from dgl.base import DGLError
import copy
import dgl.function as fn


class TestNodeCollator(dgl.dataloading.NodeCollator):
    '''
    Parameters
    ----------
    g : DGLGraph
        The global graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    '''
    def __init__(self, g, mini_batch_subgraph_eids_list):
        self.g = g
        self.is_distributed = isinstance(g, DistGraph)
        self.eids_list = mini_batch_subgraph_eids_list
        self.batch_graphs = self.generate_blocks(self.g, self.eids_list)
        nids = self.get_output_nids_from_eids(self.eids_list)
        self.nids = nids # output node ids
        self._dataset = nids

    def get_output_nids_from_eids(self, eids_list):#######
        '''

        Parameters
        ----------
        eids_list: [tensor(), tensor()]

        Returns
        nids: tensor
        -------

        '''
        # #################---------TODO----------#################################################
        nids_list =[]
        tmp = self.g.edges(form='all')
        tmp_list = list(tmp)
        graph_e_0 = tmp_list[0].tolist()
        graph_e_1 = tmp_list[1].tolist()
        graph_e_id = tmp_list[2].tolist()
        # graph_merged_list = [(graph_e_0[i], graph_e_1[i]) for i in range(0, len(graph_e_0))]
        #
        # e_0 = global_edges[0]
        # e_1 = global_edges[1]
        # merged_list = [(e_0[i], e_1[i]) for i in range(0, len(e_0))]
        print('total batch number is '+ str(len(eids_list)))
        for cur_batch_eids in eids_list:
            for e in cur_batch_eids.tolist():
                if e in graph_e_id:
                    idd = graph_e_id.index(e)
                    src = graph_e_0[idd]
                    dst = graph_e_1[idd]
                    if src not in nids_list:
                        nids_list.append(src)
                    if dst not in nids_list:
                        nids_list.append(src)

        nids = torch.tensor(nids_list)
        return nids


    def generate_blocks(self, g, eid_list):
        lenn = len(eid_list)
        batch_graphs = []
        for i in range(lenn):
            print('')
            eids = eid_list[i]
            cur_block_sub_graph = dgl.edge_subgraph(g, eids)
            print('min_batch sub_graph ------------------------------ block ' + str(i))
            # draw_graph(cur_block_sub_graph)
            print('dgl.NID')
            print(cur_block_sub_graph.ndata[dgl.NID])
            print(cur_block_sub_graph.ndata)
            print('dgl.EID')
            print(cur_block_sub_graph.edata[dgl.EID])
            print(cur_block_sub_graph.edata)

            batch_graphs.append(cur_block_sub_graph)
        return batch_graphs

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        """
        The interface of collator, input items is edge id of the attached graph
        """
        print('--------------------------------TestNodeCollator collate---------------------------------------')
        print('items')
        print(items)
        print('self.batch_graphs')
        print(self.batch_graphs)
        blocks = self.batch_graphs
        print('blocks')
        print(blocks)
        output_nodes = blocks[-1].dstdata['_ID']
        input_nodes = blocks[0].srcdata['_ID']
        print()
        return input_nodes, output_nodes, blocks


class Pseudo_DataLoader(dgl.dataloading.NodeDataLoader):

    def __init__(self, g, eids_list,  device='cpu', collator=TestNodeCollator, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v
        self.collator = collator(g, eids_list)
        self.is_distributed = False
        self.dataloader = torch.utils.data.DataLoader(
            self.collator.dataset, collate_fn=self.collator.collate, **dataloader_kwargs)
        self.device = device

        # Precompute the CSR and CSC representations so each subprocess does not
        # duplicate.
        if dataloader_kwargs.get('num_workers', 0) > 0:
            g.create_formats_()

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader)


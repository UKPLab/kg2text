import torch
from torch_geometric.data import Data, Batch


def get_embs_graph(batch, embeddings):

    use_cuda = batch.src[0].is_cuda
    list_geometric_data = []

    for idx, emb in enumerate(embeddings):
        edges_index = batch.graph[idx][0]
        edges_types = batch.graph[idx][1]

        data = Data(x=emb, edge_index=edges_index, y=edges_types)
        list_geometric_data.append(data)

    bdl = Batch.from_data_list(list_geometric_data)
    if use_cuda:
        bdl = bdl.to('cuda:' + str(torch.cuda.current_device()))

    return bdl







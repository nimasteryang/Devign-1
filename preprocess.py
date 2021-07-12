import networkx as nx
import numpy as np
import pandas as pd
import torch

from lexical_parser import lexical_parser
from gensim.models.word2vec import Word2Vec
import pygraphviz
# with pygraphviz
def dot_to_graph(_dot):
    try:
        G_non_label = nx.nx_agraph.from_agraph(pygraphviz.AGraph(_dot))
        G_out = nx.relabel_nodes(G_non_label, lambda x: int(x) - 1000100)
        return G_out
    except:
        return ''


def gen_node_fea(_CPG, _w2v_model):
    node_fea_list = []
    for node in _CPG.nodes:
        _node_dict = _CPG.nodes[node]
        if _node_dict.get('label') is None:
            _node_fea_vec = [np.zeros(100)]
        else:
            _node_code = _node_dict.get('label')
            _node_tokens = lexical_parser(_node_code.split(',', 1)[1])
            # if _node_tokens:
            _node_fea_vec = [_w2v_model.wv[i] for i in _node_tokens if i in _w2v_model.wv.key_to_index]
            if not _node_fea_vec:
                _node_fea_vec.append(np.zeros(100))
            # _node_fea_dict[_node] = np.array(_node_fea_vec)
            # continue
        arrs = np.array(_node_fea_vec)
        arrs = arrs[~np.isnan(arrs).any(axis=1)]
        mean = np.mean(arrs, axis=0)
        node_fea_list.append(mean)
    nparr = np.stack(node_fea_list)
    # pytensor = torch.from_numpy(nparr).type(torch.FloatTensor)
    # return pytensor
    return nparr

def gen_graph(_NX_CPG):
    ast_edges = [(u,0,v) for u, v, d in _NX_CPG.edges(data=True) if "AST" in d['label']]
    cfg_edges = [(u, 1, v) for u, v, d in _NX_CPG.edges(data=True) if "CFG" in d['label']]
    dfg_edges = [(u, 2, v) for u, v, d in _NX_CPG.edges(data=True) if "DDG" in d['label']]
    edges = ast_edges + cfg_edges + dfg_edges
    return edges

def process():
    input_graph_file = "/Users/xuyang/Documents/GitHub/icse2021_repo/data/graph_data/devign_output_full.json"
    df = pd.read_json(input_graph_file)
    print(df.info())
    df = df[df.No2St != 'No Node']
    df = df[df.CPG != 'No Graph']
    df = df[df.CPG != '']
    df['token'] = df.func.apply(lexical_parser)
    w2v_model = Word2Vec(df.token, min_count=1)
    df['NX_CPG'] = df.apply(lambda row: dot_to_graph(row.CPG), axis=1)
    df = df[df.NX_CPG != '']
    df['node_features'] = df.apply(lambda row: gen_node_fea(row.NX_CPG,w2v_model), axis=1)
    df['graph'] = df.apply(lambda row: gen_graph(row.NX_CPG), axis=1)
    df['target'] = df['target'].astype(int)
    df2 = df[['node_features','graph','target']]
    print(df2.info())
    traindf = df2.iloc[0:15722]
    validdf = df2.iloc[15723:17688]
    testdf = df2.iloc[17689:19653]
    print(traindf.info())
    print(validdf.info())
    print(testdf.info())
    traindf.to_json("input/train_GGNNinput.json",orient="records")
    validdf.to_json("input/valid_GGNNinput.json", orient="records")
    testdf.to_json("input/test_GGNNinput.json", orient="records")
if __name__ == '__main__':
    process()
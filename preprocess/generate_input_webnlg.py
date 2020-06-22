import sys
from xml.dom import minidom
from pathlib import Path
import re
import os

folder_source = sys.argv[1]
folder_preprocessed_files = sys.argv[2]

if not os.path.exists(folder_preprocessed_files):
    os.makedirs(folder_preprocessed_files)

datasets = ['train', 'dev', 'test']


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            new_d.append(t.lower())
    return new_d


def get_nodes(n):
    n = n.strip()
    n = n.replace('(', '')
    n = n.replace('\"', '')
    n = n.replace(')', '')
    n = n.replace(',', ' ')
    n = n.replace('_', ' ')

    n = ' '.join(re.split('(\W)', n))
    n = n.lower()
    n = n.split()

    return n


def get_relation(n):
    n = n.replace('(', '')
    n = n.replace(')', '')
    n = n.strip()
    n = n.split()
    n = "_".join(n)
    return n


def add_nodes(nodes, entity, original_node, all_nodes):
    if entity not in original_node:
        original_node[entity] = {}
        original_node[entity]['idx'] = len(original_node) - 1
        original_node[entity]['words'] = {}
        for n in nodes:
            all_nodes.append(n)
            original_node[entity]['words'][n] = len(all_nodes) - 1
        all_nodes.append('</entity>')


def add_nodes_bpe(nodes, entity, original_node, all_nodes):
    if entity not in original_node:
        original_node[entity] = {}
        original_node[entity]['idx'] = len(original_node) - 1
        all_nodes.append('<entity>')
        original_node[entity]['words'] = {}
        for n in nodes:
            all_nodes.append(n)
            original_node[entity]['words'][n] = len(all_nodes) - 1
        all_nodes.append('</entity>')


def process_triples(mtriples):

    original_node = {}
    triples = []
    nodes = []
    adj_matrix = []

    for m in mtriples:

        ms = m.firstChild.nodeValue
        ms = ms.strip().split(' | ')
        n1 = ms[0]
        n2 = ms[2]
        nodes1 = get_nodes(n1)
        nodes2 = get_nodes(n2)
        add_nodes(nodes1, n1, original_node, nodes)
        edge = get_relation(ms[1])

        edge_split = camel_case_split(edge)
        add_nodes(edge_split, edge, original_node, nodes)

        add_nodes(nodes2, n2, original_node, nodes)

        triples.append([edge, original_node[n1]['idx'], original_node[n2]['idx'],
                        original_node[edge]['idx']])


    return nodes, adj_matrix, triples


def get_data_dev_test(file_, train_cat):

    datapoints = []
    all_tripes = []
    cats = set()

    xmldoc = minidom.parse(file_)
    entries = xmldoc.getElementsByTagName('entry')
    for e in entries:
        cat = e.getAttribute('category')
        cats.add(cat)

        if cat in train_cat:

            mtriples = e.getElementsByTagName('mtriple')
            nodes, adj_matrix, triples = process_triples(mtriples)
            all_tripes.append(triples)

            lexs = e.getElementsByTagName('lex')

            surfaces = []
            for l in lexs:
                l = l.firstChild.nodeValue.strip().lower()
                new_doc = ' '.join(re.split('(\W)', l))
                new_doc = ' '.join(new_doc.split())
                surfaces.append(new_doc)
            datapoints.append((nodes, adj_matrix, surfaces))

    return datapoints, cats, all_tripes

def get_data(file_):

    datapoints = []
    all_tripes = []

    cats = set()

    xmldoc = minidom.parse(file_)
    entries = xmldoc.getElementsByTagName('entry')
    for e in entries:
        cat = e.getAttribute('category')
        cats.add(cat)

        mtriples = e.getElementsByTagName('mtriple')
        nodes, adj_matrix, triples = process_triples(mtriples)

        lexs = e.getElementsByTagName('lex')

        for l in lexs:
            l = l.firstChild.nodeValue.strip().lower()
            new_doc = ' '.join(re.split('(\W)', l))
            new_doc = ' '.join(new_doc.split())
            datapoints.append((nodes, adj_matrix, new_doc))
            all_tripes.append(triples)

    return datapoints, cats, all_tripes


def process_bpe(triples, file_, file_new, file_graph_new):
    f = open(file_, 'r').readlines()

    datapoints = []

    print('processing', len(triples), 'triples')
    assert len(f) == len(triples)

    for idx, t in enumerate(triples):
        original_node = {}
        nodes = []
        adj_matrix = []

        l = f[idx]
        l = l.strip().split('</entity>')

        nodes_file = []

        for e in l:
            e = e.lower().strip()
            e_split = e.split()
            e_split = list(filter(None, e_split))
            nodes_file.append((e_split, e))

        for triple in t:
            edge, e1, e2, rel = triple
            n1 = nodes_file[e1][1]
            n2 = nodes_file[e2][1]
            add_nodes_bpe(nodes_file[e1][0], n1, original_node, nodes)

            edges_idx = []
            nodes.append('<relation>')
            for e in nodes_file[rel][0]:
                nodes.append(e)
                edges_idx.append(len(nodes) - 1)
            nodes.append('</relation>')
            add_nodes_bpe(nodes_file[e2][0], n2, original_node, nodes)

            for k in original_node[n1]['words'].keys():
                for edge_idx in edges_idx:
                    l = '(' + str(original_node[n1]['words'][k])
                    l += ',' + str(edge_idx) + ',0,0)'
                    adj_matrix.append(l)

                    l = '(' + str(edge_idx)
                    l += ',' + str(original_node[n1]['words'][k]) + ',2,2)'
                    adj_matrix.append(l)

            for k in original_node[n2]['words'].keys():
                for edge_idx in edges_idx:
                    l = '(' + str(edge_idx)
                    l += ',' + str(original_node[n2]['words'][k]) + ',1,1)'
                    adj_matrix.append(l)

                    l = '(' + str(original_node[n2]['words'][k])
                    l += ',' + str(edge_idx) + ',3,3)'
                    adj_matrix.append(l)

        datapoints.append((nodes, adj_matrix))

    f_new = open(file_new, 'w')
    f_graph_new = open(file_graph_new, 'w')
    nodes = []
    graphs = []
    for dp in datapoints:
        nodes.append(' '.join(dp[0]))
        graphs.append(' '.join(dp[1]))

    f_new.write('\n'.join(nodes))
    f_new.close()
    f_graph_new.write('\n'.join(graphs))
    f_graph_new.close()
    print('done')


triples = {}
train_cat = set()
dataset_points = []
for d in datasets:
    triples[d] = []
    datapoints = []
    all_cats = set()
    files = Path(folder_source + d).rglob('*.xml')
    for idx, filename in enumerate(files):
        filename = str(filename)
        if d == 'test' and 'testdata_with_lex.xml' not in filename:
            continue

        if d == 'train':
            datapoint, cats, tripes = get_data(filename)
        else:
            datapoint, cats, tripes = get_data_dev_test(filename, train_cat)
        all_cats.update(cats)
        datapoints.extend(datapoint)
        triples[d].extend(tripes)
    if d == 'train':
        train_cat = all_cats
    print(d, len(datapoints))
    print(d, ' -> triples:', len(triples[d]))
    assert len(datapoints) == len(triples[d])
    print('number of cats:', len(all_cats))
    print('cats:', all_cats)
    dataset_points.append(datapoints)


path = folder_preprocessed_files
for idx, datapoints in enumerate(dataset_points):

    part = datasets[idx]
    nodes = []
    graphs = []
    surfaces = []
    surfaces_2 = []
    surfaces_3 = []
    for datapoint in datapoints:
        nodes.append(' '.join(datapoint[0]))
        graphs.append(' '.join(datapoint[1]))
        if part != 'train':
            surfaces.append(datapoint[2][0])
            if len(datapoint[2]) > 1:
                surfaces_2.append(datapoint[2][1])
            else:
                surfaces_2.append('')
            if len(datapoint[2]) > 2:
                surfaces_3.append(datapoint[2][2])
            else:
                surfaces_3.append('')
        else:
            surfaces.append(datapoint[2])

    with open(path + '/' + part + '-src.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(nodes))
    with open(path + '/' + part + '-surfaces.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(surfaces))
    if part != 'train':
        with open(path + '/' + part + '-surfaces-2.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_2))
        with open(path + '/' + part + '-surfaces-3.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_3))

num_operations = 2000
os.system('cat ' + path + '/train-src.txt ' + path + '/train-surfaces.txt > ' +
          path + '/training_source.txt')
print('creating bpe codes...')
os.system('subword-nmt learn-bpe -s ' + str(num_operations) + ' < ' +
                path + '/training_source.txt > ' + path + '/codes-bpe.txt')
print('done')
print('converting files to bpe...')

for d in datasets:
    file_pre = path + '/' + d + '-src.txt'
    file_ = path + '/' + d + '-src-bpe.txt'
    os.system('subword-nmt apply-bpe -c ' + path + '/codes-bpe.txt < ' + file_pre + ' > ' + file_)

    file_pre = path + '/' + d + '-surfaces.txt'
    file_ = path + '/' + d + '-surfaces-bpe.txt'
    os.system('subword-nmt apply-bpe -c ' + path + '/codes-bpe.txt < ' + file_pre + ' > ' + file_)
print('done')

for d in datasets:
    print('dataset:', d)
    file_ = path + '/' + d + '-src-bpe.txt'
    file_new = path + '/' + d + '-nodes.txt'
    file_graph_new = path + '/' + d + '-graph.txt'
    process_bpe(triples[d], file_, file_new, file_graph_new)


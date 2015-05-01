'''Utility functions and classes for higher level RawDataWrapper access'''
import itertools
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE
from neurom.core.dataformat import ROOT_ID
from neurom.core.tree import Tree
from neurom.core.neuron import Neuron


def get_soma_ids(rdw):
    '''Returns a list of IDs of points that are somas'''
    return rdw.get_ids(lambda r: r[COLS.TYPE] == POINT_TYPE.SOMA)


def get_initial_segment_ids(rdw):
    '''Returns a list of IDs of initial tree segments

    These are defined as non-soma points whose perent is a soma point.
    '''
    l = list(itertools.chain(*[rdw.get_children(s) for s in get_soma_ids(rdw)]))
    return [i for i in l if rdw.get_row(i)[COLS.TYPE] != POINT_TYPE.SOMA]


def make_tree(rdw, root_id=ROOT_ID):
    '''Return a tree obtained from a raw data block

    The tree contains rows of raw data.
    '''
    def add_children(t):
        '''Add children to a tree'''
        for c in rdw.get_children(t.value[COLS.ID]):
            child = Tree(rdw.get_row(c))
            t.add_child(child)
            add_children(child)
        return t

    head_node = Tree(rdw.get_row(root_id))
    return add_children(head_node)


def make_neuron(raw_data):
    '''Build a neuron from a raw data block'''
    _trees = [make_tree(raw_data, iseg)
              for iseg in get_initial_segment_ids(raw_data)]
    _soma_pts = [raw_data.get_row(s_id) for s_id in get_soma_ids(raw_data)]
    return Neuron(_soma_pts, _trees)
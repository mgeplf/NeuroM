#!/usr/bin/env python

import logging
import sys
import math

from contextlib import contextmanager

from collections import namedtuple, OrderedDict
from types import StringTypes

import neurom as nm
import numpy as np

INDENT = ' ' * 2
PT_FMT = '(%8.2f %8.2f %8.2f %8.2f)\n'
type_mapping = {
    nm.NeuriteType.apical_dendrite: 'Apical',
    nm.NeuriteType.axon: 'Axon',
    nm.NeuriteType.basal_dendrite: 'Dendrite',
}


@contextmanager
def _block(write, title='', indent=INDENT):
    def indenter(line):
        if line.strip(' ') == '\n':
            write(line)
            return
        write(indent + line)
    write('(' + title)
    write('\n')
    yield indenter
    write(')')
    write('\n')


def write_asc(morph, fd):
    fd.write(';File generated using NeuroM\n')

    #write soma
    with _block(fd.write, title='"CellBody"') as write:
        write('(CellBody)\n')
        for p in morph.soma.points:
            write(PT_FMT % tuple(p))

    fd.write('\n')

    def write_section(write, section):
        for p in section.points:
            write(PT_FMT % tuple(p[:4]))

        if section.children:
            with _block(write) as write:
                for child in section.children:
                    write_section(write, child)
                    if child != section.children[-1]:
                        write('|\n')

    for neurite in nm.iter_neurites(morph):
        title = '(' + type_mapping[neurite.type] + ')'
        with _block(fd.write, title=title) as write:
            write_section(write, neurite.root_node)
        fd.write('\n' * 2)

def main():
    logging.basicConfig(level=logging.DEBUG)

    p = 'sample.asc'
    morph = nm.load_neuron(p)

    with open('Fluo41_left.asc', 'w') as fd:
        write_asc(morph, fd)

if __name__ == '__main__':
    main()

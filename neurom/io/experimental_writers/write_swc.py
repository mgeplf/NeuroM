#!/usr/bin/env python

import argparse
import logging

import neurom as nm
from neurom.core import iter_sections
from neurom.core.dataformat import COLS
from neurom.core.types import NeuriteType

type_mapping = {
    NeuriteType.soma: 1,
    NeuriteType.apical_dendrite: 4,
    NeuriteType.axon: 2,
    NeuriteType.basal_dendrite: 3,
}
ROW_FMT = '{id:d} {type:d} {x:.2f} {y:.2f} {z:.2f} {r:.2f} {parent_id:d}\n'


def write_swc(morph, fd):
    fd.write('#File generated using NeuroM\n')

    # write spherical soma
    x, y, z = morph.soma.center
    fd.write(ROW_FMT.format(
        id=1, type=type_mapping[NeuriteType.soma],
        x=x, y=y, z=z, r=morph.soma.radius,
        parent_id=-1
    ))

    section_to_row_id = {None: 1, }

    row_id = 2
    for section in iter_sections(morph):
        type_ = type_mapping[section.type]
        pid = section.parent if section.parent is None else section.parent.id
        pid = section_to_row_id[pid]

        for point in section.points:
            x, y, z, r = point[COLS.XYZR]
            fd.write(ROW_FMT.format(
                id=row_id, type=type_, x=x, y=y, z=z, r=r, parent_id=pid
            ))
            pid = row_id
            row_id += 1

        section_to_row_id[section.id] = row_id - 1


def main():
    import sys

    logging.basicConfig(level=logging.DEBUG)

    morph = nm.load_neuron(sys.argv[1])

    with open(sys.argv[2], 'w') as fd:
        write_swc(morph, fd)


if __name__ == '__main__':
    main()

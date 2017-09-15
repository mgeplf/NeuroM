#!/usr/bin/env python

import h5py
import numpy as np
import neurom as nm
from neurom.fst._neuritefunc import n_sections, n_segments
from neurom.core import iter_sections
from neurom.core.dataformat import COLS
from neurom.core.types import NeuriteType

type_to_h5_type = {
    NeuriteType.soma: 1,
    NeuriteType.axon: 2,
    NeuriteType.basal_dendrite: 3,
    NeuriteType.apical_dendrite: 4,
}


def _create_h5_data(morph):
    section_count = n_sections(morph)  # does not include soma
    points = np.zeros((n_segments(morph) + len(morph.soma.points) + section_count, 4), dtype=float)
    structure = np.zeros((section_count + 1, 3), dtype=np.int32)

    # copy soma
    points[0:len(morph.soma.points), :] = morph.soma.points[:, :]
    structure_insert = 0
    structure[structure_insert] = [0, 1, -1]
    structure_insert += 1

    # copy sections
    # if parent is None, that means they are connected to the start of the soma
    section_to_insert_point = {None: 0, }

    insert_point = len(morph.soma.points)
    for section in iter_sections(morph):
        n = len(section.points)
        points[insert_point:insert_point + n, :] = section.points[:, COLS.XYZR]

        parent_id = section.parent if section.parent is None else section.parent.id
        structure[structure_insert] = [
            insert_point, type_to_h5_type[section.type], section_to_insert_point[parent_id]]
        section_to_insert_point[section.id] = structure_insert
        insert_point += n
        structure_insert += 1

    # radius -> diameter
    points[:, COLS.R] *= 2

    return points, structure


def write_h5_v1(morph, h5):
    points, structure = _create_h5_data(morph)
    h5.create_dataset('points', data=points, dtype='float')
    h5.create_dataset('structure', data=structure)


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    #p = 'small_morph.h5'
    #p = 'C010306G_no_recenter.h5'
    p = 'rp160229_A_idA.ASC'
    morph = nm.load_neuron(p)

    with h5py.File('test.h5', 'w') as h5:
        write_h5_v1(morph, h5)

if __name__ == '__main__':
    main()

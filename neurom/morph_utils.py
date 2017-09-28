'''utilities for changing morphologies'''
import math

import neurom as nm
import numpy as np

from neurom.io.datawrapper import BlockNeuronBuilder
from neurom.core.dataformat import POINT_TYPE, COLS, ROOT_ID
from neurom.fst._core import FstNeuron


def unravel_neuron(morph, window_half_length=5):
    '''Perform unravelling as described in
    DOI: 10.7551/mitpress/9780262013277.001.0001
    Section: 9.2 Repair of Neuronal Dendrites
    '''
    neuron_builder = BlockNeuronBuilder()
    neuron_builder.add_section(0, 0, POINT_TYPE.SOMA, morph.soma.points)

    for sec in nm.iter_sections(morph.neurites):
        points = sec.points[:, COLS.XYZR]
        segment_count = len(points)
        new_points = np.copy(points)

        if sec.parent is not None:
            new_points[0, :] = neuron_builder.sections[sec.parent.id].points[-1, :]

        for window_center in range(1, segment_count):
            window_start = max(0, window_center - window_half_length - 1)
            window_end = min(segment_count, window_center + window_half_length + 1)

            # use the principal eigenvector as the starting point
            X = np.copy(points[window_start:window_end, COLS.XYZ])
            X -= np.mean(X, axis=0)
            C = np.dot(X.T, X)
            w, v = np.linalg.eig(C)
            start = v[:, w.argmax()]

            # point it in the same direction as before
            window_diff = (points[window_end - 1, COLS.XYZ] -
                           points[window_start, COLS.XYZ])
            scale = np.dot(start, window_diff)
            start *= scale
            start /= math.sqrt(np.dot(start, start))

            # make it span length the same as the original segment within the window
            seg_diff = (points[window_center - 1, COLS.XYZ] -
                        points[window_center, COLS.XYZ])
            length = math.sqrt(np.dot(seg_diff, seg_diff))
            start *= length

            new_points[window_center, COLS.XYZ] = start + new_points[window_center - 1, COLS.XYZ]

        neuron_builder.add_section(sec.id,
                                   sec.parent.id if sec.parent is not None else 0,
                                   sec.type.value - 1,
                                   new_points)

    return FstNeuron(neuron_builder.get_datawrapper('NL-ASCII'), 'unraveled_neuron')


def points_simplify(points, epsilon):
    '''use Ramer-Douglas-Peucker to simplify the segments'''
    def dist_line2point(x0, start, end):
        '''distance of x0 from line defined by start, to end
            http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        '''
        diff_start_end = end - start
        return np.divide(np.linalg.norm(np.cross(diff_start_end, start - x0)),
                         np.linalg.norm(diff_start_end))

    def _douglas_peucker(point_list):
        max_dist = 0.0
        index = -1

        for i in xrange(1, len(point_list)):
            start, end = point_list[0, COLS.XYZ], point_list[-1, COLS.XYZ]
            x0 = point_list[i, COLS.XYZ]
            dist = dist_line2point(x0, start, end)

            if max_dist < dist:
                index = i
                max_dist = dist

        if epsilon < max_dist:
            r1 = douglas_peucker(point_list[:index + 1])
            r2 = douglas_peucker(point_list[index:])
            return np.vstack((r1[:-1], r2))
        else:
            return np.vstack((point_list[0], point_list[-1]))

    return douglas_peucker(points)


def simplify_neuron(morph):
    neuron_builder = BlockNeuronBuilder()
    neuron_builder.add_section(0, 0, POINT_TYPE.SOMA, morph.soma.points)
    for sec in nm.iter_sections(morph.neurites):
        points = points_simplify(sec.points[:, COLS.XYZR], 1)
        neuron_builder.add_section(sec.id,
                                   sec.parent.id if sec.parent is not None else 0,
                                   sec.type.value - 1,
                                   points)
    return FstNeuron(neuron_builder.get_datawrapper('NL-ASCII'), 'simplified_neuron')


if __name__ == '__main__':
    p = 'Fluo41_left.h5'
    morph = nm.load_neuron(p)
    new_morph = unravel_neuron(morph)
    viewer.draw(morph)[0].savefig('morph.png')
    viewer.draw(new_morph)[0].savefig('new_morph.png')

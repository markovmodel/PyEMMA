
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


r"""User-API for the pyemma.coordinates package

.. currentmodule:: pyemma.coordinates.api
"""
from pathlib import Path

import numpy as _np
import logging as _logging

from pyemma.util import types as _types
# lift this function to the api
from pyemma.coordinates.util.stat import histogram

from pyemma.util.exceptions import PyEMMA_DeprecationWarning as _PyEMMA_DeprecationWarning

_logger = _logging.getLogger(__name__)

__docformat__ = "restructuredtext en"
__author__ = "Frank Noe, Martin Scherer"
__copyright__ = "Copyright 2015, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['featurizer',  # IO
           'load',
           'source',
           'combine_sources',
           'histogram',
           'pipeline',
           'discretizer',
           'save_traj',
           'save_trajs',
           'pca',  # transform
           'tica',
           'tica_nystroem',
           'vamp',
           'covariance_lagged',
           'cluster_regspace',  # cluster
           'cluster_kmeans',
           'cluster_mini_batch_kmeans',
           'cluster_uniform_time',
           'assign_to_centers',
           ]

_string_types = str

# ==============================================================================
#
# DATA PROCESSING
#
# ==============================================================================

def _check_old_chunksize_arg(chunksize, chunk_size_default, **kw):
    # cases:
    # 1. chunk_size not given, return chunksize
    # 2. chunk_size given, chunksize is default, warn, return chunk_size
    # 3. chunk_size and chunksize given, warn, return chunksize
    chosen_chunk_size = NotImplemented
    deprecated_arg_given = 'chunk_size' in kw
    is_default = chunksize == chunk_size_default

    if not deprecated_arg_given:  # case 1.
        chosen_chunk_size = chunksize
    else:
        import warnings
        from pyemma.util.annotators import get_culprit
        filename, lineno = get_culprit(3)
        if is_default:  # case 2.
            warnings.warn_explicit('Passed deprecated argument "chunk_size", please use "chunksize"',
                                   category=_PyEMMA_DeprecationWarning, filename=filename, lineno=lineno)
            chosen_chunk_size = kw.pop('chunk_size')  # remove this argument to avoid further passing to other funcs.
        else:  # case 3.
            warnings.warn_explicit('Passed two values for chunk size: "chunk_size" and "chunksize", while the first one'
                                   ' is deprecated. Please use "chunksize" in the future.',
                                   category=_PyEMMA_DeprecationWarning, filename=filename, lineno=lineno)
            chosen_chunk_size = chunksize
    assert chosen_chunk_size is not NotImplemented
    return chosen_chunk_size


def featurizer(topfile):
    r""" Featurizer to select features from MD data.

    Parameters
    ----------
    topfile : str or mdtraj.Topology instance
        path to topology file (e.g pdb file) or a mdtraj.Topology object

    Returns
    -------
    feat : :class:`Featurizer <pyemma.coordinates.data.featurization.featurizer.MDFeaturizer>`

    Examples
    --------

    Create a featurizer and add backbone torsion angles to active features.
    Then use it in :func:`source`

    >>> import pyemma.coordinates # doctest: +SKIP
    >>> feat = pyemma.coordinates.featurizer('my_protein.pdb') # doctest: +SKIP
    >>> feat.add_backbone_torsions() # doctest: +SKIP
    >>> reader = pyemma.coordinates.source(["my_traj01.xtc", "my_traj02.xtc"], features=feat) # doctest: +SKIP

    or

    >>> traj = mdtraj.load('my_protein.pdb') # # doctest: +SKIP
    >>> feat = pyemma.coordinates.featurizer(traj.topology) # doctest: +SKIP

    .. autoclass:: pyemma.coordinates.data.featurization.featurizer.MDFeaturizer
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.data.featurization.featurizer.MDFeaturizer
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.data.featurization.featurizer.MDFeaturizer
            :attributes:
    """
    from pyemma.coordinates.data.featurization.featurizer import MDFeaturizer
    return MDFeaturizer(topfile)


# TODO: DOC - which topology file formats does mdtraj support? Find out and complete docstring
def load(trajfiles, features=None, top=None, stride=1, chunksize=None, **kw):
    r""" Loads coordinate features into memory.

    If your memory is not big enough consider the use of **pipeline**, or use
    the stride option to subsample the data.

    Parameters
    ----------
    trajfiles : str, list of str or nested list (one level) of str
        A filename or a list of filenames to trajectory files that can be
        processed by pyemma. Both molecular dynamics trajectory files and raw
        data files (tabulated ASCII or binary) can be loaded.

        If a nested list of filenames is given, eg.:
            [['traj1_0.xtc', 'traj1_1.xtc'], 'traj2_full.xtc'], ['traj3_0.xtc, ...]]
        the grouped fragments will be treated as a joint trajectory.

        When molecular dynamics trajectory files are loaded either a featurizer
        must be specified (for reading specific quantities such as distances or
        dihedrals), or a topology file (in that case only Cartesian coordinates
        will be read). In the latter case, the resulting feature vectors will
        have length 3N for each trajectory frame, with N being the number of
        atoms and (x1, y1, z1, x2, y2, z2, ...) being the sequence of
        coordinates in the vector.

        Molecular dynamics trajectory files are loaded through mdtraj (http://mdtraj.org/latest/),
        and can possess any of the mdtraj-compatible trajectory formats
        including:

           * CHARMM/NAMD (.dcd)
           * Gromacs (.xtc)
           * Gromacs (.trr)
           * AMBER (.binpos)
           * AMBER (.netcdf)
           * TINKER (.arc),
           * MDTRAJ (.hdf5)
           * LAMMPS trajectory format (.lammpstrj)

        Raw data can be in the following format:

           * tabulated ASCII (.dat, .txt)
           * binary python (.npy, .npz)

    features : MDFeaturizer, optional, default = None
        a featurizer object specifying how molecular dynamics files should
        be read (e.g. intramolecular distances, angles, dihedrals, etc).

    top : str, mdtraj.Trajectory or mdtraj.Topology, optional, default = None
        A molecular topology file, e.g. in PDB (.pdb) format or an already
        loaded mdtraj.Topology object. If it is an mdtraj.Trajectory object, the topology
        will be extracted from it.

    stride : int, optional, default = 1
        Load only every stride'th frame. By default, every frame is loaded

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    Returns
    -------
    data : ndarray or list of ndarray
        If a single filename was given as an input (and unless the format is
        .npz), the return will be a single ndarray of size (T, d), where T is
        the number of time steps in the trajectory and d is the number of features
        (coordinates, observables). When reading from molecular dynamics data
        without a specific featurizer, each feature vector will have size d=3N
        and will hold the Cartesian coordinates in the sequence
        (x1, y1, z1, x2, y2, z2, ...).
        If multiple filenames were given, or if the file is a .npz holding
        multiple arrays, the result is a list of appropriately shaped arrays

    See also
    --------
    :func:`pyemma.coordinates.source`
        if your memory is not big enough, specify data source and put it into your
        transformation or clustering algorithms instead of the loaded data. This
        will stream the data and save memory on the cost of longer processing
        times.

    Examples
    --------

    >>> from pyemma.coordinates import load
    >>> files = ['traj01.xtc', 'traj02.xtc'] # doctest: +SKIP
    >>> output = load(files, top='my_structure.pdb') # doctest: +SKIP

    """
    from pyemma.coordinates.data.util.reader_utils import create_file_reader
    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(load)['chunksize'], **kw)
    if isinstance(trajfiles, _string_types) or (
        isinstance(trajfiles, (list, tuple))
            and (any(isinstance(item, (list, tuple, str)) for item in trajfiles)
                 or len(trajfiles) == 0)):
        reader = create_file_reader(trajfiles, top, features, chunksize=cs, **kw)
        trajs = reader.get_output(stride=stride)
        if len(trajs) == 1:
            return trajs[0]
        else:
            return trajs
    else:
        raise ValueError('unsupported type (%s) of input' % type(trajfiles))


def source(inp, features=None, top=None, chunksize=None, **kw):
    r""" Defines trajectory data source

    This function defines input trajectories without loading them. You can pass
    the resulting object into transformers such as :func:`pyemma.coordinates.tica`
    or clustering algorithms such as :func:`pyemma.coordinates.cluster_kmeans`.
    Then, the data will be streamed instead of being loaded, thus saving memory.

    You can also use this function to construct the first stage of a data
    processing :func:`pipeline`.

    Parameters
    ----------
    inp : str (file name) or ndarray or list of strings (file names) or list of ndarrays or nested list of str|ndarray (1 level)
        The inp file names or input data. Can be given in any of
        these ways:

        1. File name of a single trajectory. It can have any of the molecular
           dynamics trajectory formats or raw data formats specified in :py:func:`load`.
        2. List of trajectory file names. It can have any of the molecular
           dynamics trajectory formats or raw data formats specified in :py:func:`load`.
        3. Molecular dynamics trajectory in memory as a numpy array of shape
           (T, N, 3) with T time steps, N atoms each having three (x,y,z)
           spatial coordinates.
        4. List of molecular dynamics trajectories in memory, each given as a
           numpy array of shape (T_i, N, 3), where trajectory i has T_i time
           steps and all trajectories have shape (N, 3).
        5. Trajectory of some features or order parameters in memory
           as a numpy array of shape (T, N) with T time steps and N dimensions.
        6. List of trajectories of some features or order parameters in memory,
           each given as a numpy array of shape (T_i, N), where trajectory i
           has T_i time steps and all trajectories have N dimensions.
        7. List of NumPy array files (.npy) of shape (T, N). Note these
           arrays are not being loaded completely, but mapped into memory
           (read-only).
        8. List of tabulated ASCII files of shape (T, N).
        9. Nested lists (1 level) like), eg.:
                [['traj1_0.xtc', 'traj1_1.xtc'], ['traj2_full.xtc'], ['traj3_0.xtc, ...]]
           the grouped fragments will be treated as a joint trajectory.

    features : MDFeaturizer, optional, default = None
        a featurizer object specifying how molecular dynamics files should be
        read (e.g. intramolecular distances, angles, dihedrals, etc). This
        parameter only makes sense if the input comes in the form of molecular
        dynamics trajectories or data, and will otherwise create a warning and
        have no effect.

    top : str, mdtraj.Trajectory or mdtraj.Topology, optional, default = None
        A topology file name. This is needed when molecular dynamics
        trajectories are given and no featurizer is given.
        In this case, only the Cartesian coordinates will be read. You can also pass an already
        loaded mdtraj.Topology object. If it is an mdtraj.Trajectory object, the topology
        will be extracted from it.

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    Returns
    -------
    reader : :class:`DataSource <pyemma.coordinates.data._base.datasource.DataSource>` object

    See also
    --------
    :func:`pyemma.coordinates.load`
        If your memory is big enough to load all features into memory, don't
        bother using source - working in memory is faster!

    :func:`pyemma.coordinates.pipeline`
        The data input is the first stage for your pipeline. Add other stages
        to it and build a pipeline to analyze big data in streaming mode.

    Examples
    --------

    Create a reader for NumPy files:

    >>> import numpy as np
    >>> from pyemma.coordinates import source
    >>> reader = source(['001.npy', '002.npy'] # doctest: +SKIP

    Create a reader for trajectory files and select some distance as feature:

    >>> reader = source(['traj01.xtc', 'traj02.xtc'], top='my_structure.pdb') # doctest: +SKIP
    >>> reader.featurizer.add_distances([[0, 1], [5, 6]]) # doctest: +SKIP
    >>> calculated_features = reader.get_output() # doctest: +SKIP

    create a reader for a csv file:

    >>> reader = source('data.csv') # doctest: +SKIP

    Create a reader for huge NumPy in-memory arrays to process them in
    huge chunks to avoid memory issues:

    >>> data = np.random.random(int(1e6))
    >>> reader = source(data, chunksize=1000)
    >>> from pyemma.coordinates import cluster_regspace
    >>> regspace = cluster_regspace(reader, dmin=0.1)

    Returns
    -------

    reader : a reader instance

    .. autoclass:: pyemma.coordinates.data.interface.ReaderInterface
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.data.interface.ReaderInterface
            :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.data.interface.ReaderInterface
            :attributes:

    """
    from pyemma.coordinates.data._base.iterable import Iterable
    from pyemma.coordinates.data.util.reader_utils import create_file_reader

    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(source)['chunksize'], **kw)

    # CASE 1: input is a string or list of strings
    # check: if single string create a one-element list
    if isinstance(inp, _string_types) or (
            isinstance(inp, (list, tuple))
            and (any(isinstance(item, (list, tuple, _string_types)) for item in inp) or len(inp) == 0)):
        reader = create_file_reader(inp, top, features, chunksize=cs, **kw)

    elif isinstance(inp, _np.ndarray) or (isinstance(inp, (list, tuple))
                                          and (any(isinstance(item, _np.ndarray) for item in inp) or len(inp) == 0)):
        # CASE 2: input is a (T, N, 3) array or list of (T_i, N, 3) arrays
        # check: if single array, create a one-element list
        # check: do all arrays have compatible dimensions (*, N, 3)? If not: raise ValueError.
        # check: if single array, create a one-element list
        # check: do all arrays have compatible dimensions (*, N)? If not: raise ValueError.
        # create MemoryReader
        from pyemma.coordinates.data.data_in_memory import DataInMemory as _DataInMemory
        reader = _DataInMemory(inp, chunksize=cs, **kw)
    elif isinstance(inp, Iterable):
        inp.chunksize = cs
        return inp
    else:
        raise ValueError('unsupported type (%s) of input' % type(inp))

    return reader


def combine_sources(sources, chunksize=None):
    r""" Combines multiple data sources to stream from.

    The given source objects (readers and transformers, eg. TICA) are concatenated in dimension axis during iteration.
    This can be used to couple arbitrary features in order to pass them to an Estimator expecting only one source,
    which is usually the case. All the parameters for iterator creation are passed to the actual sources, to ensure
    consistent behaviour.

    Parameters
    ----------
    sources : list, tuple
        list of DataSources (Readers, StreamingTransformers etc.) to combine for streaming access.

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    Notes
    -----
    This is currently only implemented for matching lengths trajectories.

    Returns
    -------
    merger : :class:`SourcesMerger <pyemma.coordinates.data.sources_merger.SourcesMerger>`

    """
    from pyemma.coordinates.data.sources_merger import SourcesMerger
    return SourcesMerger(sources, chunk=chunksize)


def pipeline(stages, run=True, stride=1, chunksize=None):
    r""" Data analysis pipeline.

    Constructs a data analysis :class:`Pipeline <pyemma.coordinates.pipelines.Pipeline>` and parametrizes it
    (unless prevented).
    If this function takes too long, consider loading data in memory.
    Alternatively if the data is to large to be loaded into memory make use
    of the stride parameter.

    Parameters
    ----------
    stages : data input or list of pipeline stages
        If given a single pipeline stage this must be a data input constructed
        by :py:func:`source`. If a list of pipelining stages are given, the
        first stage must be a data input constructed by :py:func:`source`.
    run : bool, optional, default = True
        If True, the pipeline will be parametrized immediately with the given
        stages. If only an input stage is given, the run flag has no effect at
        this time. True also means that the pipeline will be immediately
        re-parametrized when further stages are added to it.
        *Attention* True means this function may take a long time to compute.
        If False, the pipeline will be passive, i.e. it will not do any
        computations before you call parametrize()
    stride : int, optional, default = 1
        If set to 1, all input data will be used throughout the pipeline to
        parametrize its stages. Note that this could cause the parametrization
        step to be very slow for large data sets. Since molecular dynamics data
        is usually correlated at short timescales, it is often sufficient to
        parametrize the pipeline at a longer stride.
        See also stride option in the output functions of the pipeline.
    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    Returns
    -------
    pipe : :class:`Pipeline <pyemma.coordinates.pipelines.Pipeline>`
        A pipeline object that is able to conduct big data analysis with
        limited memory in streaming mode.

    Examples
    --------
    >>> import numpy as np
    >>> from pyemma.coordinates import source, tica, assign_to_centers, pipeline

    Create some random data and cluster centers:

    >>> data = np.random.random((1000, 3))
    >>> centers = data[np.random.choice(1000, 10)]
    >>> reader = source(data)

    Define a TICA transformation with lag time 10:

    >>> tica_obj = tica(lag=10)

    Assign any input to given centers:

    >>> assign = assign_to_centers(centers=centers)
    >>> pipe = pipeline([reader, tica_obj, assign])
    >>> pipe.parametrize()

    .. autoclass:: pyemma.coordinates.pipelines.Pipeline
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.pipelines.Pipeline
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.pipelines.Pipeline
            :attributes:

    """
    from pyemma.coordinates.pipelines import Pipeline

    if not isinstance(stages, list):
        stages = [stages]
    p = Pipeline(stages, param_stride=stride, chunksize=chunksize)
    if run:
        p.parametrize()
    return p


def discretizer(reader,
                transform=None,
                cluster=None,
                run=True,
                stride=1,
                chunksize=None):
    r""" Specialized pipeline: From trajectories to clustering.

    Constructs a pipeline that consists of three stages:

       1. an input stage (mandatory)
       2. a transformer stage (optional)
       3. a clustering stage (mandatory)

    This function is identical to calling :func:`pipeline` with the three
    stages, it is only meant as a guidance for the (probably) most common
    usage cases of a pipeline.

    Parameters
    ----------

    reader : instance of :class:`pyemma.coordinates.data.reader.ChunkedReader`
        The reader instance provides access to the data. If you are working
        with MD data, you most likely want to use a FeatureReader.

    transform : instance of :class: `pyemma.coordinates.Transformer`
        an optional transform like PCA/TICA etc.

    cluster : instance of :class: `pyemma.coordinates.AbstractClustering`
        clustering Transformer (optional) a cluster algorithm to assign
        transformed data to discrete states.

    stride : int, optional, default = 1
        If set to 1, all input data will be used throughout the pipeline
        to parametrize its stages. Note that this could cause the
        parametrization step to be very slow for large data sets. Since
        molecular dynamics data is usually correlated at short timescales,
        it is often sufficient to parametrize the pipeline at a longer stride.
        See also stride option in the output functions of the pipeline.

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    Returns
    -------
    pipe : a :class:`Pipeline <pyemma.coordinates.pipelines.Discretizer>` object
        A pipeline object that is able to streamline data analysis of large
        amounts of input data with limited memory in streaming mode.

    Examples
    --------

    Construct a discretizer pipeline processing all data
    with a PCA transformation and cluster the principal components
    with uniform time clustering:

    >>> from pyemma.coordinates import source, pca, cluster_regspace, discretizer
    >>> from pyemma.datasets import get_bpti_test_data
    >>> from pyemma.util.contexts import settings
    >>> reader = source(get_bpti_test_data()['trajs'], top=get_bpti_test_data()['top'])
    >>> transform = pca(dim=2)
    >>> cluster = cluster_regspace(dmin=0.1)

    Create the discretizer, access the the discrete trajectories and save them to files:

    >>> with settings(show_progress_bars=False):
    ...     disc = discretizer(reader, transform, cluster)
    ...     disc.dtrajs # doctest: +ELLIPSIS
    [array([...

    This will store the discrete trajectory to "traj01.dtraj":

    >>> from pyemma.util.files import TemporaryDirectory
    >>> import os
    >>> with TemporaryDirectory('dtrajs') as tmpdir:
    ...     disc.save_dtrajs(output_dir=tmpdir)
    ...     sorted(os.listdir(tmpdir))
    ['bpti_001-033.dtraj', 'bpti_034-066.dtraj', 'bpti_067-100.dtraj']

    .. autoclass:: pyemma.coordinates.pipelines.Pipeline
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.pipelines.Pipeline
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.pipelines.Pipeline
            :attributes:

    """
    from pyemma.coordinates.clustering.kmeans import KmeansClustering
    from pyemma.coordinates.pipelines import Discretizer
    if cluster is None:
        _logger.warning('You did not specify a cluster algorithm.'
                        ' Defaulting to kmeans(k=100)')
        cluster = KmeansClustering(n_clusters=100)
    disc = Discretizer(reader, transform, cluster, param_stride=stride, chunksize=chunksize)
    if run:
        disc.parametrize()
    return disc


def save_traj(traj_inp, indexes, outfile, top=None, stride=1, chunksize=None, image_molecules=False, verbose=True):
    r""" Saves a sequence of frames as a single trajectory.

    Extracts the specified sequence of time/trajectory indexes from traj_inp
    and saves it to one single molecular dynamics trajectory file. The output
    format will be determined by the outfile name.

    Parameters
    ----------

    traj_inp :
        traj_inp can be of two types.

            1. a python list of strings containing the filenames associated with
            the indices in :py:obj:`indexes`. With this type of input, a :py:obj:`topfile` is mandatory.

            2. a :py:func:`pyemma.coordinates.data.feature_reader.FeatureReader`
            object containing the filename list in :py:obj:`traj_inp.trajfiles`.
            Please use :py:func:`pyemma.coordinates.source` to construct it.
            With this type of input, the input :py:obj:`topfile` will be ignored.
            and :py:obj:`traj_inp.topfile` will be used instead

    indexes : ndarray(T, 2) or list of ndarray(T_i, 2)
        A (T x 2) array for writing a trajectory of T time steps. Each row
        contains two indexes (i, t), where i is the index of the trajectory
        from the input and t is the index of the time step within the trajectory.
        If a list of index arrays is given, these will be simply concatenated,
        i.e. they will be written subsequently in the same trajectory file.

    outfile : str.
        The name of the output file. Its extension will determine the file type
        written. Example: "out.dcd" If set to None, the trajectory object is
        returned to memory

    top : str, mdtraj.Trajectory, or mdtraj.Topology
        The topology needed to read the files in the list :py:obj:`traj_inp`.
        If :py:obj:`traj_inp` is not a list, this parameter is ignored.

    stride  : integer, default is 1
        This parameter informs :py:func:`save_traj` about the stride used in
        :py:obj:`indexes`. Typically, :py:obj:`indexes` contains frame-indexes
        that match exactly the frames of the files contained in :py:obj:`traj_inp.trajfiles`.
        However, in certain situations, that might not be the case. Examples
        are cases in which a stride value != 1 was used when
        reading/featurizing/transforming/discretizing the files contained
        in :py:obj:`traj_inp.trajfiles`.

    chunksize : int. Default=None.
        The chunksize for reading input trajectory files. If :py:obj:`traj_inp`
        is a :py:func:`pyemma.coordinates.data.feature_reader.FeatureReader` object,
        this input variable will be ignored and :py:obj:`traj_inp.chunksize` will be used instead.

    image_molecules: boolean, default is False
        If set to true, :py:obj:`save_traj` will call the method traj.image_molecules and try to correct for broken
        molecules accross periodic boundary conditions.
        (http://mdtraj.org/1.7.2/api/generated/mdtraj.Trajectory.html#mdtraj.Trajectory.image_molecules)

    verbose : boolean, default is True
        Inform about created filenames

    Returns
    -------
    traj : :py:obj:`mdtraj.Trajectory` object
        Will only return this object if :py:obj:`outfile` is None
    """
    from mdtraj import Topology, Trajectory

    from pyemma.coordinates.data.feature_reader import FeatureReader
    from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader
    from pyemma.coordinates.data.util.frames_from_file import frames_from_files
    from pyemma.coordinates.data.util.reader_utils import enforce_top
    import itertools

    # Determine the type of input and extract necessary parameters
    if isinstance(traj_inp, (FeatureReader, FragmentedTrajectoryReader)):
        if isinstance(traj_inp, FragmentedTrajectoryReader):
            # lengths array per reader
            if not all(isinstance(reader, FeatureReader)
                                     for reader in itertools.chain.from_iterable(traj_inp._readers)):
                raise ValueError("Only FeatureReaders (MD-data) are supported for fragmented trajectories.")
            trajfiles = traj_inp.filenames_flat
            top = traj_inp._readers[0][0].featurizer.topology
        else:
            top = traj_inp.featurizer.topology
            trajfiles = traj_inp.filenames
        chunksize = traj_inp.chunksize
        reader = traj_inp
    else:
        # Do we have what we need?
        if not isinstance(traj_inp, (list, tuple)):
            raise TypeError("traj_inp has to be of type list, not %s" % type(traj_inp))
        if not isinstance(top, (_string_types, Topology, Trajectory)):
            raise TypeError("traj_inp cannot be a list of files without an input "
                            "top of type str (eg filename.pdb), mdtraj.Trajectory or mdtraj.Topology. "
                            "Got type %s instead" % type(top))
        trajfiles = traj_inp
        reader = None

    # Enforce the input topology to actually be an md.Topology object
    top = enforce_top(top)

    # Convert to index (T,2) array if parsed a list or a list of arrays
    indexes = _np.vstack(indexes)

    # Check that we've been given enough filenames
    if len(trajfiles) < indexes[:, 0].max():
        raise ValueError("traj_inp contains %u trajfiles, "
                         "but indexes will ask for file nr. %u"
                         % (len(trajfiles), indexes[:,0].max()))

    traj = frames_from_files(trajfiles, top, indexes, chunksize, stride, reader=reader)

    # Avoid broken molecules
    if image_molecules:
        traj.image_molecules(inplace=True)

    # Return to memory as an mdtraj trajectory object
    if outfile is None:
        return traj
    # or to disk as a molecular trajectory file
    else:
        if isinstance(outfile, Path):
            outfile = str(outfile.resolve())
        traj.save(outfile)
    if verbose:
        _logger.info("Created file %s" % outfile)


def save_trajs(traj_inp, indexes, prefix='set_', fmt=None, outfiles=None,
               inmemory=False, stride=1, verbose=False):
    r""" Saves sequences of frames as multiple trajectories.

    Extracts a number of specified sequences of time/trajectory indexes from the
    input loader and saves them in a set of molecular dynamics trajectories.
    The output filenames are obtained by prefix + str(n) + .fmt, where n counts
    the output trajectory and extension is either set by the user, or else
    determined from the input. Example: When the input is in dcd format, and
    indexes is a list of length 3, the output will by default go to files
    "set_1.dcd", "set_2.dcd", "set_3.dcd". If you want files to be stored
    in a specific subfolder, simply specify the relative path in the prefix,
    e.g. prefix='~/macrostates/\pcca_'

    Parameters
    ----------
    traj_inp : :py:class:`pyemma.coordinates.data.feature_reader.FeatureReader`
        A data source as provided by Please use :py:func:`pyemma.coordinates.source` to construct it.

    indexes : list of ndarray(T_i, 2)
        A list of N arrays, each of size (T_n x 2) for writing N trajectories
        of T_i time steps. Each row contains two indexes (i, t), where i is the
        index of the trajectory from the input and t is the index of the time
        step within the trajectory.

    prefix : str, optional, default = `set_`
        output filename prefix. Can include an absolute or relative path name.

    fmt : str, optional, default = None
        Outpuf file format. By default, the file extension and format. It will
        be determined from the input. If a different format is desired, specify
        the corresponding file extension here without a dot, e.g. "dcd" or "xtc".

    outfiles : list of str, optional, default = None
        A list of output filenames. When given, this will override the settings
        of prefix and fmt, and output will be written to these files.

    inmemory : Boolean, default = False (untested for large files)
        Instead of internally calling traj_save for every (T_i,2) array in
        "indexes", only one call is made. Internally, this generates a
        potentially large molecular trajectory object in memory that is
        subsequently sliced into the files of "outfiles". Should be faster for
        large "indexes" arrays and  large files, though it is quite memory
        intensive. The optimal situation is to avoid streaming two times
        through a huge file for "indexes" of type: indexes = [[1 4000000],[1 4000001]]

    stride  : integer, default is 1
        This parameter informs :py:func:`save_trajs` about the stride used in
        the indexes variable. Typically, the variable indexes contains frame
        indexes that match exactly the frames of the files contained in
        traj_inp.trajfiles. However, in certain situations, that might not be
        the case. Examples of these situations are cases in which stride
        value != 1 was used when reading/featurizing/transforming/discretizing
        the files contained in traj_inp.trajfiles.

    verbose : boolean, default is False
        Verbose output while looking for "indexes" in the "traj_inp.trajfiles"

    Returns
    -------
    outfiles : list of str
        The list of absolute paths that the output files have been written to.

    """
    # Make sure indexes is iterable
    assert _types.is_iterable(indexes), "Indexes must be an iterable of matrices."
    # only if 2d-array, convert into a list
    if isinstance(indexes, _np.ndarray):
        if indexes.ndim == 2:
            indexes = [indexes]

    # Make sure the elements of that lists are arrays, and that they are shaped properly
    for i_indexes in indexes:
        assert isinstance(i_indexes, _np.ndarray), "The elements in the 'indexes' variable must be numpy.ndarrays"
        assert i_indexes.ndim == 2, \
            "The elements in the 'indexes' variable must have ndim = 2, and not %u" % i_indexes.ndim
        assert i_indexes.shape[1] == 2, \
            "The elements in the 'indexes' variable must be of shape (T_i,2), and not (%u,%u)" % i_indexes.shape

    # Determine output format of the molecular trajectory file
    if fmt is None:
        fname = traj_inp.filenames[0]
        while hasattr(fname, '__getitem__') and not isinstance(fname, (str, bytes)):
            fname = fname[0]
        import os

        _, fmt = os.path.splitext(fname)
    else:
        fmt = '.' + fmt

    # Prepare the list of outfiles before the loop
    if outfiles is None:
        outfiles = []
        for ii in range(len(indexes)):
            outfiles.append(prefix + '%06u' % ii + fmt)

    # Check that we have the same name of outfiles as (T, 2)-indexes arrays
    if len(indexes) != len(outfiles):
        raise Exception('len(indexes) (%s) does not match len(outfiles) (%s)' % (len(indexes), len(outfiles)))

    # This implementation looks for "i_indexes" separately, and thus one traj_inp.trajfile
    # might be accessed more than once (less memory intensive)
    if not inmemory:
        for i_indexes, outfile in zip(indexes, outfiles):
            # TODO: use **kwargs to parse to save_traj
            save_traj(traj_inp, i_indexes, outfile, stride=stride, verbose=verbose)

    # This implementation is "one file - one pass" but might temporally create huge memory objects
    else:
        traj = save_traj(traj_inp, indexes, outfile=None, stride=stride, verbose=verbose)
        i_idx = 0
        for i_indexes, outfile in zip(indexes, outfiles):
            # Create indices for slicing the mdtraj trajectory object
            f_idx = i_idx + len(i_indexes)
            # print i_idx, f_idx
            traj[i_idx:f_idx].save(outfile)
            _logger.info("Created file %s" % outfile)
            # update the initial frame index
            i_idx = f_idx

    return outfiles


# =========================================================================
#
# TRANSFORMATION ALGORITHMS
#
# =========================================================================

def pca(data=None, dim=-1, var_cutoff=0.95, stride=1, mean=None, skip=0, chunksize=None, **kwargs):
    r""" Principal Component Analysis (PCA).

    PCA is a linear transformation method that finds coordinates of maximal
    variance. A linear projection onto the principal components thus makes a
    minimal error in terms of variation in the data. Note, however, that this
    method is not optimal for Markov model construction because for that
    purpose the main objective is to preserve the slow processes which can
    sometimes be associated with small variance.

    It estimates a PCA transformation from data. When input data is given as an
    argument, the estimation will be carried out right away, and the resulting
    object can be used to obtain eigenvalues, eigenvectors or project input data
    onto the principal components. If data is not given, this object is an
    empty estimator and can be put into a :func:`pipeline` in order to use PCA
    in streaming mode.

    Parameters
    ----------

    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by
        source function data array or list of data arrays. T or T_i are the
        number of time steps in a trajectory. When data is given, the PCA is
        immediately parametrized by estimating the covariance matrix and
        computing its eigenvectors.

    dim : int, optional, default -1
        the number of dimensions (principal components) to project onto. A
        call to the :func:`map <pyemma.coordinates.transform.PCA.map>` function reduces the d-dimensional
        input to only dim dimensions such that the data preserves the
        maximum possible variance amongst dim-dimensional linear projections.
        -1 means all numerically available dimensions will be used unless
        reduced by var_cutoff. Setting dim to a positive value is exclusive
        with var_cutoff.

    var_cutoff : float in the range [0,1], optional, default 0.95
        Determines the number of output dimensions by including dimensions
        until their cumulative kinetic variance exceeds the fraction
        subspace_variance. var_cutoff=1.0 means all numerically available
        dimensions (see epsilon) will be used, unless set by dim. Setting
        var_cutoff smaller than 1.0 is exclusive with dim

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that
        this could cause this calculation to be very slow for large data
        sets. Since molecular dynamics data is usually correlated at short
        timescales, it is often sufficient to estimate transformations at
        a longer stride. Note that the stride option in the get_output()
        function of the returned object is independent, so you can parametrize
        at a long stride, and still map all frames through the transformer.

    mean : ndarray, optional, default None
        Optionally pass pre-calculated means to avoid their re-computation.
        The shape has to match the input dimension.

    skip : int, default=0
        skip the first initial n frames per trajectory.

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    Returns
    -------
    pca : a :class:`PCA<pyemma.coordinates.transform.PCA>` transformation object
        Object for Principle component analysis (PCA) analysis.
        It contains PCA eigenvalues and eigenvectors, and the projection of
        input data to the dominant PCA


    Notes
    -----
    Given a sequence of multivariate data :math:`X_t`,
    computes the mean-free covariance matrix.

    .. math:: C = (X - \mu)^T (X - \mu)

    and solves the eigenvalue problem

    .. math:: C r_i = \sigma_i r_i,

    where :math:`r_i` are the principal components and :math:`\sigma_i` are
    their respective variances.

    When used as a dimension reduction method, the input data is projected onto
    the dominant principal components.

    See `Wiki page <http://en.wikipedia.org/wiki/Principal_component_analysis>`_ for more theory and references.
    for more theory and references.

    See also
    --------
    :class:`PCA <pyemma.coordinates.transform.PCA>` : pca object

    :func:`tica <pyemma.coordinates.tica>` : for time-lagged independent component analysis


    .. autoclass:: pyemma.coordinates.transform.pca.PCA
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.transform.pca.PCA
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.transform.pca.PCA
            :attributes:

    References
    ----------
    .. [1] Pearson, K. 1901
        On Lines and Planes of Closest Fit to Systems of Points in Space
        Phil. Mag. 2, 559--572

    .. [2] Hotelling, H. 1933.
        Analysis of a complex of statistical variables into principal components.
        J. Edu. Psych. 24, 417-441 and 498-520.

    """
    from pyemma.coordinates.transform.pca import PCA

    if mean is not None:
        import warnings
        warnings.warn("provided mean ignored", DeprecationWarning)

    res = PCA(dim=dim, var_cutoff=var_cutoff, mean=None, skip=skip, stride=stride)
    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(pca)['chunksize'], **kwargs)
    if data is not None:
        res.estimate(data, chunksize=cs)
    else:
        res.chunksize = cs
    return res


def tica(data=None, lag=10, dim=-1, var_cutoff=0.95, kinetic_map=True, commute_map=False, weights='empirical',
         stride=1, remove_mean=True, skip=0, reversible=True, ncov_max=float('inf'), chunksize=None, **kwargs):
    r""" Time-lagged independent component analysis (TICA).

    TICA is a linear transformation method. In contrast to PCA, which finds
    coordinates of maximal variance, TICA finds coordinates of maximal
    autocorrelation at the given lag time. Therefore, TICA is useful in order
    to find the *slow* components in a dataset and thus an excellent choice to
    transform molecular dynamics data before clustering data for the
    construction of a Markov model. When the input data is the result of a
    Markov process (such as thermostatted molecular dynamics), TICA finds in
    fact an approximation to the eigenfunctions and eigenvalues of the
    underlying Markov operator [1]_.

    It estimates a TICA transformation from *data*. When input data is given as
    an argument, the estimation will be carried out straight away, and the
    resulting object can be used to obtain eigenvalues, eigenvectors or project
    input data onto the slowest TICA components. If no data is given, this
    object is an empty estimator and can be put into a :func:`pipeline` in
    order to use TICA in the streaming mode.

    Parameters
    ----------
    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by
        source function array with the data, if available. When given, the TICA
        transformation is immediately computed and can be used to transform data.

    lag : int, optional, default = 10
        the lag time, in multiples of the input time step

    dim : int, optional, default -1
        the number of dimensions (independent components) to project onto. A
        call to the :func:`map <pyemma.coordinates.transform.TICA.map>` function
        reduces the d-dimensional input to only dim dimensions such that the
        data preserves the maximum possible autocorrelation amongst
        dim-dimensional linear projections. -1 means all numerically available
        dimensions will be used unless reduced by var_cutoff.
        Setting dim to a positive value is exclusive with var_cutoff.

    var_cutoff : float in the range [0,1], optional, default 0.95
        Determines the number of output dimensions by including dimensions
        until their cumulative kinetic variance exceeds the fraction
        subspace_variance. var_cutoff=1.0 means all numerically available
        dimensions (see epsilon) will be used, unless set by dim. Setting
        var_cutoff smaller than 1.0 is exclusive with dim

    kinetic_map : bool, optional, default True
        Eigenvectors will be scaled by eigenvalues. As a result, Euclidean
        distances in the transformed data approximate kinetic distances [4]_.
        This is a good choice when the data is further processed by clustering.

    commute_map : bool, optional, default False
        Eigenvector_i will be scaled by sqrt(timescale_i / 2). As a result, Euclidean distances in the transformed
        data will approximate commute distances [5]_.

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this
        could cause this calculation to be very slow for large data sets. Since
        molecular dynamics data is usually correlated at short timescales, it is
        often sufficient to estimate transformations at a longer stride. Note
        that the stride option in the get_output() function of the returned
        object is independent, so you can parametrize at a long stride, and
        still map all frames through the transformer.

    weights : optional, default="empirical"
             Re-weighting strategy to be used in order to compute equilibrium covariances from non-equilibrium data.
                * "empirical":  no re-weighting
                * "koopman":    use re-weighting procedure from [6]_
                * weights:      An object that allows to compute re-weighting factors. It must possess a method
                                weights(X) that accepts a trajectory X (np.ndarray(T, n)) and returns a vector of
                                re-weighting factors (np.ndarray(T,)).

    remove_mean: bool, optional, default True
        remove mean during covariance estimation. Should not be turned off.

    skip : int, default=0
        skip the first initial n frames per trajectory.

    reversible: bool, default=True
            symmetrize correlation matrices C_0, C_{\tau}.

    ncov_max : int, default=infinity
        limit the memory usage of the algorithm from [7]_ to an amount that corresponds
        to ncov_max additional copies of each correlation matrix

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    Returns
    -------
    tica : a :class:`TICA <pyemma.coordinates.transform.TICA>` transformation object
        Object for time-lagged independent component (TICA) analysis.
        it contains TICA eigenvalues and eigenvectors, and the projection of
        input data to the dominant TICA


    Notes
    -----
    Given a sequence of multivariate data :math:`X_t`, it computes the
    mean-free covariance and time-lagged covariance matrix:

    .. math::

        C_0 &=      (X_t - \mu)^T \mathrm{diag}(w) (X_t - \mu) \\
        C_{\tau} &= (X_t - \mu)^T \mathrm{diag}(w) (X_t + \tau - \mu)

    where w is a vector of weights for each time step. By default, these weights
    are all equal to one, but different weights are possible, like the re-weighting
    to equilibrium described in [6]_. Subsequently, the eigenvalue problem

    .. math:: C_{\tau} r_i = C_0 \lambda_i r_i,

    is solved,where :math:`r_i` are the independent components and :math:`\lambda_i` are
    their respective normalized time-autocorrelations. The eigenvalues are
    related to the relaxation timescale by

    .. math::

        t_i = -\frac{\tau}{\ln |\lambda_i|}.

    When used as a dimension reduction method, the input data is projected
    onto the dominant independent components.

    TICA was originally introduced for signal processing in [2]_. It was
    introduced to molecular dynamics and as a method for the construction
    of Markov models in [1]_ and [3]_. It was shown in [1]_ that when applied
    to molecular dynamics data, TICA is an approximation to the eigenvalues
    and eigenvectors of the true underlying dynamics.

    Examples
    --------
    Invoke TICA transformation with a given lag time and output dimension:

    >>> import numpy as np
    >>> from pyemma.coordinates import tica
    >>> data = np.random.random((100,3))
    >>> projected_data = tica(data, lag=2, dim=1).get_output()[0]

    For a brief explaination why TICA outperforms PCA to extract a good reaction
    coordinate have a look `here
    <http://docs.markovmodel.org/lecture_tica.html#Example:-TICA-versus-PCA-in-a-stretched-double-well-potential>`_.

    See also
    --------
    :class:`TICA <pyemma.coordinates.transform.TICA>` : tica object

    :func:`pca <pyemma.coordinates.pca>` : for principal component analysis


    .. autoclass:: pyemma.coordinates.transform.tica.TICA
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.transform.tica.TICA
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.transform.tica.TICA
            :attributes:

    References
    ----------

    .. [1] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
       Identification of slow molecular order parameters for Markov model construction
       J. Chem. Phys. 139, 015102. doi:10.1063/1.4811489

    .. [2] L. Molgedey and H. G. Schuster. 1994.
       Separation of a mixture of independent signals using time delayed correlations
       Phys. Rev. Lett. 72, 3634.

    .. [3] Schwantes C, V S Pande. 2013.
       Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9
       J. Chem. Theory. Comput. 9, 2000-2009. doi:10.1021/ct300878a

    .. [4] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
        J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553

    .. [5] Noe, F., Banisch, R., Clementi, C. 2016. Commute maps: separating slowly-mixing molecular configurations
       for kinetic modeling. J. Chem. Theory. Comput. doi:10.1021/acs.jctc.6b00762

    .. [6] Wu, H., Nueske, F., Paul, F., Klus, S., Koltai, P., and Noe, F. 2016. Bias reduced variational
        approximation of molecular kinetics from short off-equilibrium simulations. J. Chem. Phys. (submitted),
        https://arxiv.org/abs/1610.06773.

    .. [7] Chan, T. F., Golub G. H., LeVeque R. J. 1979. Updating formulae and pairwiese algorithms for
        computing sample variances. Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.

    """
    from pyemma.coordinates.transform.tica import TICA
    from pyemma.coordinates.estimation.koopman import _KoopmanEstimator
    import types
    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(tica)['chunksize'], **kwargs)

    if isinstance(weights, _string_types):
        if weights == "koopman":
            if data is None:
                raise ValueError("Data must be supplied for reweighting='koopman'")
            if not reversible:
                raise ValueError("Koopman re-weighting is designed for reversible processes, set reversible=True")
            koop = _KoopmanEstimator(lag=lag, stride=stride, skip=skip, ncov_max=ncov_max)
            koop.estimate(data, chunksize=cs)
            weights = koop.weights
        elif weights == "empirical":
            weights = None
        else:
            raise ValueError("reweighting must be either 'empirical', 'koopman' "
                             "or an object with a weights(data) method.")
    elif hasattr(weights, 'weights') and type(getattr(weights, 'weights')) == types.MethodType:
        weights = weights
    elif isinstance(weights, (list, tuple)) and all(isinstance(w, _np.ndarray) for w in weights):
        if data is not None and len(data) != len(weights):
            raise ValueError("len of weights({}) must match len of data({}).".format(len(weights), len(data)))
    else:
        raise ValueError("reweighting must be either 'empirical', 'koopman' or an object with a weights(data) method.")

    if not remove_mean:
        import warnings
        user_msg = 'remove_mean option is deprecated. The mean is removed from the data by default, otherwise it' \
                   'cannot be guaranteed that all eigenvalues will be smaller than one. Some functionalities might' \
                   'become useless in this case (e.g. commute_maps). Also, not removing the mean will not result in' \
                   'a significant speed up of calculations.'
        warnings.warn(
            user_msg,
            category=_PyEMMA_DeprecationWarning)

    res = TICA(lag, dim=dim, var_cutoff=var_cutoff, kinetic_map=kinetic_map, commute_map=commute_map, skip=skip, stride=stride,
               weights=weights, reversible=reversible, ncov_max=ncov_max)
    if data is not None:
        res.estimate(data, chunksize=cs)
    else:
        res.chunksize = cs
    return res


def vamp(data=None, lag=10, dim=None, scaling=None, right=False, ncov_max=float('inf'),
         stride=1, skip=0, chunksize=None):
    r""" Variational approach for Markov processes (VAMP) [1]_.

      Parameters
      ----------
      lag : int
          lag time
      dim : float or int, default=None
          Number of dimensions to keep:

          * if dim is not set (None) all available ranks are kept:
              `n_components == min(n_samples, n_uncorrelated_features)`
          * if dim is an integer >= 1, this number specifies the number
            of dimensions to keep.
          * if dim is a float with ``0 < dim < 1``, select the number
            of dimensions such that the amount of kinetic variance
            that needs to be explained is greater than the percentage
            specified by dim.
      scaling : None or string
          Scaling to be applied to the VAMP order parameters upon transformation

          * None: no scaling will be applied, variance of the order parameters is 1
          * 'kinetic map' or 'km': order parameters are scaled by singular value.
            Only the left singular functions induce a kinetic map wrt the
            conventional forward propagator. The right singular functions induce
            a kinetic map wrt the backward propagator.      right : boolean
          Whether to compute the right singular functions.
          If `right==True`, `get_output()` will return the right singular
          functions. Otherwise, `get_output()` will return the left singular
          functions.
          Beware that only `frames[tau:, :]` of each trajectory returned
          by `get_output()` contain valid values of the right singular
          functions. Conversely, only `frames[0:-tau, :]` of each
          trajectory returned by `get_output()` contain valid values of
          the left singular functions. The remaining frames might
          possibly be interpreted as some extrapolation.
      epsilon : float
          eigenvalue cutoff. Eigenvalues of :math:`C_{00}` and :math:`C_{11}`
          with norms <= epsilon will be cut off. The remaining number of
          eigenvalues together with the value of `dim` define the size of the output.
      stride: int, optional, default = 1
          Use only every stride-th time step. By default, every time step is used.
      skip : int, default=0
          skip the first initial n frames per trajectory.
      ncov_max : int, default=infinity
          limit the memory usage of the algorithm from [3]_ to an amount that corresponds
          to ncov_max additional copies of each correlation matrix

      Returns
      -------
      vamp : a :class:`VAMP <pyemma.coordinates.transform.VAMP>` transformation object
         It contains the definitions of singular functions and singular values and
         can be used to project input data to the dominant VAMP components, predict
         expectations and time-lagged covariances and perform a Chapman-Kolmogorov
         test.

      Notes
      -----
      VAMP is a method for dimensionality reduction of Markov processes.

      The Koopman operator :math:`\mathcal{K}` is an integral operator
      that describes conditional future expectation values. Let
      :math:`p(\mathbf{x},\,\mathbf{y})` be the conditional probability
      density of visiting an infinitesimal phase space volume around
      point :math:`\mathbf{y}` at time :math:`t+\tau` given that the phase
      space point :math:`\mathbf{x}` was visited at the earlier time
      :math:`t`. Then the action of the Koopman operator on a function
      :math:`f` can be written as follows:

      .. math::

          \mathcal{K}f=\int p(\mathbf{x},\,\mathbf{y})f(\mathbf{y})\,\mathrm{dy}=\mathbb{E}\left[f(\mathbf{x}_{t+\tau}\mid\mathbf{x}_{t}=\mathbf{x})\right]

      The Koopman operator is defined without any reference to an
      equilibrium distribution. Therefore it is well-defined in
      situations where the dynamics is irreversible or/and non-stationary
      such that no equilibrium distribution exists.

      If we approximate :math:`f` by a linear superposition of ansatz
      functions :math:`\boldsymbol{\chi}` of the conformational
      degrees of freedom (features), the operator :math:`\mathcal{K}`
      can be approximated by a (finite-dimensional) matrix :math:`\mathbf{K}`.

      The approximation is computed as follows: From the time-dependent
      input features :math:`\boldsymbol{\chi}(t)`, we compute the mean
      :math:`\boldsymbol{\mu}_{0}` (:math:`\boldsymbol{\mu}_{1}`) from
      all data excluding the last (first) :math:`\tau` steps of every
      trajectory as follows:

      .. math::

          \boldsymbol{\mu}_{0}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\boldsymbol{\chi}(t)

          \boldsymbol{\mu}_{1}	:=\frac{1}{T-\tau}\sum_{t=\tau}^{T}\boldsymbol{\chi}(t)

      Next, we compute the instantaneous covariance matrices
      :math:`\mathbf{C}_{00}` and :math:`\mathbf{C}_{11}` and the
      time-lagged covariance matrix :math:`\mathbf{C}_{01}` as follows:

      .. math::

          \mathbf{C}_{00}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]

          \mathbf{C}_{11}	:=\frac{1}{T-\tau}\sum_{t=\tau}^{T}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]

          \mathbf{C}_{01}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t+\tau)-\boldsymbol{\mu}_{1}\right]

      The Koopman matrix is then computed as follows:

      .. math::

          \mathbf{K}=\mathbf{C}_{00}^{-1}\mathbf{C}_{01}

      It can be shown [1]_ that the leading singular functions of the
      half-weighted Koopman matrix

      .. math::

          \bar{\mathbf{K}}:=\mathbf{C}_{00}^{-\frac{1}{2}}\mathbf{C}_{01}\mathbf{C}_{11}^{-\frac{1}{2}}

      encode the best reduced dynamical model for the time series.

      The singular functions can be computed by first performing the
      singular value decomposition

      .. math::

          \bar{\mathbf{K}}=\mathbf{U}^{\prime}\mathbf{S}\mathbf{V}^{\prime}

      and then mapping the input conformation to the left singular
      functions :math:`\boldsymbol{\psi}` and right singular
      functions :math:`\boldsymbol{\phi}` as follows:

      .. math::

          \boldsymbol{\psi}(t):=\mathbf{U}^{\prime\top}\mathbf{C}_{00}^{-\frac{1}{2}}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]

          \boldsymbol{\phi}(t):=\mathbf{V}^{\prime\top}\mathbf{C}_{11}^{-\frac{1}{2}}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]


      References
      ----------
      .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
          arXiv:1707.04659v1
      .. [2] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
          J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
      .. [3] Chan, T. F., Golub G. H., LeVeque R. J. 1979. Updating formulae and pairwiese algorithms for
         computing sample variances. Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
    """
    from pyemma.coordinates.transform.vamp import VAMP
    res = VAMP(lag, dim=dim, scaling=scaling, right=right, skip=skip, ncov_max=ncov_max)
    if data is not None:
        res.estimate(data, stride=stride, chunksize=chunksize)
    else:
        res.chunksize = chunksize
    return res


def tica_nystroem(max_columns, data=None, lag=10,
                  dim=-1, var_cutoff=0.95, epsilon=1e-6,
                  stride=1, skip=0, reversible=True, ncov_max=float('inf'), chunksize=None,
                  initial_columns=None, nsel=1, neig=None):
    r""" Sparse sampling implementation [1]_ of time-lagged independent component analysis (TICA) [2]_, [3]_, [4]_.

    Parameters
    ----------
    max_columns : int
        Maximum number of columns (features) to use in the approximation.
    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by
        source function array with the data. With it, the TICA
        transformation is immediately computed and can be used to transform data.
    lag : int, optional, default 10
        lag time
    dim : int, optional, default -1
        Maximum number of significant independent components to use to reduce dimension of input data. -1 means
        all numerically available dimensions (see epsilon) will be used unless reduced by var_cutoff.
        Setting dim to a positive value is exclusive with var_cutoff.
    var_cutoff : float in the range [0,1], optional, default 0.95
        Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
        exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
        (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim.
    epsilon : float, optional, default 1e-6
        Eigenvalue norm cutoff. Eigenvalues of :math:`C_0` with norms <= epsilon will be
        cut off. The remaining number of eigenvalues define the size
        of the output.
    stride: int, optional, default 1
        Use only every stride-th time step. By default, every time step is used.
    skip : int, optional, default 0
        Skip the first initial n frames per trajectory.
    reversible: bool, optional, default True
        Symmetrize correlation matrices :math:`C_0`, :math:`C_{\tau}`.
    initial_columns : list, ndarray(k, dtype=int), int, or None, optional, default None
        Columns used for an initial approximation. If a list or an 1-d ndarray
        of integers is given, use these column indices. If an integer is given,
        use that number of randomly selected indices. If None is given, use
        one randomly selected column.
    nsel : int, optional, default 1
        Number of columns to select and add per iteration and pass through the data.
        Larger values provide for better pass-efficiency.
    neig : int or None, optional, default None
        Number of eigenvalues to be optimized by the selection process.
        If None, use the whole available eigenspace.

    Returns
    -------
    tica_nystroem : a :class:`NystroemTICA <pyemma.coordinates.transform.NystroemTICA>`
                    transformation object
        Object for sparse sampling time-lagged independent component (TICA) analysis.
        It contains TICA eigenvalues and eigenvectors, and the projection of
        input data to the dominant TICs.

    Notes
    -----
    Perform a sparse approximation of time-lagged independent component analysis (TICA)
    :class:`TICA <pyemma.coordinates.transform.TICA>`. The starting point is the
    generalized eigenvalue problem

    .. math:: C_{\tau} r_i = C_0 \lambda_i(\tau) r_i.

    Instead of computing the full matrices involved in this problem, we conduct
    a Nystroemm approximation [5]_ of the matrix :math:`C_0` by means of the
    accelerated sequential incoherence selection (oASIS) algorithm [6]_ and,
    in particular, its extension called spectral oASIS [1]_.

    Iteratively, we select a small number of columns such that the resulting
    Nystroem approximation is sufficiently accurate. This selection represents
    in turn a subset of important features, for which we obtain a generalized
    eigenvalue problem similar to the one above, but much smaller in size.
    Its generalized eigenvalues and eigenvectors provide an approximation
    to those of the full TICA solution [1]_.

    References
    ----------
    .. [1] F. Litzinger, L. Boninsegna, H. Wu, F. Nueske, R. Patel, R. Baraniuk, F. Noe, and C. Clementi.
       Rapid calculation of molecular kinetics using compressed sensing (2018). (submitted)
    .. [2] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
       Identification of slow molecular order parameters for Markov model construction
       J. Chem. Phys. 139, 015102. doi:10.1063/1.4811489
    .. [3] Schwantes C, V S Pande. 2013.
       Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9
       J. Chem. Theory. Comput. 9, 2000-2009. doi:10.1021/ct300878a
    .. [4] L. Molgedey and H. G. Schuster. 1994.
       Separation of a mixture of independent signals using time delayed correlations
       Phys. Rev. Lett. 72, 3634.
    .. [5] P. Drineas and M. W. Mahoney.
       On the Nystrom method for approximating a Gram matrix for improved kernel-based learning.
       Journal of Machine Learning Research, 6:2153-2175 (2005).
    .. [6] Raajen Patel, Thomas A. Goldstein, Eva L. Dyer, Azalia Mirhoseini, Richard G. Baraniuk.
       oASIS: Adaptive Column Sampling for Kernel Matrix Approximation.
       arXiv: 1505.05208 [stat.ML].

    """
    from pyemma.coordinates.transform.nystroem_tica import NystroemTICA
    res = NystroemTICA(lag, max_columns,
                       dim=dim, var_cutoff=var_cutoff, epsilon=epsilon,
                       stride=stride, skip=skip, reversible=reversible,
                       ncov_max=ncov_max,
                       initial_columns=initial_columns, nsel=nsel, neig=neig)
    if data is not None:
        res.estimate(data, stride=stride, chunksize=chunksize)
    else:
        res.chunksize = chunksize
    return res


def covariance_lagged(data=None, c00=True, c0t=True, ctt=False, remove_constant_mean=None, remove_data_mean=False,
                      reversible=False, bessel=True, lag=0, weights="empirical", stride=1, skip=0, chunksize=None,
                      ncov_max=float('inf'), column_selection=None, diag_only=False):
    r"""Compute lagged covariances between time series. If data is available as an array of size (TxN), where T is the
    number of time steps and N the number of dimensions, this function can compute lagged covariances like

    .. math::

        C_00 &= X^T X \\
        C_{0t} &= X^T Y \\
        C_{tt} &= Y^T Y,

    where X comprises the first T-lag time steps and Y the last T-lag time steps. It is also possible to use more
    than one time series, the number of time steps in each time series can also vary.

    Parameters
    ----------
    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by
        source function array with the data, if available. When given, the covariances are immediately computed.
    c00 : bool, optional, default=True
        compute instantaneous correlations over the first part of the data. If lag==0, use all of the data.
    c0t : bool, optional, default=False
        compute lagged correlations. Does not work with lag==0.
    ctt : bool, optional, default=False
        compute instantaneous correlations over the second part of the data. Does not work with lag==0.
    remove_constant_mean : ndarray(N,), optional, default=None
        substract a constant vector of mean values from time series.
    remove_data_mean : bool, optional, default=False
        substract the sample mean from the time series (mean-free correlations).
    reversible : bool, optional, default=False
        symmetrize correlations.
    bessel : bool, optional, default=True
        use Bessel's correction for correlations in order to use an unbiased estimator
    lag : int, optional, default=0
        lag time. Does not work with xy=True or yy=True.
    weights : optional, default="empirical"
         Re-weighting strategy to be used in order to compute equilibrium covariances from non-equilibrium data.
            * "empirical":  no re-weighting
            * "koopman":    use re-weighting procedure from [1]_
            * weights:      An object that allows to compute re-weighting factors. It must possess a method
                            weights(X) that accepts a trajectory X (np.ndarray(T, n)) and returns a vector of
                            re-weighting factors (np.ndarray(T,)).
    stride: int, optional, default = 1
        Use only every stride-th time step. By default, every time step is used.
    skip : int, optional, default=0
        skip the first initial n frames per trajectory.
    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.
    ncov_max : int, default=infinity
        limit the memory usage of the algorithm from [2]_ to an amount that corresponds
        to ncov_max additional copies of each correlation matrix
    column_selection: ndarray(k, dtype=int) or None
        Indices of those columns that are to be computed. If None, all columns are computed.
    diag_only: bool
        If True, the computation is restricted to the diagonal entries (autocorrelations) only.

    Returns
    -------
    lc : a :class:`LaggedCovariance <pyemma.coordinates.estimation.covariance.LaggedCovariance>` object.


    .. [1] Wu, H., Nueske, F., Paul, F., Klus, S., Koltai, P., and Noe, F. 2016. Bias reduced variational
       approximation of molecular kinetics from short off-equilibrium simulations. J. Chem. Phys. (submitted)
    .. [2] Chan, T. F., Golub G. H., LeVeque R. J. 1979. Updating formulae and pairwiese algorithms for
        computing sample variances. Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
    """
    from pyemma.coordinates.estimation.covariance import LaggedCovariance
    from pyemma.coordinates.estimation.koopman import _KoopmanEstimator
    import types
    if isinstance(weights, _string_types):
        if weights== "koopman":
            if data is None:
                raise ValueError("Data must be supplied for reweighting='koopman'")
            koop = _KoopmanEstimator(lag=lag, stride=stride, skip=skip, ncov_max=ncov_max)
            koop.estimate(data, chunksize=chunksize)
            weights = koop.weights
        elif weights == "empirical":
            weights = None
        else:
            raise ValueError("reweighting must be either 'empirical', 'koopman' "
                             "or an object with a weights(data) method.")
    elif hasattr(weights, 'weights') and type(getattr(weights, 'weights')) == types.MethodType:
        pass
    elif isinstance(weights, (list, tuple, _np.ndarray)):
        pass
    else:
        raise ValueError("reweighting must be either 'empirical', 'koopman' or an object with a weights(data) method.")

    # chunksize is an estimation parameter for now.
    lc = LaggedCovariance(c00=c00, c0t=c0t, ctt=ctt, remove_constant_mean=remove_constant_mean,
                          remove_data_mean=remove_data_mean, reversible=reversible, bessel=bessel, lag=lag,
                          weights=weights, stride=stride, skip=skip, ncov_max=ncov_max,
                          column_selection=column_selection, diag_only=diag_only)
    if data is not None:
        lc.estimate(data, chunksize=chunksize)
    else:
        lc.chunksize = chunksize
    return lc


# =========================================================================
#
# CLUSTERING ALGORITHMS
#
# =========================================================================

def cluster_mini_batch_kmeans(data=None, k=100, max_iter=10, batch_size=0.2, metric='euclidean',
                              init_strategy='kmeans++', n_jobs=None, chunksize=None, skip=0, clustercenters=None, **kwargs):
    r"""k-means clustering with mini-batch strategy

    Mini-batch k-means is an approximation to k-means which picks a randomly
    selected subset of data points to be updated in each iteration. Usually
    much faster than k-means but will likely deliver a less optimal result.

    Returns
    -------
    kmeans_mini : a :class:`MiniBatchKmeansClustering <pyemma.coordinates.clustering.MiniBatchKmeansClustering>` clustering object
        Object for mini-batch kmeans clustering.
        It holds discrete trajectories and cluster center information.

    See also
    --------
    :func:`kmeans <pyemma.coordinates.kmeans>` : for full k-means clustering


    .. autoclass:: pyemma.coordinates.clustering.kmeans.MiniBatchKmeansClustering
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.clustering.kmeans.MiniBatchKmeansClustering
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.clustering.kmeans.MiniBatchKmeansClustering
            :attributes:

    References
    ----------
    .. [1] http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

    """
    from pyemma.coordinates.clustering.kmeans import MiniBatchKmeansClustering
    res = MiniBatchKmeansClustering(n_clusters=k, max_iter=max_iter, metric=metric, init_strategy=init_strategy,
                                    batch_size=batch_size, n_jobs=n_jobs, skip=skip, clustercenters=clustercenters)
    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(cluster_mini_batch_kmeans)['chunksize'], **kwargs)
    if data is not None:
        res.estimate(data, chunksize=cs)
    else:
        res.chunksize = chunksize
    return res


def cluster_kmeans(data=None, k=None, max_iter=10, tolerance=1e-5, stride=1,
                   metric='euclidean', init_strategy='kmeans++', fixed_seed=False,
                   n_jobs=None, chunksize=None, skip=0, keep_data=False, clustercenters=None, **kwargs):
    r"""k-means clustering

    If data is given, it performs a k-means clustering and then assigns the
    data using a Voronoi discretization. It returns a :class:`KmeansClustering <pyemma.coordinates.clustering.KmeansClustering>`
    object that can be used to extract the discretized data sequences, or to
    assign other data points to the same partition. If data is not given, an
    empty :class:`KmeansClustering <pyemma.coordinates.clustering.KmeansClustering>`
    will be created that still needs to be parametrized, e.g. in a :func:`pipeline`.

    Parameters
    ----------
    data: ndarray (T, d) or list of ndarray (T_i, d) or a reader created by :func:`source`
        input data, if available in memory

    k: int
        the number of cluster centers. When not specified (None), min(sqrt(N), 5000) is chosen as default value,
        where N denotes the number of data points

    max_iter : int
        maximum number of iterations before stopping. When not specified (None), min(sqrt(N),5000) is chosen
        as default value, where N denotes the number of data points

    tolerance : float
        stop iteration when the relative change in the cost function

        :math:`C(S) = \sum_{i=1}^{k} \sum_{\mathbf x \in S_i} \left\| \mathbf x - \boldsymbol\mu_i \right\|^2`

        is smaller than tolerance.

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this
        could cause this calculation to be very slow for large data sets. Since
        molecular dynamics data is usually correlated at short timescales, it
        is often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the returned
        object is independent, so you can parametrize at a long stride, and
        still map all frames through the transformer.

    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')

    init_strategy : str
        determines if the initial cluster centers are chosen according to the kmeans++-algorithm
        or drawn uniformly distributed from the provided data set

    fixed_seed : bool or (positive) integer
        if set to true, the random seed gets fixed resulting in deterministic behavior; default is false.
        If an integer >= 0 is given, use this to initialize the random generator.

    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    skip : int, default=0
        skip the first initial n frames per trajectory.

    keep_data: boolean, default=False
        if you intend to quickly resume a non-converged kmeans iteration, set this to True.
        Otherwise the linear memory array will have to be re-created. Note that the data will also be deleted,
        if and only if the estimation converged within the given tolerance parameter.

    clustercenters: ndarray (k, dim), default=None
        if passed, the init_strategy is ignored and these centers will be iterated.

    Returns
    -------
    kmeans : a :class:`KmeansClustering <pyemma.coordinates.clustering.KmeansClustering>` clustering object
        Object for kmeans clustering.
        It holds discrete trajectories and cluster center information.


    Examples
    --------

    >>> import numpy as np
    >>> from pyemma.util.contexts import settings
    >>> import pyemma.coordinates as coor
    >>> traj_data = [np.random.random((100, 3)), np.random.random((100,3))]
    >>> with settings(show_progress_bars=False):
    ...     cluster_obj = coor.cluster_kmeans(traj_data, k=20, stride=1)
    ...     cluster_obj.get_output() # doctest: +ELLIPSIS
    [array([...

    .. seealso:: **Theoretical background**: `Wiki page <http://en.wikipedia.org/wiki/K-means_clustering>`_


    .. autoclass:: pyemma.coordinates.clustering.kmeans.KmeansClustering
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.clustering.kmeans.KmeansClustering
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.clustering.kmeans.KmeansClustering
            :attributes:

    References
    ----------
    The k-means algorithms was invented in [1]_. The term k-means was
    first used in [2]_.

    .. [1] Steinhaus, H. (1957).
        Sur la division des corps materiels en parties.
        Bull. Acad. Polon. Sci. (in French) 4, 801-804.

    .. [2] MacQueen, J. B. (1967).
        Some Methods for classification and Analysis of Multivariate Observations.
        Proceedings of 5th Berkeley Symposium on Mathematical Statistics and
        Probability 1. University of California Press. pp. 281-297

    """
    from pyemma.coordinates.clustering.kmeans import KmeansClustering
    res = KmeansClustering(n_clusters=k, max_iter=max_iter, metric=metric, tolerance=tolerance,
                           init_strategy=init_strategy, fixed_seed=fixed_seed, n_jobs=n_jobs, skip=skip,
                           keep_data=keep_data, clustercenters=clustercenters, stride=stride)
    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(cluster_kmeans)['chunksize'], **kwargs)
    if data is not None:
        res.estimate(data, chunksize=cs)
    else:
        res.chunksize = cs
    return res


def cluster_uniform_time(data=None, k=None, stride=1, metric='euclidean',
                         n_jobs=None, chunksize=None, skip=0, **kwargs):
    r"""Uniform time clustering

    If given data, performs a clustering that selects data points uniformly in
    time and then assigns the data using a Voronoi discretization. Returns a
    :class:`UniformTimeClustering <pyemma.coordinates.clustering.UniformTimeClustering>` object
    that can be used to extract the discretized data sequences, or to assign
    other data points to the same partition. If data is not given, an empty
    :class:`UniformTimeClustering <pyemma.coordinates.clustering.UniformTimeClustering>` will be created that
    still needs to be parametrized, e.g. in a :func:`pipeline`.

    Parameters
    ----------
    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created
        by source function input data, if available in memory

    k : int
        the number of cluster centers. When not specified (None), min(sqrt(N), 5000) is chosen as default value,
        where N denotes the number of data points

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this
        could cause this calculation to be very slow for large data sets. Since
        molecular dynamics data is usually correlated at short timescales, it is
        often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the returned
        object is independent, so you can parametrize at a long stride, and
        still map all frames through the transformer.

    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')

    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    skip : int, default=0
        skip the first initial n frames per trajectory.

    Returns
    -------
    uniformTime : a :class:`UniformTimeClustering <pyemma.coordinates.clustering.UniformTimeClustering>` clustering object
        Object for uniform time clustering.
        It holds discrete trajectories and cluster center information.


    .. autoclass:: pyemma.coordinates.clustering.uniform_time.UniformTimeClustering
         :members:
         :undoc-members:

         .. rubric:: Methods

         .. autoautosummary:: pyemma.coordinates.clustering.uniform_time.UniformTimeClustering
            :methods:

         .. rubric:: Attributes

         .. autoautosummary:: pyemma.coordinates.clustering.uniform_time.UniformTimeClustering
             :attributes:

    """
    from pyemma.coordinates.clustering.uniform_time import UniformTimeClustering
    res = UniformTimeClustering(k, metric=metric, n_jobs=n_jobs, skip=skip, stride=stride)
    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(cluster_uniform_time)['chunksize'], **kwargs)
    if data is not None:
        res.estimate(data, chunksize=cs)
    else:
        res.chunksize = cs
    return res


def cluster_regspace(data=None, dmin=-1, max_centers=1000, stride=1, metric='euclidean',
                     n_jobs=None, chunksize=None, skip=0, **kwargs):
    r"""Regular space clustering

    If given data, it performs a regular space clustering [1]_ and returns a
    :class:`RegularSpaceClustering <pyemma.coordinates.clustering.RegularSpaceClustering>` object that
    can be used to extract the discretized data sequences, or to assign other
    data points to the same partition. If data is not given, an empty
    :class:`RegularSpaceClustering <pyemma.coordinates.clustering.RegularSpaceClustering>` will be created
    that still needs to be parametrized, e.g. in a :func:`pipeline`.

    Regular space clustering is very similar to Hartigan's leader algorithm [2]_.
    It consists of two passes through the data. Initially, the first data point
    is added to the list of centers. For every subsequent data point, if it has
    a greater distance than dmin from every center, it also becomes a center.
    In the second pass, a Voronoi discretization with the computed centers is
    used to partition the data.

    Parameters
    ----------
    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by :func:`source
        input data, if available in memory

    dmin : float
        the minimal distance between cluster centers

    max_centers : int (optional), default=1000
        If max_centers is reached, the algorithm will stop to find more centers,
        but it is possible that parts of the state space are not properly `
        discretized. This will generate a warning. If that happens, it is
        suggested to increase dmin such that the number of centers stays below
        max_centers.

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this
        could cause this calculation to be very slow for large data sets. Since
        molecular dynamics data is usually correlated at short timescales, it is
        often sufficient to estimate transformations at a longer stride. Note
        that the stride option in the get_output() function of the returned
        object is independent, so you can parametrize at a long stride, and
        still map all frames through the transformer.

    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')

    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.


    Returns
    -------
    regSpace : a :class:`RegularSpaceClustering <pyemma.coordinates.clustering.RegularSpaceClustering>` clustering  object
        Object for regular space clustering.
        It holds discrete trajectories and cluster center information.


    .. autoclass:: pyemma.coordinates.clustering.regspace.RegularSpaceClustering
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.clustering.regspace.RegularSpaceClustering
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.clustering.regspace.RegularSpaceClustering
            :attributes:

    References
    ----------
    .. [1] Prinz J-H, Wu H, Sarich M, Keller B, Senne M, Held M, Chodera JD, Schuette Ch and Noe F. 2011.
        Markov models of molecular kinetics: Generation and Validation.
        J. Chem. Phys. 134, 174105.

    .. [2] Hartigan J. Clustering algorithms.
        New York: Wiley; 1975.

    """
    if dmin == -1:
        raise ValueError("provide a minimum distance for clustering, e.g. 2.0")
    from pyemma.coordinates.clustering.regspace import RegularSpaceClustering as _RegularSpaceClustering
    res = _RegularSpaceClustering(dmin, max_centers=max_centers, metric=metric,
                                  n_jobs=n_jobs, stride=stride, skip=skip)
    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(cluster_regspace)['chunksize'], **kwargs)
    if data is not None:
        res.estimate(data, chunksize=cs)
    else:
        res.chunksize = cs
    return res


def assign_to_centers(data=None, centers=None, stride=1, return_dtrajs=True,
                      metric='euclidean', n_jobs=None, chunksize=None, skip=0, **kwargs):
    r"""Assigns data to the nearest cluster centers

    Creates a Voronoi partition with the given cluster centers. If given
    trajectories as data, this function will by default discretize the
    trajectories and return discrete trajectories of corresponding lengths.
    Otherwise, an assignment object will be returned that can be used to
    assign data later or can serve as a pipeline stage.

    Parameters
    ----------
    data : ndarray or list of arrays or reader created by source function
        data to be assigned

    centers : path to file or ndarray or a reader created by source function
        cluster centers to use in assignment of data

    stride : int, optional, default = 1
        assign only every n'th frame to the centers. Usually you want to assign
        all the data and only use a stride during calculation the centers.

    return_dtrajs : bool, optional, default = True
        If True, it will return the discretized trajectories obtained from
        assigning the coordinates in the data input. This will only have effect
        if data is given. When data is not given or return_dtrajs is False,
        the :class:'AssignCenters <_AssignCenters>' object will be returned.

    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')

    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.

    chunksize: int, default=None
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed. If None is passed,
        use the default value of the underlying reader/data source. Choose zero to
        disable chunking at all.

    Returns
    -------
    assignment : list of integer arrays or an :class:`AssignCenters <pyemma.coordinates.clustering.AssignCenters>` object
        assigned data

    Examples
    --------
    Load data to assign to clusters from 'my_data.csv' by using the cluster
    centers from file 'my_centers.csv'

    >>> import numpy as np

    Generate some random data and choose 10 random centers:

    >>> data = np.random.random((100, 3))
    >>> cluster_centers = data[np.random.randint(0, 99, size=10)]
    >>> dtrajs = assign_to_centers(data, cluster_centers)
    >>> print(dtrajs) # doctest: +ELLIPSIS
    [array([...


    .. autoclass:: pyemma.coordinates.clustering.assign.AssignCenters
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.clustering.assign.AssignCenters
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.clustering.assign.AssignCenters
            :attributes:

    """
    if centers is None:
        raise ValueError('You have to provide centers in form of a filename'
                         ' or NumPy array or a reader created by source function')
    from pyemma.coordinates.clustering.assign import AssignCenters
    res = AssignCenters(centers, metric=metric, n_jobs=n_jobs, skip=skip, stride=stride)
    from pyemma.util.reflection import get_default_args
    cs = _check_old_chunksize_arg(chunksize, get_default_args(assign_to_centers)['chunksize'], **kwargs)
    if data is not None:
        res.estimate(data, chunksize=cs)
        if return_dtrajs:
            return res.dtrajs
    else:
        res.chunksize = cs

    return res

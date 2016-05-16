
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
import numpy as _np
import logging as _logging

from pyemma.util import types as _types
# lift this function to the api
from pyemma.coordinates.util.stat import histogram

from six import string_types as _string_types
from six.moves import range, zip

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
           'histogram',
           'pipeline',
           'discretizer',
           'save_traj',
           'save_trajs',
           'pca',  # transform
           'tica',
           'cluster_regspace',  # cluster
           'cluster_kmeans',
           'cluster_mini_batch_kmeans',
           'cluster_uniform_time',
           'assign_to_centers',
           ]


# ==============================================================================
#
# DATA PROCESSING
#
# ==============================================================================

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
def load(trajfiles, features=None, top=None, stride=1, chunk_size=None, **kw):
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
           * PDB trajectory format (.pdb)
           * TINKER (.arc),
           * MDTRAJ (.hdf5)
           * LAMMPS trajectory format (.lammpstrj)

        Raw data can be in the following format:

           * tabulated ASCII (.dat, .txt)
           * binary python (.npy, .npz)

    features : MDFeaturizer, optional, default = None
        a featurizer object specifying how molecular dynamics files should
        be read (e.g. intramolecular distances, angles, dihedrals, etc).

    top : str, optional, default = None
        A molecular topology file, e.g. in PDB (.pdb) format

    stride : int, optional, default = 1
        Load only every stride'th frame. By default, every frame is loaded

    chunk_size: int, optional, default = 100
        The chunk size at which the input file is being processed.

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

    if isinstance(trajfiles, _string_types) or (
        isinstance(trajfiles, (list, tuple))
            and (any(isinstance(item, (list, tuple, _string_types)) for item in trajfiles)
                 or len(trajfiles) is 0)):
        reader = create_file_reader(trajfiles, top, features, chunk_size=chunk_size if chunk_size is not None else 0, **kw)
        trajs = reader.get_output(stride=stride)
        if len(trajs) == 1:
            return trajs[0]
        else:
            return trajs
    else:
        raise ValueError('unsupported type (%s) of input' % type(trajfiles))


def source(inp, features=None, top=None, chunk_size=None, **kw):
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
                [['traj1_0.xtc', 'traj1_1.xtc'], 'traj2_full.xtc'], ['traj3_0.xtc, ...]]
           the grouped fragments will be treated as a joint trajectory.

    features : MDFeaturizer, optional, default = None
        a featurizer object specifying how molecular dynamics files should be
        read (e.g. intramolecular distances, angles, dihedrals, etc). This
        parameter only makes sense if the input comes in the form of molecular
        dynamics trajectories or data, and will otherwise create a warning and
        have no effect.

    top : str, optional, default = None
        A topology file name. This is needed when molecular dynamics
        trajectories are given and no featurizer is given.
        In this case, only the Cartesian coordinates will be read.

    chunk_size: int, optional, default = 100 for file readers and 5000 for
        already loaded data The chunk size at which the input file is being
        processed.

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

    >>> data = np.random.random(int(1e7))
    >>> reader = source(data, chunk_size=5000)
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
    from pyemma.coordinates.data.util.reader_utils import create_file_reader
    # CASE 1: input is a string or list of strings
    # check: if single string create a one-element list
    if isinstance(inp, _string_types) or (
            isinstance(inp, (list, tuple))
            and (any(isinstance(item, (list, tuple, _string_types)) for item in inp) or len(inp) is 0)):
        reader = create_file_reader(inp, top, features, chunk_size=chunk_size if chunk_size is not None else 100, **kw)

    elif isinstance(inp, _np.ndarray) or (isinstance(inp, (list, tuple))
                                          and (any(isinstance(item, _np.ndarray) for item in inp) or len(inp) is 0)):
        # CASE 2: input is a (T, N, 3) array or list of (T_i, N, 3) arrays
        # check: if single array, create a one-element list
        # check: do all arrays have compatible dimensions (*, N, 3)? If not: raise ValueError.
        # check: if single array, create a one-element list
        # check: do all arrays have compatible dimensions (*, N)? If not: raise ValueError.
        # create MemoryReader
        from pyemma.coordinates.data.data_in_memory import DataInMemory as _DataInMemory
        reader = _DataInMemory(inp, chunksize=chunk_size if chunk_size else 5000, **kw)
    else:
        raise ValueError('unsupported type (%s) of input' % type(inp))

    return reader


def pipeline(stages, run=True, stride=1, chunksize=100):
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
    chunksize : int, optiona, default = 100
        how many datapoints to process as a batch at one step

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
                chunksize=100):
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

    chunksize : int, optiona, default = 100
        how many datapoints to process as a batch at one step

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

    >>> import numpy as np
    >>> from pyemma.coordinates import source, pca, cluster_regspace, discretizer
    >>> from pyemma.datasets import get_bpti_test_data
    >>> reader = source(get_bpti_test_data()['trajs'], top=get_bpti_test_data()['top'])
    >>> transform = pca(dim=2)
    >>> cluster = cluster_regspace(dmin=0.1)
    >>> disc = discretizer(reader, transform, cluster)

    Finally you want to run the pipeline:

    >>> disc.parametrize()

    Access the the discrete trajectories and saving them to files:

    >>> disc.dtrajs # doctest: +ELLIPSIS
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


def save_traj(traj_inp, indexes, outfile, top=None, stride = 1, chunksize=1000, verbose=False):
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

    chunksize : int. Default 1000.
        The chunksize for reading input trajectory files. If :py:obj:`traj_inp`
        is a :py:func:`pyemma.coordinates.data.feature_reader.FeatureReader` object,
        this input variable will be ignored and :py:obj:`traj_inp.chunksize` will be used instead.

    verbose : boolean, default is False
        Verbose output while looking for :py:obj`indexes` in the :py:obj:`traj_inp.trajfiles`

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
                         % (len(trajfiles), indexes[0].max()))

    traj = frames_from_files(trajfiles, top, indexes, chunksize, stride, reader=reader)

    # Return to memory as an mdtraj trajectory object
    if outfile is None:
        return traj
    # or to disk as a molecular trajectory file
    else:
        traj.save(outfile)

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
        import os

        _, fmt = os.path.splitext(traj_inp.filenames[0])
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

def _get_input_stage(previous_stage):
    # this is a pipelining stage, so let's parametrize from it
    from pyemma.coordinates.data._base.iterable import Iterable
    from pyemma.coordinates.data.data_in_memory import DataInMemory as _DataInMemory

    if isinstance(previous_stage, Iterable):
        inputstage = previous_stage
    # second option: data is array or list of arrays
    else:
        data = _types.ensure_traj_list(previous_stage)
        inputstage = _DataInMemory(data)

    return inputstage


def _param_stage(previous_stage, this_stage, stride=1, chunk_size=0):
    r""" Parametrizes the given pipelining stage if a valid source is given.

    Parameters
    ----------
    source : one of the following: None, Transformer (subclass), ndarray, list
        of ndarrays data source from which this transformer will be parametrized.
        If None, there is no input data and the stage will be returned without
        any other action.
    stage : the transformer object to be parametrized given the source input.

    """
    # no input given - nothing to do
    if previous_stage is None:
        return this_stage

    input_stage = _get_input_stage(previous_stage)
    input_stage.chunksize = chunk_size
    assert input_stage.default_chunksize == chunk_size
    # parametrize transformer
    this_stage.data_producer = input_stage
    this_stage.chunksize = input_stage.chunksize
    assert this_stage.chunksize == chunk_size
    this_stage.estimate(X=input_stage, stride=stride)
    return this_stage


def pca(data=None, dim=-1, var_cutoff=0.95, stride=1, mean=None):
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

    Examples
    --------
    Create some input data:

    >>> import numpy as np
    >>> from pyemma.coordinates import pca
    >>> data = np.ones((1000, 2))
    >>> data[0, -1] = 0

    Project all input data on the first principal component:

    >>> pca_obj = pca(data, dim=1)
    >>> pca_obj.get_output() # doctest: +ELLIPSIS
    [array([[-0.99900001],
           [ 0.001     ],
           [ 0.001     ],...

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

    res = PCA(dim=dim, var_cutoff=var_cutoff, mean=None)
    return _param_stage(data, res, stride=stride)


def tica(data=None, lag=10, dim=-1, var_cutoff=0.95, kinetic_map=True, stride=1,
         force_eigenvalues_le_one=False, mean=None, remove_mean=True):
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

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this
        could cause this calculation to be very slow for large data sets. Since
        molecular dynamics data is usually correlated at short timescales, it is
        often sufficient to estimate transformations at a longer stride. Note
        that the stride option in the get_output() function of the returned
        object is independent, so you can parametrize at a long stride, and
        still map all frames through the transformer.

    force_eigenvalues_le_one : boolean, deprecated (eigenvalues are always <= 1, since 2.1)
        Compute covariance matrix and time-lagged covariance matrix such
        that the generalized eigenvalues are always guaranteed to be <= 1.

    mean : ndarray, optional, default None
        This option is deprecated, and setting this value is non-effective.

    remove_mean: bool, optional, default True
        remove mean during covariance estimation. Should not be turned off.


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

        C_0 &=      (X_t - \mu)^T (X_t - \mu) \\
        C_{\tau} &= (X_t - \mu)^T (X_t + \tau - \mu)

    and solves the eigenvalue problem

    .. math:: C_{\tau} r_i = C_0 \lambda_i r_i,

    where :math:`r_i` are the independent components and :math:`\lambda_i` are
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

    .. [4] Noe, F. and C. Clementi. 2015.
        Kinetic distance and kinetic maps from molecular dynamics simulation
        (in preparation).

    """
    from pyemma.coordinates.transform.tica import TICA
    if mean is not None:
        import warnings
        warnings.warn("user provided mean for TICA is deprecated and its value is ignored.")

    res = TICA(lag, dim=dim, var_cutoff=var_cutoff, kinetic_map=kinetic_map,
               mean=mean, remove_mean=remove_mean)
    return _param_stage(data, res, stride=stride)


# =========================================================================
#
# CLUSTERING ALGORITHMS
#
# =========================================================================

def cluster_mini_batch_kmeans(data=None, k=100, max_iter=10, batch_size=0.2, metric='euclidean',
                              init_strategy='kmeans++', n_jobs=None, chunk_size=5000):
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
                                    batch_size=batch_size, n_jobs=n_jobs)
    return _param_stage(data, res, stride=1, chunk_size=chunk_size)


def cluster_kmeans(data=None, k=None, max_iter=10, tolerance=1e-5, stride=1,
                   metric='euclidean', init_strategy='kmeans++', fixed_seed=False, n_jobs=None, chunk_size=5000):
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

    fixed_seed : bool
        if set to true, the random seed gets fixed resulting in deterministic behavior; default is false

    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.

    chunk_size: int, default=5000
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed.

    Returns
    -------
    kmeans : a :class:`KmeansClustering <pyemma.coordinates.clustering.KmeansClustering>` clustering object
        Object for kmeans clustering.
        It holds discrete trajectories and cluster center information.


    Examples
    --------

    >>> import numpy as np
    >>> import pyemma.coordinates as coor
    >>> traj_data = [np.random.random((100, 3)), np.random.random((100,3))]
    >>> cluster_obj = coor.cluster_kmeans(traj_data, k=20, stride=1)
    >>> cluster_obj.get_output() # doctest: +ELLIPSIS
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
                           init_strategy=init_strategy, fixed_seed=fixed_seed, n_jobs=n_jobs)
    return _param_stage(data, res, stride=stride, chunk_size=chunk_size)


def cluster_uniform_time(data=None, k=None, stride=1, metric='euclidean', n_jobs=None, chunk_size=5000):
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

    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.

    chunk_size: int, default=5000
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed.

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
    res = UniformTimeClustering(k, metric=metric, n_jobs=n_jobs)
    return _param_stage(data, res, stride=stride, chunk_size=chunk_size)


def cluster_regspace(data=None, dmin=-1, max_centers=1000, stride=1, metric='euclidean', n_jobs=None, chunk_size=5000):
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

    chunk_size: int, default=5000
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed.


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
    res = _RegularSpaceClustering(dmin, max_centers=max_centers, metric=metric, n_jobs=n_jobs)
    return _param_stage(data, res, stride=stride, chunk_size=chunk_size)


def assign_to_centers(data=None, centers=None, stride=1, return_dtrajs=True,
                      metric='euclidean', n_jobs=None, chunk_size=5000):
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
        If set to 1, all input data will be used for estimation. Note that
        this could cause this calculation to be very slow for large data sets.
        Since molecular dynamics data is usually correlated at short timescales,
        it is often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the
        returned object is independent, so you can parametrize at a long stride,
        and still map all frames through the transformer.

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

    chunk_size: int, default=3000
        Number of data frames to process at once. Choose a higher value here,
        to optimize thread usage and gain processing speed.

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
    res = AssignCenters(centers, metric=metric, n_jobs=n_jobs)

    parametrized_stage = _param_stage(data, res, stride=stride, chunk_size=chunk_size)
    if return_dtrajs and data is not None:
        return parametrized_stage.dtrajs

    return parametrized_stage

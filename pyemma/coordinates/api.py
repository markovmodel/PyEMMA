
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""User-API for the pyemma.coordinates package

.. currentmodule:: pyemma.coordinates.api
"""

from pyemma.util.annotators import deprecated
from pyemma.util.log import getLogger as _getLogger
from pyemma.util import types as _types

from pyemma.coordinates.pipelines import Discretizer as _Discretizer
from pyemma.coordinates.pipelines import Pipeline as _Pipeline
# io
from pyemma.coordinates.data.featurizer import MDFeaturizer as _MDFeaturizer
from pyemma.coordinates.data.feature_reader import FeatureReader as _FeatureReader
from pyemma.coordinates.data.data_in_memory import DataInMemory as _DataInMemory
from pyemma.coordinates.data.util.reader_utils import create_file_reader as _create_file_reader, \
    preallocate_empty_trajectory as _preallocate_empty_trajectory, enforce_top as _enforce_top, \
    copy_traj_attributes as _copy_traj_attributes
from pyemma.coordinates.data.frames_from_file import frames_from_file as _frames_from_file
# transforms
from pyemma.coordinates.transform.transformer import Transformer as _Transformer
from pyemma.coordinates.transform.pca import PCA as _PCA
from pyemma.coordinates.transform.tica import TICA as _TICA
# clustering
from pyemma.coordinates.clustering.kmeans import KmeansClustering as _KmeansClustering
from pyemma.coordinates.clustering.kmeans import MiniBatchKmeansClustering as _MiniBatchKmeansClustering
from pyemma.coordinates.clustering.uniform_time import UniformTimeClustering as _UniformTimeClustering
from pyemma.coordinates.clustering.regspace import RegularSpaceClustering as _RegularSpaceClustering
from pyemma.coordinates.clustering.assign import AssignCenters as _AssignCenters

# stat
from pyemma.coordinates.util.stat import histogram

# types
from mdtraj import Topology as _Topology, Trajectory as _Trajectory

import numpy as _np
import itertools as _itertools

_logger = _getLogger('coordinates.api')

__docformat__ = "restructuredtext en"
__author__ = "Frank Noe, Martin Scherer"
__copyright__ = "Copyright 2015, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
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
           'cluster_uniform_time',
           'assign_to_centers',
           'feature_reader',  # deprecated:
           'memory_reader',
           'kmeans',
           'regspace',
           'assign_centers',
           'uniform_time']


# ==============================================================================
#
# DATA PROCESSING
#
# ==============================================================================

def featurizer(topfile):
    r""" Featurizer to select features from MD data.

    Parameters
    ----------
    topfile : str
        path to topology file (e.g pdb file)

    Returns
    -------
    feat : :class:`Featurizer <pyemma.coordinates.data.featurizer.MDFeaturizer>`

    See also
    --------
    data.MDFeaturizer

    Examples
    --------

    Create a featurizer and add backbone torsion angles to active features.
    Then use it in :func:`source`

    >>> import pyemma.coordinates # doctest: +SKIP
    >>> feat = pyemma.coordinates.featurizer('my_protein.pdb') # doctest: +SKIP
    >>> feat.add_backbone_torsions() # doctest: +SKIP
    >>> reader = pyemma.coordinates.source(["my_traj01.xtc", "my_traj02.xtc"], features=feat) # doctest: +SKIP


    .. autoclass:: pyemma.coordinates.data.featurizer.MDFeaturizer
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.data.featurizer.MDFeaturizer
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.data.featurizer.MDFeaturizer
            :attributes:
    """
    return _MDFeaturizer(topfile)


# TODO: DOC - which topology file formats does mdtraj support? Find out and complete docstring
def load(trajfiles, features=None, top=None, stride=1, chunk_size=100):
    r""" Loads coordinate features into memory.

    If your memory is not big enough consider the use of **pipeline**, or use the stride option to subsample the data.

    Parameters
    ----------
    trajfiles : str or list of str
        A filename or a list of filenames to trajectory files that can be processed by pyemma.
        Both molecular dynamics trajectory files and raw data files (tabulated ASCII or binary) can be loaded.

        When molecular dynamics trajectory files are loaded either a featurizer must be specified (for
        reading specific quantities such as distances or dihedrals), or a topology file (in that case only
        Cartesian coordinates will be read). In the latter case, the resulting feature vectors will have length
        3N for each trajectory frame, with N being the number of atoms and (x1, y1, z1, x2, y2, z2, ...) being
        the sequence of coordinates in the vector.

        Molecular dynamics trajectory files are loaded through mdtraj (http://mdtraj.org/latest/),
        and can possess any of the mdtraj-compatible trajectory formats including:

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
        a featurizer object specifying how molecular dynamics files should be read (e.g. intramolecular distances,
        angles, dihedrals, etc).

    top : str, optional, default = None
        A molecular topology file, e.g. in PDB (.pdb) format

    stride : int, optional, default = 1
        Load only every stride'th frame. By default, every frame is loaded

    chunk_size: int, optional, default = 100
        The chunk size at which the input file is being processed.

    Returns
    -------
    data : ndarray or list of ndarray
        If a single filename was given as an input (and unless the format is .npz), the return will be a single ndarray
        of size (T, d), where T is the number of time steps in the trajectory and d is the number of features
        (coordinates, observables). When reading from molecular dynamics data without a specific featurizer,
        each feature vector will have size d=3N and will hold the Cartesian coordinates in the sequence
        (x1, y1, z1, x2, y2, z2, ...).
        If multiple filenames were given, or if the file is a .npz holding multiple arrays, the result is a list
        of appropriately shaped arrays

    See also
    --------
    :func:`pyemma.coordinates.pipeline`
        if your memory is not big enough, use pipeline to process it in a streaming manner

    Examples
    --------

    >>> from pyemma.coordinates import load
    >>> files = ['traj01.xtc', 'traj02.xtc'] # doctest: +SKIP
    >>> output = load(files, top='my_structure.pdb') # doctest: +SKIP

    """
    if isinstance(trajfiles, basestring) or (
        isinstance(trajfiles, (list, tuple))
            and (any(isinstance(item, basestring) for item in trajfiles) or len(trajfiles) is 0)):
        reader = _create_file_reader(trajfiles, top, features, chunk_size=chunk_size)
        trajs = reader.get_output(stride=stride)
        if len(trajs) == 1:
            return trajs[0]
        else:
            return trajs
    else:
        raise ValueError('unsupported type (%s) of input' % type(trajfiles))


def source(inp, features=None, top=None, chunk_size=None):
    r""" Wraps input as data source for pipeline.

    Use this function to construct the first stage of a data processing :func:`pipeline`.

    Parameters
    ----------
    inp : str (file name) or ndarray or list of strings (file names) or list of ndarrays
        The inp file names or input data. Can be given in any of these ways:

        1. File name of a single trajectory. It can have any of the molecular dynamics trajectory formats or
           raw data formats specified in :py:func:`load`.
        2. List of trajectory file names. It can have any of the molecular dynamics trajectory formats or
           raw data formats specified in :py:func:`load`.
        3. Molecular dynamics trajectory in memory as a numpy array of shape (T, N, 3) with T time steps, N atoms
           each having three (x,y,z) spatial coordinates.
        4. List of molecular dynamics trajectories in memory, each given as a numpy array of shape (T_i, N, 3),
           where trajectory i has T_i time steps and all trajectories have shape (N, 3).
        5. Trajectory of some features or order parameters in memory
           as a numpy array of shape (T, N) with T time steps and N dimensions.
        6. List of trajectories of some features or order parameters in memory, each given as a numpy array
           of shape (T_i, N), where trajectory i has T_i time steps and all trajectories have N dimensions.
        7. List of NumPy array files (.npy) of shape (T, N). Note these
           arrays are not being loaded completely, but mapped into memory (read-only).
        8. List of tabulated ASCII files of shape (T, N).

    features : MDFeaturizer, optional, default = None
        a featurizer object specifying how molecular dynamics files should be read (e.g. intramolecular distances,
        angles, dihedrals, etc). This parameter only makes sense if the input comes in the form of molecular dynamics
        trajectories or data, and will otherwise create a warning and have no effect

    top : str, optional, default = None
        A topology file name. This is needed when molecular dynamics trajectories are given and no featurizer is given.
        In this case, only the Cartesian coordinates will be read.

    chunk_size: int, optional, default = 100 for file readers and 5000 for already loaded data
        The chunk size at which the input file is being processed.

    Returns
    -------
    reader obj: type depends on input data

        1. :class:`FeatureReader <pyemma.coordinates.data.feature_reader.FeatureReader>` for MD-data
        2. :class:`NumPyFileReader <pyemma.coordinates.data.numpy_filereader.NumPyFileReader>` for .npy files
        3. :class:`PyCSVReader <pyemma.coordinates.data.py_csv_reader.PyCSVReader>` for csv files.
        4. :class:`DataInMemory <pyemma.coordinates.data.data_in_memory.DataInMemory>` for already loaded data (e.g NumPy arrays)

    See also
    --------
    :func:`pyemma.coordinates.pipeline`
        The data input is the first stage for your pipeline. Add other stages to it and build a pipeline
        to analyze big data in streaming mode.

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

    Create a reader for huge NumPy in-memory arrays to process them in huge chunks
    to avoid memory issues:
    >>> data = np.random.random(int(1e7))
    >>> reader = source(data, chunk_size=5000)
    >>> from pyemma.coordinates import cluster_regspace
    >>> regspace = cluster_regspace(reader, dmin=0.1)

    """
    # CASE 1: input is a string or list of strings
    # check: if single string create a one-element list
    if isinstance(inp, basestring) or (isinstance(inp, (list, tuple))
                                       and (any(isinstance(item, basestring) for item in inp) or len(inp) is 0)):
        reader = _create_file_reader(inp, top, features, chunk_size=chunk_size if chunk_size else 100)

    elif isinstance(inp, _np.ndarray) or (isinstance(inp, (list, tuple))
                                      and (any(isinstance(item, _np.ndarray) for item in inp) or len(inp) is 0)):
        # CASE 2: input is a (T, N, 3) array or list of (T_i, N, 3) arrays
        # check: if single array, create a one-element list
        # check: do all arrays have compatible dimensions (*, N, 3)? If not: raise ValueError.
        # check: if single array, create a one-element list
        # check: do all arrays have compatible dimensions (*, N)? If not: raise ValueError.
        # create MemoryReader
        reader = _DataInMemory(inp, chunksize=chunk_size if chunk_size else 5000)
    else:
        raise ValueError('unsupported type (%s) of input' % type(inp))

    return reader


def pipeline(stages, run=True, stride=1, chunksize=100):
    r""" Data analysis pipeline.

    Constructs a data analysis :class:`Pipeline <pyemma.coordinates.pipelines.Pipeline>` and parametrizes it
    (unless prevented).
    If this function takes too long, consider loading data in memory. Alternatively if the data is to large to be loaded
    into memory make use of the stride parameter.

    Parameters
    ----------
    stages : data input or list of pipeline stages
        If given a single pipeline stage this must be a data input constructed by :py:func:`source`.
        If a list of pipelining stages are given, the first stage must be a data input constructed by :py:func:`source`.
    run : bool, optional, default = True
        If True, the pipeline will be parametrized immediately with the given stages. If only an input stage is given,
        the run flag has no effect at this time. True also means that the pipeline will be immediately re-parametrized
        when further stages are added to it.
        *Attention* True means this function may take a long time to compute.
        If False, the pipeline will be passive, i.e. it will not do any computations before you call parametrize()
    stride : int, optional, default = 1
        If set to 1, all input data will be used throughout the pipeline to parametrize its stages. Note that this
        could cause the parametrization step to be very slow for large data sets. Since molecular dynamics data is
        usually correlated at short timescales, it is often sufficient to parametrize the pipeline at a longer stride.
        See also stride option in the output functions of the pipeline.
    chunksize : int, optiona, default = 100
        how many datapoints to process as a batch at one step

    Returns
    -------
    pipe : :class:`Pipeline <pyemma.coordinates.pipelines.Pipeline>`
        A pipeline object that is able to conduct big data analysis with limited memory in streaming mode.

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

    if not isinstance(stages, list):
        stages = [stages]
    p = _Pipeline(stages, param_stride=stride, chunksize=chunksize)
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

    This function is identical to calling :func:`pipeline` with the three stages, it is only meant as a guidance
    for the (probably) most common usage cases of a pipeline.

    Parameters
    ----------

    reader : instance of :class:`pyemma.coordinates.data.reader.ChunkedReader`
        The reader instance provides access to the data. If you are working with
        MD data, you most likely want to use a FeatureReader.

    transform : instance of :class: `pyemma.coordinates.Transformer`
        an optional transform like PCA/TICA etc.

    cluster : instance of :class: `pyemma.coordinates.AbstractClustering` clustering Transformer (optional)
        a cluster algorithm to assign transformed data to discrete states.

    stride : int, optional, default = 1
        If set to 1, all input data will be used throughout the pipeline to parametrize its stages. Note that this
        could cause the parametrization step to be very slow for large data sets. Since molecular dynamics data is
        usually correlated at short timescales, it is often sufficient to parametrize the pipeline at a longer stride.
        See also stride option in the output functions of the pipeline.

    chunksize : int, optiona, default = 100
        how many datapoints to process as a batch at one step

    Returns
    -------
    pipe : a :class:`Pipeline <pyemma.coordinates.pipelines.Discretizer>` object
        A pipeline object that is able to streamline data analysis of large amounts of input data
        with limited memory in streaming mode.


    Examples
    --------

    Construct a discretizer pipeline processing all data
    with a PCA transformation and cluster the principal components
    with uniform time clustering:

    >>> import numpy as np
    >>> from pyemma.coordinates import source, pca, cluster_regspace, discretizer
    >>> data = np.random.random((1000, 3))
    >>> reader = source(data)
    >>> transform = pca(dim=2)
    >>> cluster = cluster_regspace(dmin=0.1)
    >>> disc = discretizer(reader, transform, cluster)

    Finally you want to run the pipeline:

    >>> disc.parametrize()

    Access the the discrete trajectories and saving them to files:

    >>> disc.dtrajs # doctest: +ELLIPSIS
    [array([...

    This will store the discrete trajectory to "traj01.dtraj":

    >>> disc.save_dtrajs() # doctest: +SKIP

    """
    if cluster is None:
        _logger.warning('You did not specify a cluster algorithm.'
                        ' Defaulting to kmeans(k=100)')
        cluster = _KmeansClustering(n_clusters=100)
    disc = _Discretizer(reader, transform, cluster, param_stride=stride)
    if run:
        disc.parametrize()
    return disc


@deprecated('Use either pyemma.coordinates.source() or pyemma.coordinates.load()')
def feature_reader(trajfiles, topfile):
    r"""*Deprecated.* Constructs a molecular feature reader.

    This funtion is deprecated. Use :func:`source` instead

    Parameters
    ----------

    trajfiles : list of str
        list of filenames to read sequentially
    topfile : str
        path to a topology file (eg. pdb)

    Returns
    -------
    obj : :class:`io.FeatureReader`

    Notes
    -----
    To select features refer to the documentation of the :class:`io.featurizer.MDFeaturizer`

    See also
    --------
    pyemma.coordinates.data.FeatureReader
        Reader object

    """
    return _FeatureReader(trajfiles, topfile)


@deprecated("Please use pyemma.coordinates.load()")
def memory_reader(data):
    r"""*Deprecated.* Constructs a reader from an in-memory ndarray.

    This funtion is deprecated. Use :func:`source` instead

    Parameters
    ----------
    data : (N,d) ndarray
        array with N frames of d dimensions

    Returns
    -------
    obj : :class:`DataInMemory`

    See also
    --------
    pyemma.coordinates.data.DataInMemory
        Reader object

    """
    return _DataInMemory(data)


def save_traj(traj_inp, indexes, outfile, top=None, stride=1, chunksize=1000, verbose=False):
    r""" Saves a sequence of frames as a single trajectory.

    Extracts the specified sequence of time/trajectory indexes from traj_inp
    and saves it to one single molecular dynamics trajectory file. The output format will be determined
    by the outfile name.

    Parameters
    ----------

    traj_inp :
        traj_inp can be of two types.

            1. a python list of strings containing the filenames associated with the indices in
            :py:obj:`indexes`. With this type of input, a :py:obj:`topfile` is mandatory.

            2. a :py:func:`pyemma.coordinates.data.feature_reader.FeatureReader` object containing the filename
            list in :py:obj:`traj_inp.trajfiles`. Please use :py:func:`pyemma.coordinates.source` to construct it.
            With this type of input, the input :py:obj:`topfile` will be ignored. and :py:obj:`traj_inp.topfile` will
            be used instead

    indexes : ndarray(T, 2) or list of ndarray(T_i, 2)
        A (T x 2) array for writing a trajectory of T time steps. Each row contains two indexes (i, t), where
        i is the index of the trajectory from the input and t is the index of the time step within the trajectory.
        If a list of index arrays are given, these will be simply concatenated, i.e. they will be written
        subsequently in the same trajectory file.

    outfile : str.
        The name of the output file. Its extension will determine the file type written. Example: "out.dcd"
        If set to None, the trajectory object is returned to memory

    top : str, mdtraj.Trajectory, or mdtraj.Topology
        The topology needed to read the files in the list :py:obj:`traj_inp`. If :py:obj:`traj_inp` is not a list,
        this parameter is ignored.

    stride  : integer, default is 1
        This parameter informs :py:func:`save_traj` about the stride used in :py:obj:`indexes`. Typically, :py:obj:`indexes`
        contains frame-indexes that match exactly the frames of the files contained in :py:obj:`traj_inp.trajfiles`.
        However, in certain situations, that might not be the case. Examples are cases in which a stride value != 1
        was used when reading/featurizing/transforming/discretizing the files contained in :py:obj:`traj_inp.trajfiles`.

    chunksize : int. Default 1000.
        The chunksize for reading input trajectory files. If :py:obj:`traj_inp` is a
        :py:func:`pyemma.coordinates.data.feature_reader.FeatureReader` object, this input variable will be ignored and
        :py:obj:`traj_inp.chunksize` will be used instead.

    verbose : boolean, default is False
        Verbose output while looking for :py:obj`indexes` in the :py:obj:`traj_inp.trajfiles`

    Returns
    -------
    traj : :py:obj:`mdtraj.Trajectory` object
        Will only return this object if :py:obj:`outfile` is None
    """

    # Determine the type of input and extract necessary parameters
    if isinstance(traj_inp, _FeatureReader):
        trajfiles = traj_inp.trajfiles
        top  = traj_inp.topfile
        chunksize = traj_inp.chunksize
    else:
        # Do we have what we need?
        assert isinstance(traj_inp, list), "traj_inp has to be of type list, not %"%type(traj_inp)
        assert isinstance(top,(str,_Topology, _Trajectory)), "traj_inp cannot be a list of files without an input " \
                                        "top of type str (eg filename.pdb), mdtraj.Trajectory or mdtraj.Topology. " \
                                        "Got type %s instead"%type(top)
        trajfiles = traj_inp

    # Enforce the input topology to actually be an md.Topology object
    top = _enforce_top(top)

    # Convert to index (T,2) array if parsed a list or a list of arrays
    indexes = _np.vstack(indexes)

    # Check that we've been given enough filenames
    assert (len(trajfiles) >= indexes[:,0].max()), "traj_inp contains %u trajfiles, " \
                                                   "but indexes will ask for file nr. %u"%(len(trajfiles), indexes[0].max())

    # Instantiate  a list of iterables that will contain mdtraj trajectory objects
    trajectory_iterator_list = []

    # Cycle only over files that are actually mentioned in "indexes"
    file_idxs, file_pos = _np.unique(indexes[:, 0], return_inverse=True)
    for ii, ff in enumerate(file_idxs):
        # Slice the indexes array (frame column) where file ff was mentioned
        frames = indexes[file_pos == ii, 1]
        # Store the trajectory object that comes out of _frames_from_file
        # directly as an iterator in trajectory_iterator_list
        trajectory_iterator_list.append(_itertools.islice(_frames_from_file(trajfiles[ff],
                                                                            top,
                                                                            frames, chunksize=chunksize,
                                                                            verbose=verbose, stride = stride,
                                                                            copy_not_join=True),
                                                          None)
                                        )
    # Prepare the trajectory object
    traj = _preallocate_empty_trajectory(top, indexes.shape[0])

    # Iterate directly over the index of files and pick the trajectory that you need from the iterator list
    for ii, traj_idx in enumerate(file_pos):
        # Append the trajectory from the respective list of iterators
        # and advance that iterator
        traj = _copy_traj_attributes(traj, trajectory_iterator_list[traj_idx].next(), ii)

    # Return to memory as an mdtraj trajectory object
    if outfile is None:
        return traj
    # or to disk as a molecular trajectory file
    else:
        traj.save(outfile)

    _logger.info("Created file %s" % outfile)


def save_trajs(traj_inp, indexes, prefix = 'set_', fmt = None, outfiles = None,
               inmemory = False, stride = 1, verbose = False):
    r""" Saves sequences of frames as multiple trajectories.

    Extracts a number of specified sequences of time/trajectory indexes from the input loader
    and saves them in a set of molecular dynamics trajectories.
    The output filenames are obtained by prefix + str(n) + .fmt, where n counts the output
    trajectory and extension is either set by the user, or else determined from the input.
    Example: When the input is in dcd format, and indexes is a list of length 3, the output will
    by default go to files "set_1.dcd", "set_2.dcd", "set_3.dcd". If you want files to be stored
    in a specific subfolder, simply specify the relative path in the prefix, e.g. prefix='~/macrostates/\pcca_'

    Parameters
    ----------
    traj_inp : :py:class:`pyemma.coordinates.data.feature_reader.FeatureReader`
        A data source as provided by Please use :py:func:`pyemma.coordinates.source` to construct it.

    indexes : list of ndarray(T_i, 2)
        A list of N arrays, each of size (T_n x 2) for writing N trajectories of T_i time steps.
        Each row contains two indexes (i, t), where i is the index of the trajectory from the input
        and t is the index of the time step within the trajectory.

    prefix : str, optional, default = `set_`
        output filename prefix. Can include an absolute or relative path name.

    fmt : str, optional, default = None
        Outpuf file format. By default, the file extension and format. It will be determined from the input. If a
        different format is desired, specify the corresponding file extension here without a dot, e.g. "dcd" or "xtc".

    outfiles : list of str, optional, default = None
        A list of output filenames. When given, this will override the settings of prefix and fmt, and output
        will be written to these files.

    inmemory : Boolean, default = False (untested for large files)
        Instead of internally calling traj_save for every (T_i,2) array in "indexes", only one call is made. Internally,
        this generates a potentially large molecular trajectory object in memory that is subsequently sliced into the
        files of "outfiles". Should be faster for large "indexes" arrays and large files, though it is quite memory
        intensive. The optimal situation is to avoid streaming two times through a huge file for "indexes" of type:
        indexes = [[1 4000000],[1 4000001]]

    stride  : integer, default is 1
        This parameter informs :py:func:`save_trajs` about the stride used in the indexes variable. Typically, the variable
        indexes contains frame-indexes that match exactly the frames of the files contained in traj_inp.trajfiles.
        However, in certain situations, that might not be the case. Examples of these situations are cases in
        which stride value != 1 was used when reading/featurizing/transforming/discretizing the files contained in
        traj_inp.trajfiles.

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

        _, fmt = os.path.splitext(traj_inp.trajfiles[0])
    else:
        fmt = '.' + fmt

    # Prepare the list of outfiles before the loop
    if outfiles is None:
        outfiles = []
        for ii in xrange(len(indexes)):
            outfiles.append(prefix + '%06u' % ii + fmt)

    # Check that we have the same name of outfiles as (T, 2)-indexes arrays
    if len(indexes) != len(outfiles):
        raise Exception('len(indexes) (%s) does not match len(outfiles) (%s)' % (len(indexes), len(outfiles)))

    # This implementation looks for "i_indexes" separately, and thus one traj_inp.trajfile 
    # might be accessed more than once (less memory intensive)
    if not inmemory:
        for i_indexes, outfile in _itertools.izip(indexes, outfiles):
            # TODO: use **kwargs to parse to save_traj
            save_traj(traj_inp, i_indexes, outfile, stride = stride, verbose=verbose)

    # This implementation is "one file - one pass" but might temporally create huge memory objects
    else:
        traj = save_traj(traj_inp, indexes, outfile=None, stride = stride, verbose=verbose)
        i_idx = 0
        for i_indexes, outfile in _itertools.izip(indexes, outfiles):
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
    if isinstance(previous_stage, _Transformer):
        inputstage = previous_stage
    # second option: data is array or list of arrays
    else:
        data = _types.ensure_traj_list(previous_stage)
        inputstage = _DataInMemory(data)

    return inputstage

def _param_stage(previous_stage, this_stage, stride=1):
    r""" Parametrizes the given pipelining stage if a valid source is given.

    Parameters
    ----------
    source : one of the following: None, Transformer (subclass), ndarray, list of ndarrays
        data source from which this transformer will be parametrized. If None,
        there is no input data and the stage will be returned without any other action.
    stage : the transformer object to be parametrized given the source input.

    """
    # no input given - nothing to do
    if previous_stage is None:
        return this_stage

    inputstage = _get_input_stage(previous_stage)
    # parametrize transformer
    this_stage.data_producer = inputstage
    this_stage.chunksize = inputstage.chunksize
    this_stage.parametrize(stride=stride)
    return this_stage


def pca(data=None, dim=2, var_cutoff=1.0, stride=1, mean=None):
    r""" Principal Component Analysis (PCA).

    PCA is a linear transformation method that finds coordinates of maximal variance.
    A linear projection onto the principal components thus makes a minimal error in terms
    of variation in the data. Note, however, that this method is not optimal
    for Markov model construction because for that purpose the main objective is to
    preserve the slow processes which can sometimes be associated with small variance.

    It estimates a PCA transformation from data. When input data is given as an
    argument, the estimation will be carried out right away, and the resulting
    object can be used to obtain eigenvalues, eigenvectors or project input data
    onto the principal components. If data is not given, this object is an
    empty estimator and can be put into a :func:`pipeline` in order to use PCA
    in streaming mode.

    Parameters
    ----------

    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by source function
        data array or list of data arrays. T or T_i are the number of time steps in a
        trajectory. When data is given, the PCA is immediately parametrized by estimating
        the covariance matrix and computing its eigenvectors.

    dim : int, optional, default -1
        the number of dimensions (principal components) to project onto. A call to the
        :func:`map <pyemma.coordinates.transform.PCA.map>` function reduces the d-dimensional
        input to only dim dimensions such that the data preserves the maximum possible variance
        amongst dim-dimensional linear projections.
        -1 means all numerically available dimensions will be used unless reduced by var_cutoff.
        Setting dim to a positive value is exclusive with var_cutoff.

    var_cutoff : float in the range [0,1], optional, default 1
        Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
        exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
        (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this could cause this calculation
        to be very slow for large data sets. Since molecular dynamics data is usually
        correlated at short timescales, it is often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the returned object is independent, so
        you can parametrize at a long stride, and still map all frames through the transformer.

    mean : ndarray, optional, default None
        Optionally pass pre-calculated means to avoid their re-computation.
        The shape has to match the input dimension.

    Returns
    -------
    pca : a :class:`PCA<pyemma.coordinates.transform.PCA>` transformation object
        Object for Principle component analysis (PCA) analysis.
        It contains PCA eigenvalues and eigenvectors, and the projection of input data to the dominant PCA


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


    .. autoclass:: pyemma.coordinates.transform.pca.PCA
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.transform.pca.PCA
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.transform.pca.PCA
            :attributes:

    See also
    --------
    :class:`PCA <pyemma.coordinates.transform.PCA>` : pca object

    :func:`tica <pyemma.coordinates.tica>` : for time-lagged independent component analysis

    References
    ----------
    .. [1] Hotelling, H. 1933.
        Analysis of a complex of statistical variables into principal components.
        J. Edu. Psych. 24, 417-441 and 498-520.

    """
    if mean is not None:
        data = _get_input_stage(data)
        indim = data.dimension()
        mean = _types.ensure_ndarray(mean, shape=(indim,), dtype=_np.float)

    res = _PCA(dim=dim, var_cutoff=var_cutoff)
    return _param_stage(data, res, stride=stride)


def tica(data=None, lag=10, dim=-1, var_cutoff=1.0, kinetic_map=False, stride=1,
         force_eigenvalues_le_one=False, mean=None):
    r""" Time-lagged independent component analysis (TICA).

    TICA is a linear transformation method. In contrast to PCA, which finds
    coordinates of maximal variance, TICA finds coordinates of maximal autocorrelation
    at the given lag time. Therefore, TICA is useful in order to find the *slow* components
    in a dataset and thus an excellent choice to transform molecular dynamics
    data before clustering data for the construction of a Markov model.
    When the input data is the result of a Markov process (such as thermostatted
    molecular dynamics), TICA finds in fact an approximation to the eigenfunctions and
    eigenvalues of the underlying Markov operator [1]_.

    It estimates a TICA transformation from *data*. When input data is given as an
    argument, the estimation will be carried out straight away, and the resulting
    object can be used to obtain eigenvalues, eigenvectors or project input data
    onto the slowest TICA components. If no data is given, this object is an
    empty estimator and can be put into a :func:`pipeline` in order to use TICA
    in the streaming mode.

    Parameters
    ----------
    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by source function
        array with the data, if available. When given, the TICA transformation
        is immediately computed and can be used to transform data.

    lag : int, optional, default = 10
        the lag time, in multiples of the input time step

    dim : int, optional, default -1
        the number of dimensions (independent components) to project onto. A call to the
        :func:`map <pyemma.coordinates.transform.TICA.map>` function reduces the d-dimensional
        input to only dim dimensions such that the data preserves the maximum possible autocorrelation
        amongst dim-dimensional linear projections.
        -1 means all numerically available dimensions will be used unless reduced by var_cutoff.
        Setting dim to a positive value is exclusive with var_cutoff.

    var_cutoff : float in the range [0,1], optional, default 1
        Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
        exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
        (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim

    kinetic_map : bool, optional, default False
        Eigenvectors will be scaled by eigenvalues. As a result, Euclidean distances in the transformed data
        approximate kinetic distances [4]_. This is a good choice when the data is further processed by clustering.

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this could cause this calculation
        to be very slow for large data sets. Since molecular dynamics data is usually
        correlated at short timescales, it is often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the returned object is independent, so
        you can parametrize at a long stride, and still map all frames through the transformer.

    force_eigenvalues_le_one : boolean
        Compute covariance matrix and time-lagged covariance matrix such
        that the generalized eigenvalues are always guaranteed to be <= 1.

    mean : ndarray, optional, default None
        Optionally pass pre-calculated means to avoid their re-computation.
        The shape has to match the input dimension.


    Returns
    -------
    tica : a :class:`TICA <pyemma.coordinates.transform.TICA>` transformation  object
        Object for time-lagged independent component (TICA) analysis.
        it contains TICA eigenvalues and eigenvectors, and the projection of input data to the dominant TICA


    Notes
    -----
    Given a sequence of multivariate data :math:`X_t`, it computes the mean-free
    covariance and time-lagged covariance matrix:

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

    TICA was originally introduced for signal processing in [2]_. It was introduced
    to molecular dynamics and as a method for the construction of Markov models in
    [1]_ and [3]_. It was shown in [1]_ that when applied to molecular dynamics data,
    TICA is an approximation to the eigenvalues and eigenvectors of the true underlying
    dynamics.

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


    .. autoclass:: pyemma.coordinates.transform.tica.TICA
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.transform.tica.TICA
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.transform.tica.TICA
            :attributes:

    See also
    --------
    :class:`TICA <pyemma.coordinates.transform.TICA>` : tica object

    :func:`pca <pyemma.coordinates.pca>` : for principal component analysis

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
    if mean is not None:
        data = _get_input_stage(data)
        indim = data.dimension()
        mean = _types.ensure_ndarray(mean, shape=(indim,), dtype=_np.float)
    res = _TICA(lag, dim=dim, var_cutoff=var_cutoff, kinetic_map=kinetic_map,
                force_eigenvalues_le_one=force_eigenvalues_le_one, mean=mean)
    return _param_stage(data, res, stride=stride)


# =========================================================================
#
# CLUSTERING ALGORITHMS
#
# =========================================================================

@deprecated("Please use pyemma.coordinates.cluster_kmeans()")
def kmeans(data=None, k=100, max_iter=1000, stride=1):
    return cluster_kmeans(data, k, max_iter, stride=stride)


def cluster_mini_batch_kmeans(data=None, k=100, max_iter=10, batch_size=0.2, metric='euclidean', init_strategy='kmeans++'):
    res = _MiniBatchKmeansClustering(n_clusters=k, max_iter=max_iter, metric=metric, init_strategy=init_strategy, batch_size=batch_size)
    return _param_stage(data, res, stride=1)


def cluster_kmeans(data=None, k=100, max_iter=10, stride=1, metric='euclidean', init_strategy='kmeans++'):
    r"""k-means clustering

    If data is given, it performs a k-means clustering and then assigns the data using a Voronoi discretization.
    It returns a :class:`KmeansClustering <pyemma.coordinates.clustering.KmeansClustering>` object
    that can be used to extract the discretized
    data sequences, or to assign other data points to the same partition. If data is not given, an
    empty :class:`KmeansClustering <pyemma.coordinates.clustering.KmeansClustering>` will be created that
    still needs to be parametrized, e.g. in a :func:`pipeline`.

    .. seealso:: **Theoretical background**: `Wiki page <http://en.wikipedia.org/wiki/K-means_clustering>`_

    Parameters
    ----------
    data: ndarray (T, d) or list of ndarray (T_i, d) or a reader created by source function
        input data, if available in memory

    k: int
        the number of cluster centers

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this could cause this calculation
        to be very slow for large data sets. Since molecular dynamics data is usually
        correlated at short timescales, it is often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the returned object is independent, so
        you can parametrize at a long stride, and still map all frames through the transformer.

    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')
    init_strategy : str
        determines if the initial cluster centers are chosen according to the kmeans++-algorithm
        or uniformly distributed

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


    .. autoclass:: pyemma.coordinates.clustering.kmeans.KmeansClustering
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.coordinates.clustering.kmeans.KmeansClustering
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.coordinates.clustering.kmeans.KmeansClustering
            :attributes:

    """
    res = _KmeansClustering(n_clusters=k, max_iter=max_iter, metric=metric, init_strategy=init_strategy)
    return _param_stage(data, res, stride=stride)


@deprecated("Please use pyemma.coordinates.cluster_uniform_time()")
def uniform_time(data=None, k=100, stride=1):
    return cluster_uniform_time(data, k, stride=stride)


def cluster_uniform_time(data=None, k=100, stride=1, metric='euclidean'):
    r"""Uniform time clustering

    If given data, performs a clustering that selects data points uniformly in time and then assigns the data
    using a Voronoi discretization. Returns a
    :class:`UniformTimeClustering <pyemma.coordinates.clustering.UniformTimeClustering>` object
    that can be used to extract the discretized data sequences, or to assign other data points to the same partition.
    If data is not given, an empty
    :class:`UniformTimeClustering <pyemma.coordinates.clustering.UniformTimeClustering>` will be created that
    still needs to be parametrized, e.g. in a :func:`pipeline`.

    Parameters
    ----------
    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by source function
        input data, if available in memory

    k : int
        the number of cluster centers

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this could cause this calculation
        to be very slow for large data sets. Since molecular dynamics data is usually
        correlated at short timescales, it is often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the returned object is independent, so
        you can parametrize at a long stride, and still map all frames through the transformer.

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
    res = _UniformTimeClustering(k, metric=metric)
    return _param_stage(data, res)


@deprecated("Please use pyemma.coordinates.cluster_regspace()")
def regspace(data=None, dmin=-1, max_centers=1000, stride=1):
    return cluster_regspace(data, dmin, max_centers, stride=stride)


def cluster_regspace(data=None, dmin=-1, max_centers=1000, stride=1, metric='euclidean'):
    r"""Regular space clustering

    If given data, it performs a regular space clustering [1]_ and returns a
    :class:`RegularSpaceClustering <pyemma.coordinates.clustering.RegularSpaceClustering>` object that
    can be used to extract the discretized data sequences, or to assign other data points to the same partition.
    If data is not given, an empty
    :class:`RegularSpaceClustering <pyemma.coordinates.clustering.RegularSpaceClustering>` will be created
    that still needs to be parametrized, e.g. in a :func:`pipeline`.

    Regular space clustering is very similar to Hartigan's leader algorithm [2]_. It consists of two passes through
    the data. Initially, the first data point is added to the list of centers. For every subsequent data point, if
    it has a greater distance than dmin from every center, it also becomes a center. In the second pass, a Voronoi
    discretization with the computed centers is used to partition the data.

    Parameters
    ----------
    data : ndarray (T, d) or list of ndarray (T_i, d) or a reader created by source function
        input data, if available in memory

    dmin : float
        the minimal distance between cluster centers

    max_centers : int (optional), default=1000
        If max_centers is reached, the algorithm will stop to find more centers,
        but it is possible that parts of the state space are not properly discretized. This will generate a
        warning. If that happens, it is suggested to increase dmin such that the number of centers stays below
        max_centers.

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this could cause this calculation
        to be very slow for large data sets. Since molecular dynamics data is usually
        correlated at short timescales, it is often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the returned object is independent, so
        you can parametrize at a long stride, and still map all frames through the transformer.

    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')

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
    res = _RegularSpaceClustering(dmin, max_centers, metric=metric)
    return _param_stage(data, res, stride=stride)


@deprecated("Please use pyemma.coordinates.assign_to_centers()")
def assign_centers(data=None, centers=None, stride=1):
    return assign_to_centers(data, centers, stride=stride)


def assign_to_centers(data=None, centers=None, stride=1, return_dtrajs=True,
                      metric='euclidean'):
    r"""Assigns data to the nearest cluster centers

    Creates a Voronoi partition with the given cluster centers. If given trajectories as data, this function
    will by default discretize the trajectories and return discrete trajectories of corresponding lengths.
    Otherwise, an assignment object will be returned that can be used to assign data later or can serve
    as a pipeline stage.

    Parameters
    ----------
    data : ndarray or list of arrays or reader created by source function
        data to be assigned

    centers : path to file or ndarray or a reader created by source function
        cluster centers to use in assignment of data

    stride : int, optional, default = 1
        If set to 1, all input data will be used for estimation. Note that this could cause this calculation
        to be very slow for large data sets. Since molecular dynamics data is usually
        correlated at short timescales, it is often sufficient to estimate transformations at a longer stride.
        Note that the stride option in the get_output() function of the returned object is independent, so
        you can parametrize at a long stride, and still map all frames through the transformer.

    return_dtrajs : bool, optional, default = True
        If True, it will return the discretized trajectories obtained from assigning the coordinates in the data
        input. This will only have effect if data is given. When data is not given or return_dtrajs is False,
        the :class:'AssignCenters <_AssignCenters>' object will be returned.

    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')


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
    >>> print dtrajs # doctest: +ELLIPSIS
    [array([...

    """
    if centers is None:
        raise ValueError('You have to provide centers in form of a filename'
                         ' or NumPy array or a reader created by source function')
    res = _AssignCenters(centers, metric=metric)
    parametrized_stage = _param_stage(data, res, stride=stride)
    if return_dtrajs and data is not None:
        return parametrized_stage.dtrajs

    return parametrized_stage

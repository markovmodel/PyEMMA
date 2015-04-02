from pyemma.coordinates.io import FeatureReader as _FeatureReader


def get_file_reader(input, topology, featurizer):
    if isinstance(input, basestring) or (
        isinstance(input, (list, tuple)) and (any(isinstance(item, basestring) for item in input) or len(input) is 0)
    ):
        reader = None
        # check: if single string create a one-element list
        if isinstance(input, basestring):
            input_list = [input]
        elif len(input) > 0 and all(isinstance(item, basestring) for item in input):
            input_list = input
        else:
            if len(input) is 0:
                raise ValueError("The passed input list should not be empty.")
            else:
                raise ValueError("The passed list did not exclusively contain strings.")

        try:
            idx = input_list[0].rindex(".")
            suffix = input_list[0][idx:]
        except ValueError as ex:
            raise ValueError("The specified files %s had no file extension!" % input_list, ex)

        # check: do all files have the same file type? If not: raise ValueError.
        if all(item.endswith(suffix) for item in input_list):
            from mdtraj.formats.registry import _FormatRegistry

            # CASE 1.1: file types are MD files
            if suffix in _FormatRegistry.loaders.keys():
                # check: do we either have a featurizer or a topology file name? If not: raise ValueError.
                # create a MD reader with file names and topology
                if not featurizer and not topology:
                    raise ValueError("The input files were MD files which makes it mandatory to have either a "
                                     "featurizer or a topology file.")
                if not topology:
                    # we have a featurizer
                    reader = _FeatureReader.init_from_featurizer(input_list, featurizer)
                else:
                    # we have a topology file
                    reader = _FeatureReader(input_list, topology)
            else:
                # TODO: CASE 1.2: file types are raw data files
                # TODO: create raw data reader from file names
                pass
        else:
            raise ValueError("Not all elements in the input list were of the type %s!" % suffix)
    else:
        raise ValueError("Input \"%s\" was no string or list of strings." % input)
    return reader

# This file is part of PyEMMA.
#
# Copyright (c) 2014-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import sys


def main(argv=None):
    import argparse
    from pyemma import load
    from pyemma._base.serialization.h5file import H5File

    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true', default=False)
    parser.add_argument('files', metavar='files', nargs='+', help='files to inspect')
    parser.add_argument('--recursive', action='store_true', default=False,
                        help='If the pipeline of the stored estimator was stored, '
                             'gather these information as well. This will require to load the model, '
                             'so it could take a while, if the pipeline contains lots of data.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args(argv)
    # store found models by filename
    from collections import defaultdict
    models = defaultdict(dict)

    for f in args.files:
        try:
            with H5File(f) as fh:
                m = fh.models_descriptive
            for k in m:
                models[f][k] = m[k]
            for model_name, values in m.items():
                if values['saved_streaming_chain']:
                    restored = load(f)
                    models[f][model_name]['input_chain'] = [repr(x) for x in restored._data_flow_chain()]
        except BaseException as e:
            print('{} did not contain a valid PyEMMA model. Error was {err}. '
                  'If you are sure, that it does, please post an issue on Github'.format(f, err=e))
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    if not args.json:
        from io import StringIO

        buff = StringIO()
        buff.write('PyEMMA models\n')
        buff.write('=' * (buff.tell() - 1))
        buff.write('\n' * 2)
        for f in models:
            buff.write('file: {}'.format(f))
            buff.write('\n')
            buff.write('-' * 80)
            buff.write('\n')
            model_file = models[f]
            for i, model_name in enumerate(model_file):
                attrs = model_file[model_name]
                buff.write('{index}. name: {key}\n'
                           'created: {created}\n'
                           '{repr}\n'.format(key=model_name, index=i+1,
                                             created=attrs['created_readable'],
                                             repr=attrs['class_str']))
                if attrs['saved_streaming_chain']:
                    buff.write('\n---------Input chain---------\n')
                    for j, x in enumerate(attrs['input_chain']):
                        buff.write('{index}. {repr}\n'.format(index=j+1, repr=x))
            buff.write('-' * 80)
            buff.write('\n')
        buff.seek(0)
        print(buff.read())
    else:
        import json
        json.dump(models, fp=sys.stdout)
    return 0

if __name__ == '__main__':
    sys.exit(main())

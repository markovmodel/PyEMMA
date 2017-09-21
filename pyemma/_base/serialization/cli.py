import sys


def main(argv=None):
    import argparse
    from pyemma._base.serialization.serialization import list_models
    from pyemma import load

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
            m = list_models(f)
            for k in m:
                models[f][k] = m[k]
            for model_name, values in m.items():
                if 'saved_streaming_chain' in values:
                    restored = load(f)
                    models[f][model_name]['input_chain'] = [repr(x) for x in restored._data_flow_chain()]
        except BaseException as e:
            print('{} did not contain a valid PyEMMA model. Error was {err}. '
                  'If you are sure, that it does, please post an issue on Github'.format(f, err=e), file=sys.stderr)
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
                                             created=attrs['created'],
                                             repr=attrs['repr']))
                if 'saved_streaming_chain' in attrs:
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

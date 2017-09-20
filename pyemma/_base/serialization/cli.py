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
                             'gather these information as well.')
    args = parser.parse_args(argv)
    models = []

    for f in args.files:
        try:
            m = list_models(f)
            if m['saved_streaming_chain']:
                restored = load(f)
                m['input_chain'] = [repr(x) for x in restored._data_flow_chain()]
            models.append((f, m))
        except KeyError:
            print('{} did not contain a valid PyEMMA model. '
                  'If you are sure, that it does, please post an issue on Github'.format(f), file=sys.stderr)
            return 1

    if not args.json:
        from io import StringIO

        buff = StringIO()
        buff.write('PyEMMA models\n')
        buff.write('=' * (buff.tell() - 1))
        buff.write('\n' * 2)
        # buff.write('file\t\t\tmodel\n')
        for f, model_file in models:
            buff.write('file: {}'.format(f))
            buff.write('\n')
            buff.write('-' * 80)
            buff.write('\n')
            for i, m in enumerate(model_file):
                buff.write('{index}. name: {key}\n'
                           'created: {created}\n'
                           '{repr}\n'.format(key=m, index=i+1,
                                             created=model_file[m]['created'],
                                             repr=model_file[m]['repr']))
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

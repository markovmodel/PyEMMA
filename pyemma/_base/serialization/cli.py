

def main():
    import argparse
    import sys
    from pyemma._base.serialization.serialization import list_models

    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true', default=False)
    parser.add_argument('files', metavar='files', nargs='+', help='files to inspect')
    args = parser.parse_args()

    models = []

    for f in args.files:
        try:
            models.append((f, list_models(f)))
        except KeyError:
            print('{} did not contain a valid PyEMMA model. '
                  'If you are sure, that it does, please post an issue on Github'.format(f), file=sys.stderr)

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


if __name__ == '__main__':
    main()

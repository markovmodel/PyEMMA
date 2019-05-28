from nbconvert.preprocessors import Preprocessor


class RemoveWidgetNotice(Preprocessor):
    # in ipywidgets 7, the state of a closed widget (progress bars) is saved to the ipynb file, generating a stupid
    # message like 'If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean\n",
    #   "  that the widgets JavaScript is still loading.'
    # If this preprocessor finds such a thing in the output, we just clear it.

    def preprocess_cell(self, cell, resources, index):
        if 'outputs' in cell:
            outputs_ = [o for o in cell['outputs']  # list of dicts
                        if (('data' in o and 'application/vnd.jupyter.widget-view+json' not in o['data'])
                            or 'data' not in o)
                       ]
            cell['outputs'] = outputs_

        return cell, resources


class RemoveSolutionStubs(Preprocessor):
    """For rendering executed versions of the notebooks, we do not want to have the solution stubs."""
    enabled = True  # enable by default, because we use it as a default preprocessor, which should be executed prior ExecutePreprocessor.

    def preprocess(self, nb, resources):
        filtered_cells = [
            cell for cell in nb['cells']
            if not cell['metadata'].get('solution2_first', False)
        ]
        nb['cells'] = filtered_cells
        return nb, resources


class RewriteNotebookLinks(Preprocessor):

    def preprocess_cell(self, cell, resources, index):
        new_input = cell.source.replace('.ipynb', '.html')
        cell.source = new_input
        return cell, resources

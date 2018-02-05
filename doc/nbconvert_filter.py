from nbconvert.preprocessors import Preprocessor


class RemoveWidgetNotice(Preprocessor):
    # in ipywidgets 7, the state of a closed widget (progress bars) is saved to the ipynb file, generating a stupid
    # message like 'If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean\n",
    #   "  that the widgets JavaScript is still loading.'
    # If this preprocessor finds such a thing in the output, we just clear it.

    def preprocess_cell(self, cell, resources, index):
        """
        Override if you want to apply some preprocessing to each cell.
        Must return modified cell and resource dictionary.

        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        index : int
            Index of the cell being processed
        """
        if 'outputs' in cell:
            outputs = cell['outputs']  # list
            to_delete = []
            for i, o in enumerate(outputs):
                #print(o)
                if 'data' in o:
                    data = o['data']
                    if 'application/vnd.jupyter.widget-view+json' in data:
                        to_delete.append(o)
            for o in to_delete:
                #print('removing: ', o)
                outputs.remove(o)

        return cell, resources

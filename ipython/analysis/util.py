import numpy as np

class MapToConnectedStateLabels():
    def __init__(self, lcc):
        self.lcc=lcc
        self.new_labels=np.arange(len(self.lcc))
        self.dictmap=dict(zip(self.lcc, self.new_labels))
    
    def map(self, A):
        """Map subset of microstates to subset of connected
        microstates.

        Parameters
        ----------
        A : list of int
            Subset of microstate labels

        Returns
        -------
        A_cc : list of int
            Corresponding subset of mircrostate labels
            in largest connected set lcc. 

        """
        if not set(A).issubset(set(self.lcc)):
            raise ValueError("A is not a subset of the set of "+\
                                 "completely connected states.")
        else:
            return [self.dictmap[i] for i in A]

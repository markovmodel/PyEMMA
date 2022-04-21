from deeptime.markov import PCCAModel, pcca

from pyemma._base.serialization.serialization import SerializableMixIn


class PCCA(PCCAModel, SerializableMixIn):
    __serialize_version = 1

    """
    PCCA+ spectral clustering method with optimized memberships [1]_
    Clusters the first m eigenvectors of a transition matrix in order to cluster the states.
    This function does not assume that the transition matrix is fully connected. Disconnected sets
    will automatically define the first metastable states, with perfect membership assignments.

    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix.
    m : int
        Number of clusters to group to.

    References
    ----------
    [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
        application to Markov state models and data classification.
        Adv Data Anal Classif 7, 147-179 (2013).
    [2] F. Noe, multiset PCCA and HMMs, in preparation.
    [3] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
        Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
        J. Chem. Phys. 139, 184114 (2013)
    """
    def __init__(self, P, m):
        dt_pcca = pcca(P, m)
        super(PCCA, self).__init__(transition_matrix_coarse=dt_pcca.coarse_grained_transition_matrix,
                                   pi_coarse=dt_pcca.coarse_grained_stationary_probability,
                                   memberships=dt_pcca.memberships,
                                   metastable_distributions=dt_pcca.metastable_distributions)
        self.P = P

    @property
    def transition_matrix(self):
        return self.P

    @property
    def stationary_probability(self):
        return self.coarse_grained_stationary_probability

    @property
    def output_probabilities(self):
        return self.metastable_distributions

    @property
    def metastable_sets(self):
        return self.sets

    @property
    def metastable_assignment(self):
        return self.assignments

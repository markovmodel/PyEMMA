import numpy as np
import os
import mdtraj as md

__all__ = ['regroup_RAM','regroup_DISK','PCCA_disctrajs']

def regroup_RAM(trajs,disctrajs):
    '''Regroups MD trajectories into clusters according to discretised trajectories.
    
       Parameters
       ----------
       trajs : list of `mdtraj.Trajectory`s
       disctrajs : list of array-likes
               len(disctrajs[i])==trajs[i].n_frames for all i
    
       Returns
       -------
       cluster : list of `mdtraj.Trajectory`s or `None`, len(cluster)=np.max(trajs)+1
           each element cluster[i] is either `None` if i wasn't found in disctrajs or
           is a new trajectory that holds all frames that were assigned to cluster i.
           
        Note
        ----
        This function is RAM intensive.
    '''

    # handle single element invocation
    if not isinstance(trajs,list):
        trajs = [trajs]
    if not isinstance(disctrajs,list):
        disctrajs = [disctrajs]
        
    assert len(disctrajs)==len(trajs), 'Number of disctrajs and number of trajs doesn\'t agree.'
    states = np.unique(np.hstack(([np.unique(disctraj) for disctraj in disctrajs]))) 
    states = np.setdiff1d(states,[-1]) # exclude invalid states
    cluster = [None]*(np.max(states)+1) 
    for disctraj,traj,i in zip(disctrajs,trajs,xrange(len(trajs))):
        assert len(disctraj)==traj.n_frames, 'Length of disctraj[%d] doesn\'t match number of frames in traj[%d].'%(i,i)
        for s in states:
            match = (disctraj==s)
            if np.count_nonzero(match)>0:
                if cluster[s] is None:
                    cluster[s] = traj.xyz[match,:,:]
                else:
                    cluster[s] = np.concatenate((cluster[s],traj.xyz[match,:,:]),axis=0)
    for i in xrange(len(cluster)):
        if not cluster[i] is None:
            cluster[i] = md.Trajectory(cluster[i],trajs[0].topology)
    return cluster

def regroup_DISK(trajs,topology_file,disctrajs,path,stride=1):
    '''Regroups MD trajectories into clusters according to discretised trajectories.
    
       Parameters
       ----------
       trajs : list of strings 
           xtc/dcd/... trajectory file names
       topology_file : string
           name of topology file that matches `trajs`
       disctrajs : list of array-likes
           discretized trajectories
       path : string
           file system path to directory where cluster trajectories are written
       stride : int
           stride of disctrajs with respect to the (original) trajs
            
       Returns
       -------
       cluster : list of file names or `None`, len(cluster)=np.max(trajs)+1
           each element cluster[i] is either `None` if i wasn't found in disctrajs or
           is a the file name of a new trajectory that holds all frames that were 
           assigned to cluster i.
    '''    
    # handle single element invocation
    if not isinstance(trajs,list):
        trajs = [trajs]
    if not isinstance(disctrajs,list):
        disctrajs = [disctrajs]

    states = np.unique(np.hstack(([np.unique(disctraj) for disctraj in disctrajs])))
    states = np.setdiff1d(states,[-1]) # exclude invalid states
    writer = [None]*(max(states)+1)
    cluster = [None]*(max(states)+1)

    for i in states:
        cluster[i] = path+os.sep+('%d.xtc'%i)
        writer[i] = md.formats.XTCTrajectoryFile(cluster[i],'w',force_overwrite=True)

    for disctraj,traj in zip(disctrajs,trajs):
        reader = md.iterload(traj,top=topology_file,stride=stride)
        start = 0
        for chunk in reader:
            chunk_length = chunk.xyz.shape[0]
            for i in xrange(chunk_length):
                cl = disctraj[i+start]
                if cl!=-1:
                    writer[cl].write(chunk.xyz[i,:,:]) # np.newaxis?
            start += chunk_length
         # TODO: check that whole disctrajs was used
    for i in states:
        writer[i].close()
        
    return cluster 
    
   
def PCCA_disctrajs(disctrajs,connected_set,memberships):
    '''Compute disctrajs coarse-grained to the PCCA sets.
   
       Parameters
       ----------
       disctrajs : list of array-likes
           discretzed trajectories
       connected_set : (N) ndarray 
           connected set as returned by `pyemma.msm.estimation.largest_connected_set`
       memberships : (N,M) ndarray
           PCCA memberships as returned by `pyemma.msm.analysis.pcca`
          
       Returns
       -------
       cgdisctraj : list of array likes in the same shape as parameter `disctrajs`
       If disctaj[i][j] was assigned to PCCA set k, then cgdisctraj[i][j]==k.
       If disctaj[i][j] isn't in the connected set, cgdisctraj[i][j]==maxint.
    '''

    if not isinstance(disctrajs,list):
        disctrajs = [disctrajs]

    assert connected_set.ndim == 1
    assert connected_set.shape[0] == memberships.shape[0]

    # compute the forward map : old index -> new index
    backward_map = connected_set # map : new index -> old index
    n_states = np.max(disctrajs)+1
    forward_map = np.ones(n_states,dtype=int)*(-1)
    forward_map[backward_map] = np.arange(backward_map.shape[0]) # forward(backward)=Identity
    pcca_map = np.hstack((np.argmax(memberships,axis=1),[-1]))
    return [ pcca_map[forward_map[d]] for d in disctrajs ]        


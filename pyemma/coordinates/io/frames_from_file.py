import mdtraj as md
import numpy as np
from logging import info

__all__ =['frames_from_file']
def frames_from_file(file_name, pdbfile, frames, chunksize=int(1e5), verbose=False):

    traj = None
    cum_frames = 0

    # Because the trajectory is streamed "chronologically", but "frames" can have any arbitrary order
    # we store that order in "orig_order" to reshuffle the traj at the end
    orig_order = frames.argsort().argsort()

    for jj, traj_chunk in enumerate(md.iterload(file_name, top=pdbfile, chunk=chunksize)):

        # Create an indexing array for this trajchunk
        i_idx=jj*chunksize
        f_idx=i_idx+chunksize-1
        chunk_frames=np.arange(i_idx,f_idx+1)[:traj_chunk.n_frames]

        # Frames that appear more than one time will be kept
        good_frames = np.hstack([np.argwhere(ff == chunk_frames).squeeze() for ff in np.sort(frames)])

        # Append the good frames of this chunk
        if np.size(good_frames)>0:
           if traj is None:
               traj=traj_chunk[good_frames]
           else:
               traj=traj.join(traj_chunk[good_frames])

        cum_frames += np.size(good_frames)

        if verbose:
            info('chunk %u of traj has size %u, indices %6u...%6u. Accumulated frames %u'%(jj, traj_chunk.n_frames,chunk_frames[0], chunk_frames[-1], cum_frames))

        # Check if we can already stop iterating
        if chunk_frames[-1] >= frames.max():
            break

    # Make sure that "frames" did not contain impossible frames
    if (frames > chunk_frames[-1]).any():
        raise Exception('Cannot provide frames %s for trajectory %s with n_frames = %u'%(frames[frames > chunk_frames[-1]], file_name, chunk_frames[-1]))

    # Trajectory coordinates are is returned "reshuffled"
    return traj[orig_order]


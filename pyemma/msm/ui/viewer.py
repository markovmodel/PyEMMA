import mdtraj as md
import IPython
from mdtraj.html import TrajectoryView, enable_notebook

import IPython
def view_traj(traj,topology_file=None, stride=1):
    '''Opens a trajectory viewer (from mdtraj).
    
    Parameters:
    -----------
    traj : `mdtraj.Trajectory` or string
        mdtraj.Trajectory object or file name for MD trajectory
    topology_file : string (default=None)
        If traj is a file name, topology_file is the file name
        of the accompanying topology file (.pdb/.mol2/...)
    stride : int
       If traj is a file name, this is the number of frames 
       to skip between two successive trajectory reads.
    '''
    if isinstance(traj,str):
        traj = md.load(traj,top=topology_file,stride=stride)
        
    widget=md.html.TrajectoryView(traj,colorBy='atom')
    IPython.display.display(widget)
    slider = IPython.html.widgets.IntSliderWidget(max=traj.n_frames-1)
    def on_value_change(name,val):
        widget.frame = val
    slider.on_trait_change(on_value_change, 'value')
    IPython.display.display(slider)
    None

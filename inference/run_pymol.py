# import pymol
# from pymol import cmd
#
# # Start PyMOL
# pymol.finish_launching()

# Load the topology file (PDB)
pdb_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_solver1/1a10A/gt_1a10A.pdb'
cmd.load(pdb_path, 'gt_1a10A')

# Load the trajectory file (XTC) into the existing topology
xtc_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_solver1/1a10A/output_trajectory.xtc'
cmd.load_traj(xtc_path, 'gt_1a10A')

# Center the view on the molecule
cmd.center("gt_1a10A", animate=-1)

# Perform intra_fit to align all states of the object to the first state
cmd.intra_fit("gt_1a10A")

# Align all states to the first state using the RMS command for all atoms
cmd.rms("gt_1a10A", "gt_1a10A and state 1")

# Center the view again
cmd.center('all')

# Zoom to fit the molecule in the view
cmd.zoom('all')

# Optional: Show the structure as cartoon representation
cmd.show('cartoon', 'gt_1a10A')

# Optional: Set the background color to white
# cmd.bg_color('white')

# Save the session if needed
cmd.save('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_solver1/1a10A/aligned_trajectory.pse')

# import pymol
# from pymol import cmd
#
# # Start PyMOL
# pymol.finish_launching()

# Load the topology file (PDB)
pdb_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_debug_forward/1a10A/1a10A/final_structure_no_water.pdb'
cmd.load(pdb_path, 'final_structure_no_water')

# Load the trajectory file (XTC) into the existing topology
xtc_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_debug_forward/1a10A/forward_trajectory.xtc'
cmd.load_traj(xtc_path, 'final_structure_no_water')

# Center the view on the molecule
cmd.center("final_structure_no_water", animate=-1)

# Perform intra_fit to align all states of the object to the first state
cmd.intra_fit("final_structure_no_water")

# Align all states to the first state using the RMS command for all atoms
cmd.rms("final_structure_no_water", "final_structure_no_water and state 1")

# Center the view again
cmd.center('all')

# Zoom to fit the molecule in the view
cmd.zoom('all')

# Optional: Show the structure as cartoon representation
cmd.show('cartoon', 'final_structure_no_water')

# Optional: Set the background color to white
# cmd.bg_color('white')

# Save the session if needed
cmd.save('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_debug_forward/1a10A/aligned_forward_trajectory.pse')

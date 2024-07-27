# import pymol
# from pymol import cmd
#
# # Start PyMOL
# pymol.finish_launching()

# Load the topology file (PDB)
pdb_path = (
    "/home/florian/Repos/small_molecule/data/alanine_final_structure_no_water.pdb"
)
cmd.load(pdb_path, "alanine")

# Load the trajectory file (XTC) into the existing topology
xtc_path = "/home/florian/Repos/small_molecule/results/output_trajectory.xtc"
cmd.load_traj(xtc_path, "alanine")

# Center the view on the molecule
cmd.center("alanine", animate=-1)

# Perform intra_fit to align all states of the object to the first state
cmd.intra_fit("alanine")

# Align all states to the first state using the RMS command for all atoms
cmd.rms("alanine", "alanine and state 1")

# Center the view again
cmd.center("all")

# Zoom to fit the molecule in the view
cmd.zoom("all")

# Optional: Show the structure as cartoon representation
cmd.show("cartoon", "alanine")

# Optional: Set the background color to white
# cmd.bg_color('white')

# Save the session if needed
cmd.save("/home/florian/Repos/small_molecule/results/aligned_trajectory.pse")

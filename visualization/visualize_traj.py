import os
import pymol
from pymol import cmd

# Start PyMOL
pymol.finish_launching()

# Define the base path
base_path = "/storage/florian/ziyu_project/ala2"

# Load the topology file (PDB)
pdb_path = os.path.join(base_path, "ala2.pdb")
cmd.load(pdb_path, "ala2")

# Load the trajectory file (DCD) into the existing topology
dcd_path = os.path.join(base_path, "trajectory-0.dcd")
cmd.load_traj(dcd_path, "ala2")

# Center the view on the molecule
cmd.center("ala2", animate=-1)

# Perform intra_fit to align all states of the object to the first state
cmd.intra_fit("ala2")

# Align all states to the first state using the RMS command for all atoms
cmd.rms("ala2", "ala2 and state 1")

# Center the view again
cmd.center("all")

# Zoom to fit the molecule in the view
cmd.zoom("all")

# Show the structure as cartoon representation
cmd.show("cartoon", "ala2")

# Set the background color to white (optional)
# md.bg_color("white")

# Set the stick radius for a thicker stick representation
cmd.set("stick_radius", 0.2, "ala2")

# Optional: Save the trajectory as a PDB file with multiple frames
# output_pdb_path = os.path.join(base_path, "output_trajectory_with_frames.pdb")
# cmd.save(output_pdb_path, "ala2", state=0)

print("Trajectory visualization enhanced and saved.")

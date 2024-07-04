import os
import subprocess
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from openmm.app import *
from openmm import *
from openmm.unit import *
import mdtraj as md
import numpy as np

# Step 1: Generate the molecule and initial structure using RDKit
smiles = "C(CO)O"  # Example: Ethylene Glycol
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)

# Print number of atoms in the initial molecule
print("Number of atoms in the initial molecule:", mol.GetNumAtoms())

# Save initial structure to PDB
pdb_filename = "molecule.pdb"
Chem.MolToPDBFile(mol, pdb_filename)

# Step 2: Use Antechamber to generate GAFF parameters
subprocess.run(["antechamber", "-i", "molecule.pdb", "-fi", "pdb", "-o", "molecule.mol2", "-fo", "mol2", "-c", "bcc", "-s", "2"], check=True)
subprocess.run(["parmchk2", "-i", "molecule.mol2", "-f", "mol2", "-o", "molecule.frcmod"], check=True)

# Step 3: Generate Amber input files for OpenMM including solvent using LEaP
leap_script = """
source leaprc.gaff
source leaprc.water.tip3p
loadamberparams molecule.frcmod
m = loadmol2 molecule.mol2
solvateBox m TIP3PBOX 12.0 iso
saveamberparm m molecule_solvated.prmtop molecule_solvated.inpcrd
savepdb m molecule_solvated.pdb
quit
"""
with open("leap.in", "w") as f:
    f.write(leap_script)

# Run tleap and capture output
result = subprocess.run(["tleap", "-f", "leap.in"], capture_output=True, text=True)
print("Standard Output from LEaP:")
print(result.stdout)
print("Standard Error from LEaP:")
print(result.stderr)

# Check if the files were created successfully
if not os.path.exists("molecule_solvated.prmtop") or not os.path.exists("molecule_solvated.inpcrd"):
    raise FileNotFoundError("Failed to generate prmtop or inpcrd files with LEaP.")

# Print the contents of the solvated PDB file for verification
with open("molecule_solvated.pdb", "r") as f:
    print("Contents of molecule_solvated.pdb:")
    print(f.read())

# Step 4: Set up and run molecular dynamics simulation using OpenMM
prmtop = AmberPrmtopFile('molecule_solvated.prmtop')
inpcrd = AmberInpcrdFile('molecule_solvated.inpcrd')

# Print number of atoms in the solvated system
print("Number of atoms in the solvated system:", prmtop.topology.getNumAtoms())

# Create OpenMM system
system = prmtop.createSystem(nonbondedMethod=PME,
                             nonbondedCutoff=1.0*nanometers, constraints=HBonds)

# Set periodic box vectors to ensure cubic box
if inpcrd.boxVectors:
    box_length = max(inpcrd.boxVectors[0][0], inpcrd.boxVectors[1][1], inpcrd.boxVectors[2][2]).value_in_unit(nanometer)
    box_vectors = [Vec3(box_length, 0, 0), Vec3(0, box_length, 0), Vec3(0, 0, box_length)]
    system.setDefaultPeriodicBoxVectors(*box_vectors)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.getPositions())

# Minimize energy
print("Minimizing energy...")
simulation.minimizeEnergy()

# Extended Equilibration
print("Equilibrating...")
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.step(20000)  # 20,000 steps for better equilibration

# Production run
total_steps = 500000  # Adjust this value based on your desired simulation length to ensure 10,000 frames
reporting_interval = total_steps // 10000  # To get 10,000 frames

traj_filename = "trajectory.dcd"
pdb_output = "output.pdb"
simulation.reporters.append(PDBReporter(pdb_output, reporting_interval))
simulation.reporters.append(DCDReporter(traj_filename, reporting_interval))
simulation.reporters.append(StateDataReporter(sys.stdout, reporting_interval, step=True,
                                              potentialEnergy=True, temperature=True))
print("Running production simulation...")
simulation.step(total_steps)

# Step 5: Convert DCD to XTC using MDTraj, align all frames, and then remove water
# Use the correct topology file (solvated system) for loading the trajectory
traj = md.load(traj_filename, top='molecule_solvated.pdb')

# Remove water molecules after alignment
traj_no_water = traj.remove_solvent()

# Align all frames to the first frame without water
traj_no_water.superpose(traj_no_water[0])

# Save the aligned trajectory without water
traj_no_water.save_xtc("trajectory_aligned_no_water.xtc")

# Convert the last frame to PDB for visualization without water
traj_no_water[-1].save_pdb("final_structure_no_water.pdb")

print("Simulation complete. Aligned PDB and XTC files without water have been generated.")

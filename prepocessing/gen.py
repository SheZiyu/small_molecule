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

# Function to run shell commands and capture output
def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"Running command: {' '.join(command)}")
    print(f"Standard Output:\n{result.stdout}")
    print(f"Standard Error:\n{result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Command {' '.join(command)} failed with return code {result.returncode}")
    return result

# Step 1: Generate the molecule and initial structure using RDKit
smiles = "CC(C(=O)O)N"  # SMILES for alanine
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)

# Print number of atoms in the initial molecule
print("Number of atoms in the initial molecule:", mol.GetNumAtoms())

# Save initial structure to PDB
pdb_filename = "alanine.pdb"
Chem.MolToPDBFile(mol, pdb_filename)

# Ensure there are no CONECT records in the PDB file
with open(pdb_filename, "r") as file:
    lines = file.readlines()
with open(pdb_filename, "w") as file:
    for line in lines:
        if not line.startswith("CONECT"):
            file.write(line)

# Step 2: Use Antechamber to generate GAFF parameters
run_command(["antechamber", "-i", "alanine.pdb", "-fi", "pdb", "-o", "alanine.mol2", "-fo", "mol2", "-c", "bcc", "-s", "2"])
run_command(["parmchk2", "-i", "alanine.mol2", "-f", "mol2", "-o", "alanine.frcmod"])

# Step 3: Generate Amber input files for OpenMM including solvent using LEaP
leap_script = """
source leaprc.gaff
source leaprc.water.tip3p
loadamberparams alanine.frcmod
m = loadmol2 alanine.mol2
solvateBox m TIP3PBOX 12.0 iso
saveamberparm m alanine_solvated.prmtop alanine_solvated.inpcrd
savepdb m alanine_solvated.pdb
quit
"""
with open("leap.in", "w") as f:
    f.write(leap_script)

# Run tleap and capture output
result = run_command(["tleap", "-f", "leap.in"])

# Check if the files were created successfully
if not os.path.exists("alanine_solvated.prmtop") or not os.path.exists("alanine_solvated.inpcrd"):
    raise FileNotFoundError("Failed to generate prmtop or inpcrd files with LEaP.")

# Print the contents of the solvated PDB file for verification
with open("alanine_solvated.pdb", "r") as f:
    print("Contents of alanine_solvated.pdb:")
    print(f.read())

# Step 4: Set up and run molecular dynamics simulation using OpenMM
prmtop = AmberPrmtopFile('alanine_solvated.prmtop')
inpcrd = AmberInpcrdFile('alanine_solvated.inpcrd')

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

traj_filename = "alanine_trajectory.dcd"
pdb_output = "alanine_output.pdb"
simulation.reporters.append(PDBReporter(pdb_output, reporting_interval))
simulation.reporters.append(DCDReporter(traj_filename, reporting_interval))
simulation.reporters.append(StateDataReporter(sys.stdout, reporting_interval, step=True,
                                              potentialEnergy=True, temperature=True))
print("Running production simulation...")
simulation.step(total_steps)

# Step 5: Convert DCD to XTC using MDTraj, align all frames, remove water, and align again
# Use the correct topology file (solvated system) for loading the trajectory
traj = md.load(traj_filename, top='alanine_solvated.pdb')

# Align all frames to the first frame with water
traj.superpose(traj[0])

# Remove water molecules after alignment
traj_no_water = traj.remove_solvent()

# Align all frames to the first frame without water
traj_no_water.superpose(traj_no_water[0])

# Save the aligned trajectory without water
traj_no_water.save_xtc("alanine_trajectory_aligned_no_water.xtc")

# Convert the last frame to PDB for visualization without water
traj_no_water[-1].save_pdb("alanine_final_structure_no_water.pdb")

print("Simulation complete. Aligned PDB and XTC files without water have been generated.")

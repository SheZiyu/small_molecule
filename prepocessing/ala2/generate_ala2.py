"""Generate ala2 trajectory data
env: generate_md
"""

from __future__ import print_function
import sys
import os
import numpy as np
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)


class NumpyReporter:
    def __init__(self, interval):
        self.interval = interval
        self.positions = []

    def report(self, simulation, state):
        self.positions.append(state.getPositions(asNumpy=True))

    def finalize(self, filename):
        np.save(filename, np.array(self.positions))


def load_pdb(filename):
    return app.PDBFile(filename)


def create_system(pdb):
    forcefield = app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
    )
    return system


def create_simulation(system, pdb, temperature=300 * unit.kelvin):
    integrator = mm.LangevinIntegrator(
        temperature, 91.0 / unit.picoseconds, 2.0 * unit.femtoseconds
    )
    integrator.setConstraintTolerance(0.00001)
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"CudaPrecision": "mixed"}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    return simulation


def equilibrate(simulation, steps=2000):
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    print("Equilibrating...")
    simulation.step(steps)


def run_simulation(simulation, trajectory_length, timestep, interval, output_prefix):
    n_steps = int(trajectory_length / timestep)
    numpy_reporter = NumpyReporter(interval)
    simulation.reporters.append(
        app.StateDataReporter(
            Tee(f"{output_prefix}.log", "w"),
            interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=n_steps,
            separator="\t",
        )
    )

    print("Starting trajectory")
    for step in range(n_steps):
        if step % interval == 0:
            state = simulation.context.getState(getPositions=True)
            numpy_reporter.report(simulation, state)
        simulation.step(1)

    numpy_reporter.finalize(f"{output_prefix}.npy")
    del simulation.reporters[:]


def set_gpu_devices(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")


def main():
    pdb_filename = "/storage/florian/ziyu_project/ala2/ala2.pdb"
    trajectory_length = 200 * unit.nanosecond
    timestep = 2 * unit.femtosecond
    interval = int(1 * unit.picosecond / timestep)
    output_prefix = "data/trajectory"
    gpu_ids = [6,7]  # List of GPU IDs to use, e.g., [0, 1] for using GPUs 0 and 1

    set_gpu_devices(gpu_ids)

    pdb = load_pdb(pdb_filename)
    system = create_system(pdb)
    simulation = create_simulation(system, pdb)
    equilibrate(simulation)
    run_simulation(simulation, trajectory_length, timestep, interval, output_prefix)


if __name__ == "__main__":
    main()

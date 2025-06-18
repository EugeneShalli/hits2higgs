#!/usr/bin/env python3
# Simulation script for MG+Pythia output using Fatras + Digitization

from pathlib import Path
import acts
import acts.examples
from acts.examples.simulation import addFatras, addDigitization

from acts.examples import RootMeasurementWriter

u = acts.UnitConstants

def simulate_with_fatras(name, particle_file, vertex_file, output_dir, trackingGeometry, field, decorators, digi_config_file):
    if not particle_file.exists() or not vertex_file.exists():
        print(particle_file, vertex_file)
        print(f"[WARNING] Skipping {name}: Missing input files.")
        return

    print(f"[INFO] Running Fatras + Digitization for {name}")

    # Sequencer setup
    s = acts.examples.Sequencer(events=1, numThreads=-1, logLevel=acts.logging.INFO)

    for d in decorators:
        s.addContextDecorator(d)

    # RNG
    rnd = acts.examples.RandomNumbers(seed=42)

    # Readers
    s.addReader(
        acts.examples.RootParticleReader(
            level=acts.logging.INFO,
            filePath=str(particle_file.resolve()),
            outputParticles="particles_input",
        )
    )

    s.addReader(
        acts.examples.RootVertexReader(
            level=acts.logging.INFO,
            filePath=str(vertex_file.resolve()),
            outputVertices="vertices_input",
        )
    )

    # Output directories
    output_fatras_dir = output_dir / "fatras" / name
    output_fatras_dir.mkdir(parents=True, exist_ok=True)

    # Fatras simulation
    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        enableInteractions=True,
        inputParticles="particles_input",
        outputDirRoot=output_fatras_dir,
    )

    # CSV output
    output_csv_fatras_dir = output_dir / "csv" / name
    output_csv_fatras_dir.mkdir(parents=True, exist_ok=True)

    csv_cfg = acts.examples.CsvSimHitWriter.Config()
    csv_cfg.inputSimHits = "simhits"
    csv_cfg.outputDir = str(output_csv_fatras_dir.resolve())
    csv_cfg.outputStem = "fatras_hits"
    s.addWriter(acts.examples.CsvSimHitWriter(csv_cfg, acts.logging.INFO))

    # Digitization
    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digi_config_file,
        outputDirCsv=output_dir / "csv" / name / "digitization",
        outputDirRoot=output_dir / "root" / name / "digitization",
        rnd=rnd,
    )

    # Run it
    s.run()

if __name__ == "__main__":
    # Base output directory
    output_base = Path.cwd() / "mg_pythia_sim_output"

    # Input files (assuming they are in root/background)
    input_dir = output_base / "root" / "background"
    # background_particles = input_dir / "background_particles.root"
    # background_vertices = input_dir / "background_vertices.root"
    background_particles = input_dir / "run_01particles.root"
    background_vertices = input_dir / "run_01vertices.root"
    # background_vertices = input_dir / "shifted_vertices.root"
    
    # Detector setup
    gdc = acts.examples.GenericDetector.Config()
    # gdc.buildLevel = 3

    detector = acts.examples.GenericDetector(gdc)
    trackingGeometry = detector.trackingGeometry()
    decorators = detector.contextDecorators()

    # Magnetic field
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    # Digitization config
    digi_config_file = Path(__file__).parent / "acts-41.1.0/Examples/Configs/generic-digi-geometric-config.json"
    assert digi_config_file.exists(), f"Digitization config not found: {digi_config_file}"

    try:
        simulate_with_fatras("background", background_particles, background_vertices, output_base, trackingGeometry, field, decorators, digi_config_file)
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")

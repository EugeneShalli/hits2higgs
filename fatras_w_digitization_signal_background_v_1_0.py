#!/usr/bin/env python3
# Eugene Shalugin <eugene.shalugin@ru.nl>
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
    s = acts.examples.Sequencer(events=20000, numThreads=-1, logLevel=acts.logging.INFO)

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

    print("Read events")

    # Output directory for Fatras sim
    output_fatras_dir = output_dir / "fatras" / name
    output_fatras_dir.mkdir(parents=True, exist_ok=True)

    # Run Fatras simulation
    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        enableInteractions=True,
        inputParticles="particles_input",
        outputDirRoot=output_fatras_dir,
    )

    # Write SimHits as CSV
    output_csv_fatras_dir = output_dir / "csv" / name
    output_csv_fatras_dir.mkdir(parents=True, exist_ok=True)

    csv_cfg = acts.examples.CsvSimHitWriter.Config()
    csv_cfg.inputSimHits = "simhits"
    csv_cfg.outputDir = str(output_csv_fatras_dir.resolve())
    csv_cfg.outputStem = "fatras_hits"
    s.addWriter(acts.examples.CsvSimHitWriter(csv_cfg, acts.logging.INFO))

    # Add digitization
    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digi_config_file,
        outputDirCsv=output_dir / "csv" / name / "digitization",
        outputDirRoot=output_dir / "root" / name / "digitization",
        rnd=rnd,
    )

    # root_meas_cfg = RootMeasurementWriter.Config(
    #     inputMeasurements="measurements",
    #     inputClusters="clusters",
    #     inputSimHits="simhits",
    #     inputMeasurementSimHitsMap="measurement_simhits_map",
    #     filePath=str((output_dir / "root" / name / "digitization" / "measurements.root").resolve()),
    #     surfaceByIdentifier=trackingGeometry.geoIdSurfaceMap(),
    # )

    # s.addWriter(RootMeasurementWriter(config=root_meas_cfg, level=acts.logging.INFO))
    
    # Run the sequencer
    s.run()

if __name__ == "__main__":
    output_base = Path.cwd() / "data" / "ttbar_H_production_p50"

    # Input files
    # signal_particles = output_base / "root" / "signal" / "pythia8_particles.root"
    # signal_vertices = output_base / "root" / "signal" / "pythia8_vertices.root"
    # background_particles = output_base / "root" / "background" / "pythia8_particles.root"
    # background_vertices = output_base / "root" / "background" / "pythia8_vertices.root"
    signal_particles = output_base / "root" / "signal" / "particles.root"
    signal_vertices = output_base / "root" / "signal" / "vertices.root"
    background_particles = output_base / "root" / "background" / "particles.root"
    background_vertices = output_base / "root" / "background" / "vertices.root"

    # Detector
    gdc = acts.examples.GenericDetector.Config()
    detector = acts.examples.GenericDetector()
    # trackingGeometry, decorators = detector.finalize(gdc, None)
    trackingGeometry = detector.trackingGeometry()
    decorators = detector.contextDecorators()

    # Magnetic field
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    # Digitization config
    #digi_config_file = (
    #    Path(__file__).resolve().parent.parent / "Examples" / "Configs" / "generic-digi-smearing-config.json"
    #)
    # digi_config_file = Path(__file__).parent / "generic-digi-smearing-config.json"
    # digi_config_file = r"/app/acts-41.1.0/Examples/Configs/generic-digi-smearing-config.json"
    # digi_config_file = Path(__file__).parent / "acts-41.1.0/Examples/Configs/generic-digi-smearing-config.json"
    digi_config_file = Path(__file__).parent / "acts-41.1.0/Examples/Configs/generic-digi-geometric-config.json"
    
    assert digi_config_file.exists(), f"Digitization config not found: {digi_config_file}"

    # Run both simulations
    # simulate_with_fatras("signal", signal_particles, signal_vertices, output_base, trackingGeometry, field, decorators, digi_config_file)
    try:
        simulate_with_fatras("background", background_particles, background_vertices, output_base, trackingGeometry, field, decorators, digi_config_file)
    except:
        pass

    try:
        simulate_with_fatras("signal", signal_particles, signal_vertices, output_base, trackingGeometry, field, decorators, digi_config_file)
    except:
        pass

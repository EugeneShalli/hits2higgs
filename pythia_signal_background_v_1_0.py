#!/usr/bin/env python3
# Eugene Shalugin <eugene.shalugin@ru.nl>
from pathlib import Path

import acts
import acts.examples
from acts.examples.simulation import addPythia8
from tqdm import tqdm

u = acts.UnitConstants


def create_vertex_generator():
    return acts.examples.GaussianVertexGenerator(
        mean=acts.Vector4(0, 0, 0, 0),
        stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
    )


def generate_ttH_signal(outputDir, events=1000):
    print("[INFO] Generating signal: tt̄ + H")
    rnd = acts.examples.RandomNumbers()
    s = acts.examples.Sequencer(events=events, numThreads=-1, logLevel=acts.logging.INFO)

    # Ensure output directories exist
    (outputDir / "csv" / "signal").mkdir(parents=True, exist_ok=True)
    (outputDir / "root" / "signal").mkdir(parents=True, exist_ok=True)

    addPythia8(
        s,
        hardProcess=[
            # Enable only tt'H production
            "HiggsSM:all=off",
            "HiggsSM:gg2Httbar=on",
            "HiggsSM:qqbar2Httbar=on",

            # Parton shower and ISR/FSR
            "PartonLevel:ISR=on",
            "PartonLevel:FSR=on",
            "TimeShower:weightGluonToQuark=8",
            "TimeShower:scaleGluonToQuark=1.0",

            # Phase-space cuts
            "PhaseSpace:pTHatMin=100.",

            # Higgs decays
            "25:onMode=off",
            "25:onIfMatch=5 -5",

            # Decay all to stable
            "HadronLevel:all=on",
        ],
        #hardProcess=[
        #    "HiggsSM:gg2Httbar=on",
        #    "HiggsSM:qqbar2Httbar=on",
        #    "PartonLevel:ISR=on",
        #    "PartonLevel:FSR=on",
        #    "PhaseSpace:pTHatMin=100.",
        #],
        npileup=50,
        vtxGen=create_vertex_generator(),
        rnd=rnd,
        outputDirCsv=outputDir / "csv" / "signal",
        outputDirRoot=outputDir / "root" / "signal",
    )

    return s


def generate_ttbar_background(outputDir, events=1000):
    print("[INFO] Generating background: tt̄ (+ jets)")
    rnd = acts.examples.RandomNumbers()
    s = acts.examples.Sequencer(events=events, numThreads=-1, logLevel=acts.logging.INFO)

    # Ensure output directories exist
    (outputDir / "csv" / "background").mkdir(parents=True, exist_ok=True)
    (outputDir / "root" / "background").mkdir(parents=True, exist_ok=True)

    addPythia8(
        s,
        hardProcess=[
            "Top:qqbar2ttbar=on",
            "Top:gg2ttbar=on",
            "PartonLevel:ISR=on",
            "PartonLevel:FSR=on",
            "PhaseSpace:pTHatMin=100.",
        ],
        npileup=50,
        vtxGen=create_vertex_generator(),
        rnd=rnd,
        outputDirCsv=outputDir / "csv" / "background",
        outputDirRoot=outputDir / "root" / "background",
    )

    return s


if __name__ == "__main__":
    output_base = Path.cwd() / "data" / "ttbar_H_production_p50"

    # Generate signal and background separately
    generate_ttH_signal(output_base, events=20000).run()
    generate_ttbar_background(output_base, events=20000).run()

"""Degrade a WAV file through a phone-call profile and print channel metrics.

Usage:
    python examples/degrade.py --in input.wav --out output.wav \\
        --profile voip_to_cellular_narrowband --seed 0

Without ``--in`` a 1 s synthetic multi-tone is used. The default profile needs an AMR-capable ffmpeg and libopus;
``--profile voip_g722_wideband`` runs on a stock ffmpeg.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from phonesim import CodecUnavailableError, PhoneCallSimulator, analyze_channel, load_audio, save_audio


def _synthetic_24k(dur=1.0, sr=24000):
    t = np.arange(int(sr * dur)) / sr
    sig = sum(0.2 * np.sin(2 * np.pi * f * t) for f in (300, 1500, 4000, 8000))
    return sig.astype(np.float32), sr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", default=None, help="input WAV (default: synthetic tone)")
    ap.add_argument("--out", dest="outfile", default="degraded.wav")
    ap.add_argument("--profile", default="voip_to_cellular_narrowband")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.infile:
        x, sr = load_audio(args.infile, sr=24000)
    else:
        x, sr = _synthetic_24k()
        print("No --in given; using a synthetic 1 s multi-tone at 24 kHz.")

    try:
        sim = PhoneCallSimulator(profile=args.profile, output_sample_rate=24000)
    except CodecUnavailableError as e:
        sys.exit(f"phonesim: {e}")
    y = sim(x, seed=args.seed)
    save_audio(args.outfile, y, sr=24000)
    print(f"Wrote {args.outfile}: {len(np.asarray(y).reshape(-1))} samples @ 24 kHz "
          f"(profile={args.profile}, seed={args.seed})")

    report = analyze_channel(x, y, sample_rate=24000, compute_pesq=False, compute_stoi=False)
    print(f"  SNR (aligned):        {report['snr_db']:.1f} dB")
    print(f"  HF energy >4 kHz:     clean {report['hf_energy_clean_>4k']:.4f} "
          f"-> degraded {report['hf_energy_degraded_>4k']:.4f}")
    print(f"  band energy (degraded): {report['band_energy_degraded']}")


if __name__ == "__main__":
    main()

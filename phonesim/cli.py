"""Command-line interface.

Examples
--------
List profiles and codec availability::

    python -m phonesim.cli info

Degrade a file with a profile::

    python -m phonesim.cli run --profile voip_to_cellular_narrowband \
        --in tts.wav --out degraded.wav --seed 0

Degrade with a YAML/JSON config::

    python -m phonesim.cli run --config experiment.yaml --in tts.wav --out out.wav

Analyse clean vs degraded and write a plot + JSON metrics::

    python -m phonesim.cli analyze --clean tts.wav --degraded degraded.wav \
        --plot analysis.png --json metrics.json

Batch a directory::

    python -m phonesim.cli batch --profile pstn_narrowband \
        --in-dir clean/ --out-dir degraded/ --seed 0
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

_DEFAULT_SR = 24000
_DEFAULT_PROFILE = "voip_to_cellular_narrowband"


def _fail(msg, code: int = 1):
    print(f"phonesim: {msg}", file=sys.stderr)
    sys.exit(code)


def _reason(e: BaseException) -> str:
    """The exception's message; ``str()`` of a KeyError would quote it."""
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


def _require_file(path: str, what: str) -> None:
    if not os.path.isfile(path):
        _fail(f"{what} not found: {path}")


def _cmd_info(args):
    from phonesim import list_profiles
    from phonesim import ffmpeg_backend as fb

    from phonesim.profiles import list_versions

    print("phonesim profiles (name@version; bare name = latest):")
    for p in list_profiles():
        print(f"  - {p}  versions {list_versions(p)}")
    from phonesim import opus_backend as ob

    print(f"\nffmpeg present: {fb.have_ffmpeg()}")
    print(f"real codecs available: {sorted(fb.available_codecs().keys())}")
    print(f"libopus (Opus erasures): {ob.version()}")


def _build_sim(args):
    """Build the simulator; files are loaded and saved at its rates.

    A configuration sets the pipeline's rates, so an explicit ``--sr`` or
    ``--out-sr`` must agree with it. Without a configuration ``--profile``
    applies and the flags default to 24 kHz.
    """
    from phonesim import PhoneCallSimulator
    from phonesim.config import load_config

    randomize = not args.deterministic
    try:
        if not args.config:
            return PhoneCallSimulator(
                input_sample_rate=_DEFAULT_SR if args.sr is None else args.sr,
                output_sample_rate=_DEFAULT_SR if args.out_sr is None else args.out_sr,
                profile=args.profile or _DEFAULT_PROFILE,
                randomize=randomize,
            )
        if args.profile is not None:
            _fail("--profile and --config cannot be combined", code=2)
        if not os.path.isfile(args.config):
            _fail(f"config file not found: {args.config}", code=2)
        cfg = load_config(args.config)
        sim = PhoneCallSimulator.from_config(cfg, randomize=randomize)
    except (KeyError, ValueError, TypeError, RuntimeError) as e:
        _fail(_reason(e))
    for flag, key, given, configured in (
        ("--sr", "input_sr", args.sr, sim.input_sample_rate),
        ("--out-sr", "output_sr", args.out_sr, sim.output_sample_rate),
    ):
        if given is not None and given != configured:
            source = (f"{key}: {configured}" if cfg.get(key) is not None
                      else f"{key} unset, default {configured}")
            _fail(f"{flag} {given} disagrees with the config ({source} Hz)", code=2)
    return sim


def _cmd_run(args):
    from phonesim import load_audio, save_audio

    sim = _build_sim(args)
    _require_file(args.infile, "input file")
    x, sr = load_audio(args.infile, sr=sim.input_sample_rate)
    y, log = sim(x, seed=args.seed, return_log=True)
    save_audio(args.outfile, y, sr=sim.output_sample_rate)
    print(f"Wrote {args.outfile} ({len(y)} samples @ {sim.output_sample_rate} Hz)")
    if args.verbose:
        print("\n".join(log))


def _cmd_analyze(args):
    from phonesim import load_audio
    from phonesim.analysis import analyze_channel, plot_channel

    _require_file(args.clean, "clean file")
    _require_file(args.degraded, "degraded file")
    if args.plot and importlib.util.find_spec("matplotlib") is None:
        _fail("--plot needs matplotlib (pip install 'phonesim[plot]')")
    clean, sr = load_audio(args.clean, sr=args.sr)
    degraded, _ = load_audio(args.degraded, sr=args.sr)
    report = analyze_channel(clean, degraded, sample_rate=args.sr)
    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    if args.plot:
        plot_channel(clean, degraded, sample_rate=args.sr, path=args.plot)
        print(f"Wrote plot to {args.plot}")


def _cmd_batch(args):
    from phonesim import load_audio, save_audio

    sim = _build_sim(args)
    if not os.path.isdir(args.in_dir):
        _fail(f"input directory not found: {args.in_dir}")
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.wav")))
    if not files:
        _fail(f"no .wav files in {args.in_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    for i, fp in enumerate(files):
        x, sr = load_audio(fp, sr=sim.input_sample_rate)
        seed = None if args.seed is None else args.seed + i
        y = sim(x, seed=seed)
        out = os.path.join(args.out_dir, os.path.basename(fp))
        save_audio(out, y, sr=sim.output_sample_rate)
        print(f"[{i+1}/{len(files)}] {os.path.basename(fp)} -> {out}")


def _add_pipeline_options(p: argparse.ArgumentParser) -> None:
    """Options shared by ``run`` and ``batch``: what to simulate and at which rates."""
    p.add_argument("--profile", default=None,
                   help=f"profile name (default: {_DEFAULT_PROFILE}); not with --config")
    p.add_argument("--config", default=None,
                   help="YAML/JSON pipeline configuration file; not with --profile")
    p.add_argument("--sr", type=int, default=None,
                   help="input rate in Hz (default: the config's input_sr, else 24000)")
    p.add_argument("--out-sr", dest="out_sr", type=int, default=None,
                   help="output rate in Hz (default: the config's output_sr, else 24000)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the random draws (batch: seed + file index)")
    p.add_argument("--deterministic", action="store_true",
                   help="use each stage's nominal parameters instead of random draws")


def build_parser():
    p = argparse.ArgumentParser(prog="phonesim", description="Phone-call degradation simulator")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("info", help="List profiles and codec availability")
    pi.set_defaults(func=_cmd_info)

    pr = sub.add_parser("run", help="Degrade a single file")
    pr.add_argument("--in", dest="infile", required=True, help="input audio file")
    pr.add_argument("--out", dest="outfile", required=True, help="output wav path")
    _add_pipeline_options(pr)
    pr.add_argument("--verbose", action="store_true", help="print the stage log")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="Compare clean vs degraded")
    pa.add_argument("--clean", required=True, help="clean reference file")
    pa.add_argument("--degraded", required=True, help="degraded file")
    pa.add_argument("--sr", type=int, default=_DEFAULT_SR,
                    help="rate both files are analysed at (default: 24000)")
    pa.add_argument("--plot", default=None, help="write a PNG comparison (needs matplotlib)")
    pa.add_argument("--json", default=None, help="write the metrics to this JSON file")
    pa.set_defaults(func=_cmd_analyze)

    pb = sub.add_parser("batch", help="Degrade a directory of wavs")
    pb.add_argument("--in-dir", dest="in_dir", required=True, help="directory of .wav inputs")
    pb.add_argument("--out-dir", dest="out_dir", required=True,
                    help="output directory (created if missing)")
    _add_pipeline_options(pb)
    pb.set_defaults(func=_cmd_batch)
    return p


def main(argv=None):
    import soundfile as sf
    from phonesim import CodecUnavailableError

    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except CodecUnavailableError as e:
        _fail(e)
    except (ValueError, OSError, sf.SoundFileError) as e:
        _fail(_reason(e))


if __name__ == "__main__":
    main()

"""Write the versioning goldens under tests/data/ (see tests/test_versions.py).

    python tests/regen_goldens.py              # add what is missing, keep the rest
    python tests/regen_goldens.py --force KEY  # rewrite one name@N

Fingerprints are written for every registered version. Run logs and output
goldens need the codecs, so a version this machine cannot run is reported
and left for a machine that can.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # this checkout's phonesim and tests

from tests import test_versions as tv
from phonesim import CodecUnavailableError, PhoneCallSimulator
from phonesim import profiles as P


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _build(key: str):
    try:
        return PhoneCallSimulator(profile=key)
    except CodecUnavailableError as e:
        print(f"skipped   {key}: {e}")
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", metavar="KEY", help="rewrite this name@N even when it is stored")
    args = ap.parse_args(argv)
    if args.force and args.force not in tv.KEYS:
        ap.error(f"unknown version {args.force!r}; known: {tv.KEYS}")
    tv.DATA.mkdir(exist_ok=True)

    def wanted(stored: bool, key: str) -> bool:
        return not stored or key == args.force

    fingerprints = tv.load_json(tv.FINGERPRINTS)
    with tv.all_codecs_available():
        for key in tv.KEYS:
            if wanted(key in fingerprints, key):
                fingerprints[key] = tv.fingerprint(P.build_profile(key))
                print(f"fingerprint {key}")
    _write_json(tv.FINGERPRINTS, fingerprints)

    logs = tv.load_json(tv.RUN_LOGS)
    x = tv.multitone()
    for key in tv.KEYS:
        want_log = wanted(key in logs, key)
        want_out = key in tv.OUTPUT_GOLDENS and wanted(tv.output_path(key).exists(), key)
        if not (want_log or want_out):
            continue
        sim = _build(key)
        if sim is None:
            continue
        y, log = sim(x, seed=tv.SEED, return_log=True)
        if want_log:
            logs[key] = tv.mask_log(log)
            print(f"run log   {key}")
        if want_out:
            np.save(tv.output_path(key), np.asarray(y, dtype=np.float32))
            print(f"output    {key}")
    _write_json(tv.RUN_LOGS, logs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

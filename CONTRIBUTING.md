# Contributing

- Open a pull request against `main`; every change is reviewed.
- Run `pytest -q` before pushing, with an AMR-capable ffmpeg on `PATH` and libopus
  installed if you can (see README); without them the AMR and Opus-erasure tests
  skip.
- A change to a profile's stage chain or parameters is a new profile version,
  never an edit in place; `pytest` enforces it against
  `tests/data/profile_fingerprints.json`, and `python tests/regen_goldens.py`
  adds the entries for a new version. The codec implementation (ffmpeg build,
  decoder) is logged, not versioned.
- Claims about realism need a measurement in the PR, not an adjective.

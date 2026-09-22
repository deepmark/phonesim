"""CLI sample rates and argument errors.

With ``--config`` the configuration's ``input_sr``/``output_sr`` drive loading
and saving; an explicit flag must agree with it. In profile mode the flags
apply and default to 24 kHz. Mistakes end in a one-line ``phonesim: ...``
message on stderr: exit 2 for argument errors, 1 for everything else.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import soundfile as sf

from phonesim import cli, save_audio
from phonesim.analysis import plot_channel

# 24 kHz in, 8 kHz out, no ffmpeg needed (native G.711).
CONFIG = {
    "input_sr": 24000,
    "output_sr": 8000,
    "stages": [
        {"type": "ResampleStage", "from_sr": 24000, "to_sr": 8000},
        {"type": "CodecStage", "codec": "g711_ulaw", "backend": "native"},
    ],
}
# 16 kHz in and out through an 8 kHz codec. The file must be loaded at the
# configured 16 kHz: loaded at 24 kHz it would come out 1.5x longer with the
# tone shifted down by the same ratio.
CONFIG_16K = {
    "input_sr": 16000,
    "output_sr": 16000,
    "stages": [
        {"type": "ResampleStage", "from_sr": 16000, "to_sr": 8000},
        {"type": "CodecStage", "codec": "g711_ulaw", "backend": "native"},
        {"type": "ResampleStage", "from_sr": 8000, "to_sr": 16000},
    ],
}
SECONDS = 0.5
TONE_HZ = 440.0


def _wav(path, sr: int = 24000, hz: float = TONE_HZ) -> str:
    t = np.arange(int(sr * SECONDS)) / sr
    save_audio(str(path), (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr=sr)
    return str(path)


def _rate_and_frames(path) -> tuple[int, int]:
    info = sf.info(str(path))
    return info.samplerate, info.frames


def _dominant_hz(path) -> float:
    x, sr = sf.read(str(path), dtype="float32")
    spectrum = np.abs(np.fft.rfft(x))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spectrum)])


def _fails(argv, code: int, capsys) -> str:
    """Run the CLI, assert it exits with ``code``, return its one-line stderr."""
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == code
    captured = capsys.readouterr()
    assert captured.err.startswith("phonesim: ") and captured.err.count("\n") == 1
    return captured.err


@pytest.fixture
def config(tmp_path) -> str:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(CONFIG))
    return str(p)


@pytest.fixture
def config16(tmp_path) -> str:
    p = tmp_path / "cfg16.json"
    p.write_text(json.dumps(CONFIG_16K))
    return str(p)


def test_run_config_writes_configured_output_rate(tmp_path, config):
    out = tmp_path / "out.wav"
    cli.main(["run", "--config", config, "--in", _wav(tmp_path / "in.wav"), "--out", str(out)])
    assert _rate_and_frames(out) == (8000, int(8000 * SECONDS))


def test_batch_config_writes_configured_output_rate(tmp_path, config):
    (tmp_path / "in").mkdir()
    for name, hz in (("a.wav", 300.0), ("b.wav", 600.0)):
        _wav(tmp_path / "in" / name, hz=hz)
    cli.main(["batch", "--config", config, "--in-dir", str(tmp_path / "in"),
              "--out-dir", str(tmp_path / "out")])
    outs = sorted((tmp_path / "out").iterdir())
    assert [p.name for p in outs] == ["a.wav", "b.wav"]
    assert all(_rate_and_frames(p) == (8000, int(8000 * SECONDS)) for p in outs)


def test_run_config_accepts_agreeing_flags(tmp_path, config):
    out = tmp_path / "out.wav"
    cli.main(["run", "--config", config, "--sr", "24000", "--out-sr", "8000",
              "--in", _wav(tmp_path / "in.wav"), "--out", str(out)])
    assert _rate_and_frames(out) == (8000, int(8000 * SECONDS))


def test_run_profile_defaults_to_24k(tmp_path):
    out = tmp_path / "out.wav"
    cli.main(["run", "--profile", "pstn_narrowband", "--seed", "0",
              "--in", _wav(tmp_path / "in.wav"), "--out", str(out)])
    assert _rate_and_frames(out) == (24000, int(24000 * SECONDS))


def test_run_profile_honours_out_sr(tmp_path):
    out = tmp_path / "out.wav"
    cli.main(["run", "--profile", "pstn_narrowband", "--seed", "0", "--out-sr", "16000",
              "--in", _wav(tmp_path / "in.wav"), "--out", str(out)])
    assert _rate_and_frames(out) == (16000, int(16000 * SECONDS))


def test_run_config_loads_at_configured_input_rate(tmp_path, config16):
    out = tmp_path / "out.wav"
    cli.main(["run", "--config", config16, "--in", _wav(tmp_path / "in.wav", sr=16000),
              "--out", str(out)])
    assert _rate_and_frames(out) == (16000, int(16000 * SECONDS))
    assert abs(_dominant_hz(out) - TONE_HZ) < 5


def test_batch_config_loads_at_configured_input_rate(tmp_path, config16):
    (tmp_path / "in").mkdir()
    _wav(tmp_path / "in" / "a.wav", sr=16000)
    cli.main(["batch", "--config", config16, "--in-dir", str(tmp_path / "in"),
              "--out-dir", str(tmp_path / "out")])
    out = tmp_path / "out" / "a.wav"
    assert _rate_and_frames(out) == (16000, int(16000 * SECONDS))
    assert abs(_dominant_hz(out) - TONE_HZ) < 5


@pytest.mark.parametrize("flag, key", [("--sr", "input_sr"), ("--out-sr", "output_sr")])
def test_run_config_rejects_disagreeing_flag(tmp_path, config, capsys, flag, key):
    out = tmp_path / "out.wav"
    err = _fails(["run", "--config", config, flag, "16000",
                  "--in", _wav(tmp_path / "in.wav"), "--out", str(out)], 2, capsys)
    assert flag in err and f"{key}: {CONFIG[key]}" in err
    assert not out.exists()


def test_run_config_without_rate_keys_names_the_default(tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"stages": CONFIG["stages"]}))
    err = _fails(["run", "--config", str(cfg), "--sr", "16000",
                  "--in", _wav(tmp_path / "in.wav"), "--out", str(tmp_path / "out.wav")],
                 2, capsys)
    assert "input_sr unset" in err and "24000" in err


@pytest.mark.parametrize("command", ["run", "batch"])
def test_profile_and_config_are_mutually_exclusive(tmp_path, config, capsys, command):
    io = (["--in", _wav(tmp_path / "in.wav"), "--out", str(tmp_path / "out.wav")]
          if command == "run" else ["--in-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    err = _fails([command, "--profile", "pstn_narrowband", "--config", config, *io], 2, capsys)
    assert "--profile" in err and "--config" in err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", ["run", "batch"])
def test_missing_config_file_exits_2(tmp_path, capsys, command):
    io = (["--in", _wav(tmp_path / "in.wav"), "--out", str(tmp_path / "out.wav")]
          if command == "run" else ["--in-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    err = _fails([command, "--config", str(tmp_path / "nope.yaml"), *io], 2, capsys)
    assert "not found" in err and "nope.yaml" in err


# Native G.711 only, so these run on any ffmpeg build.
G711 = ["--profile", "pstn_narrowband"]


def test_run_missing_input_exits_1(tmp_path, capsys):
    err = _fails(["run", *G711, "--in", str(tmp_path / "nope.wav"),
                  "--out", str(tmp_path / "out.wav")], 1, capsys)
    assert "not found" in err and "nope.wav" in err


def test_run_unreadable_input_exits_1(tmp_path, capsys):
    bad = tmp_path / "in.wav"
    bad.write_text("not audio")
    err = _fails(["run", *G711, "--in", str(bad), "--out", str(tmp_path / "out.wav")], 1, capsys)
    assert "in.wav" in err


def test_batch_without_wavs_exits_1(tmp_path, capsys):
    err = _fails(["batch", *G711, "--in-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")],
                 1, capsys)
    assert "no .wav files" in err
    assert not (tmp_path / "out").exists()


def test_analyze_json_writes_report(tmp_path, capsys):
    wav = _wav(tmp_path / "a.wav")
    report = tmp_path / "m.json"
    cli.main(["analyze", "--clean", wav, "--degraded", wav, "--json", str(report)])
    assert json.loads(report.read_text())["sample_rate"] == 24000
    assert json.loads(capsys.readouterr().out)["len_clean"] == int(24000 * SECONDS)


def test_analyze_plot_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    wav = _wav(tmp_path / "a.wav")
    png = tmp_path / "p.png"
    cli.main(["analyze", "--clean", wav, "--degraded", wav, "--plot", str(png)])
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_analyze_plot_without_matplotlib_exits_before_analysing(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    wav = _wav(tmp_path / "a.wav")
    png = tmp_path / "p.png"
    err = _fails(["analyze", "--clean", wav, "--degraded", wav, "--plot", str(png)], 1, capsys)
    assert "matplotlib" in err and "phonesim[plot]" in err
    assert capsys.readouterr().out == "" and not png.exists()


def test_plot_channel_without_matplotlib_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    x = np.zeros(2400, dtype=np.float32)
    with pytest.raises(ImportError, match=r"phonesim\[plot\]"):
        plot_channel(x, x, sample_rate=24000, path="unused.png")

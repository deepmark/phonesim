"""Config loading and validation (``phonesim.config``).

The pipelines use native G.711 only, so no ffmpeg build is needed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from phonesim import cli, save_audio
from phonesim.config import pipeline_from_config, stage_from_dict

PROFILE_CFG = {"profile": "pstn_narrowband", "input_sr": 24000, "output_sr": 24000}
STAGES = [
    {"type": "ResampleStage", "from_sr": 24000, "to_sr": 8000},
    {"type": "CodecStage", "codec": "g711_ulaw", "backend": "native"},
    {"type": "ResampleStage", "from_sr": 8000, "to_sr": 24000},
]


def test_params_none_is_accepted():
    pipe, in_sr, out_sr = pipeline_from_config({**PROFILE_CFG, "params": None})
    assert (in_sr, out_sr) == (24000, 24000)
    assert pipe.describe() == pipeline_from_config(PROFILE_CFG)[0].describe()


def test_params_empty_yaml_key_is_accepted():
    pytest.importorskip("yaml")
    pipe, _, _ = pipeline_from_config("profile: pstn_narrowband\nparams:\n")
    assert pipe.describe() == pipeline_from_config(PROFILE_CFG)[0].describe()


@pytest.mark.parametrize("params", [[1, 2], "law=alaw", 3])
def test_params_must_be_a_mapping(params):
    with pytest.raises(ValueError, match="params must be a mapping"):
        pipeline_from_config({**PROFILE_CFG, "params": params})


@pytest.mark.parametrize("cfg", ["just a string", "- a\n- b\n", "", None, 42])
def test_non_mapping_config_is_rejected(cfg):
    pytest.importorskip("yaml")
    with pytest.raises(ValueError, match="config must be a mapping with 'profile' or 'stages'"):
        pipeline_from_config(cfg)


def test_config_needs_profile_or_stages():
    with pytest.raises(ValueError, match="'profile' or 'stages'"):
        pipeline_from_config({"input_sr": 24000})


def test_unknown_stage_type_lists_the_known_ones():
    with pytest.raises(KeyError) as e:
        stage_from_dict({"type": "FooStage"})
    msg = str(e.value)
    assert "FooStage" in msg
    assert all(name in msg for name in ("ResampleStage", "CodecStage", "PacketLossStage"))
    with pytest.raises(KeyError, match="Unknown stage type 'FooStage'"):
        pipeline_from_config({"stages": [{"type": "FooStage"}]})


@pytest.mark.parametrize("stages", [None, "CodecStage", ["CodecStage"], [{"codec": "g711_ulaw"}]])
def test_malformed_stage_list_is_rejected(stages):
    with pytest.raises((ValueError, KeyError)):
        pipeline_from_config({"stages": stages})


def test_rates_default_to_24000():
    assert pipeline_from_config({"stages": STAGES})[1:] == (24000, 24000)
    assert pipeline_from_config({"stages": STAGES, "input_sr": None, "output_sr": None})[1:] == (
        24000, 24000)
    assert pipeline_from_config({"stages": STAGES, "input_sr": 16000, "output_sr": 8000})[1:] == (
        16000, 8000)


def test_yaml_and_json_files_load_the_same_pipeline(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = {"name": "g711_loop", "input_sr": 16000, "output_sr": 16000, "stages": STAGES}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    from_json = pipeline_from_config(str(tmp_path / "cfg.json"))
    from_yaml = pipeline_from_config(str(tmp_path / "cfg.yaml"))
    assert from_json[1:] == from_yaml[1:] == (16000, 16000)
    assert from_json[0].describe() == from_yaml[0].describe()
    assert from_yaml[0].name == "g711_loop"


def test_invalid_yaml_is_a_value_error(tmp_path):
    pytest.importorskip("yaml")
    bad = tmp_path / "cfg.yaml"
    bad.write_text("stages: [\n  - {type: CodecStage\n")
    with pytest.raises(ValueError, match="not valid YAML") as e:
        pipeline_from_config(str(bad))
    assert "\n" not in str(e.value)


def test_cli_reports_unknown_profile(tmp_path, capsys):
    wav = tmp_path / "in.wav"
    save_audio(str(wav), np.zeros(12000, dtype=np.float32), sr=24000)
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--profile", "nope", "--in", str(wav), "--out", str(tmp_path / "out.wav")])
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("phonesim: Unknown profile 'nope'") and err.count("\n") == 1
    assert "pstn_narrowband" in err

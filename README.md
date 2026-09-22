# phonesim

A provider-free phone-call audio-degradation simulator: a local, seeded model
of the signal path of a telephone call, running the real telephony codecs.

`phonesim` recreates, locally and with no third-party telephony services (no
Twilio / Telnyx / Vonage), the chain of distortions a signal accumulates when it
travels through a real phone call: VoIP/WebRTC transport, PSTN and cellular
interconnects, mobile voice codecs, packet loss, jitter, automatic gain control,
level control, background noise, and clock drift. **The codecs are real**:
G.722, G.726 and (with an AMR-capable ffmpeg build) AMR-NB and AMR-WB run
through ffmpeg, Opus through ffmpeg or the libopus library; G.711 companding is
the ITU-T segmented coder, in-process.
No codec is approximated: a profile whose codec this machine cannot run fails
when built, with the install hint.

Input and output default to **24 kHz**; both rates are parameters. Batches
(`[B, T]`) can be processed as one call or, with `per_example=True`, as one
independent call per row, which is the mode for data generation.

---

## Why this exists

A phone call band-limits the signal to a few kHz, compresses it with a lossy
speech codec (often more than once when the call crosses network boundaries),
chops it into packets that can be lost or delayed, re-levels it, and re-digitises
it at the far end. Any audio system that has to work over calls (speech
recognition, speaker verification, watermark detection, enhancement) needs to be
measured and trained against that channel.

Doing so with real calls means a telephony provider, cost and no
reproducibility. `phonesim` is a local, seeded model of the same signal path,
so you can measure over representative paths, generate degraded data at scale,
and reproduce a result from a seed.

---

## What it does and does not claim to do

**It models** the *signal-level* transformations of a call path: resampling,
the channel's band edges, real codec encode/decode, frame erasures with
receiver-side concealment, playout-buffer behaviour, send-side level control, ambient
noise, multi-transcode chains, and clock drift.

**Every codec is real.** A stock ffmpeg covers G.711, G.722, G.726 and Opus; an
**AMR-capable ffmpeg build** (libopencore-amr + libvo-amrwbenc) adds the real
**AMR-NB and AMR-WB** cellular codecs, the ones that carry mobile voice. AMR
and Opus are decoded with `libopencore_amrnb` / `libopencore_amrwb` / `libopus`
rather than ffmpeg's own decoders, and the run log records the ffmpeg version,
decoder and mode that ran. There is no approximation to fall back to: building
a profile whose codec is missing raises `CodecUnavailableError`. EVS has no
open encoder and ffmpeg has no G.729 encoder, so neither is offered.

---

## Installation

```bash
# from the repo root:
pip install -e .            # core: numpy, torch, soundfile
pip install -e ".[full]"    # + pyyaml, matplotlib, pesq, pystoi, pytest

# codec profiles need ffmpeg on the PATH:
#   apt-get install ffmpeg   (or: brew install ffmpeg)
# the default profile also needs the AMR encoders and libopus (both below)
```

This installs the package (so `import phonesim` works from anywhere) and a
`phonesim` console command. For GPU use, install the torch build matching your
CUDA version from <https://pytorch.org>. Quick check:

```python
import phonesim
print(phonesim.list_profiles())
```

### Real codecs via ffmpeg

`phonesim` probes which codecs the ffmpeg on `PATH` (or the one named by
`PHONESIM_FFMPEG`) can round-trip:

```python
from phonesim import ffmpeg_backend
print(sorted(ffmpeg_backend.available_codecs()))
# stock ffmpeg → ['g711_alaw', 'g711_ulaw', 'g722', 'g726', 'opus']
# AMR-capable  → … plus 'amr_nb' and 'amr_wb'
```

A **stock distro ffmpeg cannot run AMR**: it has neither the encoders
(`libopencore-amrnb`, `libvo-amrwbenc`) nor the OpenCORE decoders. The
no-compile path on Linux is a static GPL build:

```bash
base=https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
curl -fL -o ffmpeg-release-amd64-static.tar.xz "$base"
curl -fL -o ffmpeg-release-amd64-static.tar.xz.md5 "$base.md5"
md5sum -c ffmpeg-release-amd64-static.tar.xz.md5
tar xf ffmpeg-release-amd64-static.tar.xz
install ffmpeg-*-amd64-static/ffmpeg ~/.local/bin/ffmpeg     # ~/.local/bin on PATH

ffmpeg -hide_banner -encoders | grep -i amr
#  A....D libopencore_amrnb   OpenCORE AMR-NB ... (codec amr_nb)
#  A....D libvo_amrwbenc      Android VisualOn AMR-WB ... (codec amr_wb)
```

macOS: Homebrew's `ffmpeg` is built without AMR; build ffmpeg with
`--enable-version3 --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libvo-amrwbenc`.

Licensing: phonesim is MIT and links nothing at build time; it calls the `ffmpeg`
binary at runtime. The AMR libraries (opencore-amr, vo-amrwbenc) are
Apache-2.0; ffmpeg accepts them only with `--enable-version3`, which makes the
ffmpeg build LGPLv3, or GPLv3 when `--enable-gpl` is also set (the static build
above sets both, so it is GPLv3). Distributing such a binary together with your
software carries that licence's obligations; the AMR codecs are also subject to
patents in some jurisdictions.

Without AMR, the `voip_to_cellular_*` and `stress_multi_transcode` profiles
raise `CodecUnavailableError` when built; the PSTN and G.722 profiles work with
a stock ffmpeg, `voip_opus_wideband` also needs libopus (next section).

### Opus erasures need libopus

Erasing Opus frames runs Opus in-process through the libopus shared library
(`apt-get install libopus0`, `brew install opus`; system and Homebrew paths are
searched, `PHONESIM_LIBOPUS` names any other file), so the decoder's own
concealment and in-band FEC apply. Without it, `voip_opus_wideband` and the
two `voip_to_cellular_*` profiles raise `CodecUnavailableError`.

> **Device support:** CPU and NVIDIA CUDA; torch tensors are processed on the
> device they arrive on. Codecs always run on the CPU (ffmpeg, libopus or the
> in-process G.711 coder).

---

## Quick start

### Degrade an audio file

```python
from phonesim import PhoneCallSimulator, load_audio, save_audio

x, sr = load_audio("input.wav", sr=24000)               # loads & resamples to 24 kHz
sim = PhoneCallSimulator(profile="voip_to_cellular_narrowband")
y = sim(x, seed=1234)                                   # reproducible degraded audio
save_audio("degraded.wav", y, sr=24000)
```

`sim(x)` accepts and returns a NumPy array **or** a torch tensor, of rank
`[T]`, `[B, T]`, or `[B, C, T]`, and gives back the same type and rank at 24 kHz.
`sim(x, seed=1234, per_example=True)` gives each row of a batch its own call.

---


## The built-in profiles

A *profile* is a named combination of stages modelling one call path. All
start and end at 24 kHz by default.

Profiles are versioned: `name` resolves to the latest version, `name@N` pins
version `N`. A version fixes the stage chain and its parameters; a change to
either ships as a new version. The codec implementation is not part of the
version: the ffmpeg or libopus build and the decoder that ran are recorded in
the run log, and output is reproducible for a given seed, version and codec
build on one machine and torch thread count. `phonesim info` lists the versions.

| Profile | Models | Internal rate | Path |
|---|---|---|---|
| `pstn_narrowband` | Analogue loop into a G.711 exchange | 8 kHz | send-side level + ambient noise, ~300 Hz low edge, roll-off from 3.4 kHz to 4 kHz, G.711, clock drift |
| `pstn_g726` | PSTN over G.726 ADPCM | 8 kHz | as above with G.726 (32 kbit/s) |
| `voip_opus_wideband` | WebRTC / VoIP | 16 kHz | send-side level + ambient noise, ~50–80 Hz low edge, roll-off from ~7 kHz, Opus (libopus) with erasures up to 5 % (FEC + PLC), adaptive playout, clock drift |
| `voip_g722_wideband` | SIP HD voice over G.722 | 16 kHz | send-side level + ambient noise, ~50–80 Hz low edge, roll-off from ~7 kHz, G.722 with erasures up to 1 % (G.711 App. I PLC), adaptive playout, clock drift |
| `voip_to_cellular_wideband` | VoIP into mobile **HD** voice | 16 kHz | send-side level + ambient noise, Opus hop with erasures up to 1 % → wideband edges → AMR-WB with radio erasures up to 1 %, optional AMR-NB second transcode, adaptive playout, clock drift |
| `voip_to_cellular_narrowband` | VoIP into a **regular (non-HD)** mobile call | 8 kHz | send-side level + ambient noise, Opus hop with erasures up to 1 % → 8 kHz, ~100 Hz low edge, roll-off from 3.4 kHz to 4 kHz → AMR-NB with radio erasures up to 1 %, adaptive playout, clock drift; `bitrate` selects the AMR-NB mode |
| `stress_multi_transcode` | Synthetic stress chain (not a real route) | 16/8/16 kHz | brick-wall band-pass, Opus → packet loss → AMR-NB → G.711 → packet loss → AMR-WB, jitter buffer, AGC, noise, clipping, speed drift, time offset |

The six call-path profiles share one structure. On the send side the active
speech level is set and ambient noise added before the first encoder, as a
handset or platform does. Band edges are the ones the codec and handset
define: a 2nd-order low edge (250–320 Hz on the analogue loop, ~100 Hz on
the digital narrowband path, 50–80 Hz on the wideband ones) and a roll-off
from the codec's passband to the channel Nyquist. The four packetised
profiles erase frames and conceal them on the receiving side. AMR and Opus
frames are removed from the coded stream before the decoder: AMR runs its
decoder's error concealment, which carries the error into the following
frames; libopus decodes the frame from the next packet's in-band FEC when it
carries one and otherwise runs its PLC. G.722 is decoded in full and the
erased frames are replaced in the decoded PCM by the ITU-T G.711 Appendix I
waveform substitution (`phonesim.plc`), scaled to 16 kHz. Network loss and
late arrivals are one erasure process per hop, in bursts: up to 1 % of frames
on a managed trunk or a radio leg (the LTE conversational-voice loss target),
up to 5 % on the public Internet (`voip_opus_wideband`). The playout buffer
adapts by expanding or dropping single frames; clock drift is within ±50 ppm.
`stress_multi_transcode` instead uses brick-wall band-passes, packet loss and
a jitter buffer in the decoded signal, AGC, added noise, clipping, speed drift
and a start offset.

Use `phonesim.list_profiles()` to enumerate profiles and
`PhoneCallSimulator(profile=...).describe()` to print the exact stage chain with
the backend and mode of each codec.

**Wideband vs narrowband cellular.** `voip_to_cellular_wideband` models an **HD
voice** call (AMR-WB, ~7 kHz), which the model assumes is negotiated along the
whole path. `voip_to_cellular_narrowband` models a call delivered **narrowband**
(AMR-NB, ~3.4 kHz), the assumption for a call that does not negotiate HD end to
end. Its `bitrate` argument selects the AMR-NB mode (`12.2k`, the default, down
to `4.75k`); lower modes model poorer radio conditions, e.g.
`profile_params={"bitrate": "7.4k"}`.

---

## The signal path, stage by stage

The package is built from small `Stage` modules (each an `nn.Module`) composed
into a `Pipeline`. The stages, grouped by what they model:

- **Rate & bandwidth**: `ResampleStage` (polyphase resampling
  between rates), `ChannelEdgeStage` (2nd-order low edge plus a roll-off to the
  channel Nyquist; the edges of a digital channel), `BandlimitStage` (brick-wall
  FIR band-pass; `stress_multi_transcode` and custom chains).
- **Send side**: `AmbientNoiseStage` (room noise at an SNR relative to the active
  speech level, entering before the encoder), `SpeechLevelStage` (P.56-style
  active speech level), `LimiterStage` (peak limiter, identity below the knee).
- **Codecs**: `CodecStage`, three backends: `ffmpeg` (G.711/G.722/G.726/Opus,
  plus AMR-NB/AMR-WB when built in; AMR and Opus through their reference
  decoders, delay-compensated), `native` (G.711 via `CompandingStage`, the
  exact segmented coder; the default for G.711, used by `pstn_narrowband` and
  the G.711 hop of `stress_multi_transcode`, while `pstn_g726` runs G.726
  through `ffmpeg`) and `libopus` (Opus in-process). `erasure_rate` erases
  20 ms frames in bursts. AMR and Opus frames are removed from the coded
  stream before the decoder (AMR: the slot becomes a NO_DATA frame and the
  decoder runs its error concealment, which carries the error into the
  following frames; Opus: libopus decodes from the next packet's in-band FEC
  when it carries one, otherwise runs its PLC). G.711, G.722 and G.726 are
  decoded in full and the erased frames are replaced in the decoded PCM by
  the ITU-T G.711 Appendix I waveform substitution (`phonesim.plc`).
- **Packetization / transport**: `PlayoutBufferStage` (per-frame under/over-run
  with overlap-add; bursty late arrivals concealed), `PacketLossStage` (bursty
  loss with repeat-and-fade in the decoded signal), `JitterBufferStage` (late
  frames and under-runs in the decoded signal); the packetised profiles erase
  frames in `CodecStage` and use `PlayoutBufferStage`, `stress_multi_transcode`
  and custom chains use the other two.
- **Level & nonlinearity**: `AGCStage`, `GainStage`, `ClipStage`.
- **Noise**: `NoiseStage` (white / pink / mains-hum at a target SNR).
- **Timing**: `ClockDriftStage` (clock mismatch in ppm, windowed-sinc
  fractional resampling), `SpeedDriftStage` (drift as a linear-interpolation
  resample), `TimeOffsetStage` (recording start offset).

A final resample returns the signal to 24 kHz regardless of the internal path.

---

## Signal analysis

`phonesim.analyze_channel(clean, degraded, sample_rate=24000, metrics=None)`
returns a dictionary of metrics: alignment-corrected SNR, per-band
energy ratios, high-frequency energy above 4 kHz / 8 kHz, PESQ and STOI (if those
packages are installed), and, if you pass `metrics={name: fn}` with
`fn(audio, sr) -> float`, each of your own task metrics on clean vs degraded
audio.

`phonesim.plot_channel(clean, degraded, path="channel_analysis.png")` renders a
side-by-side comparison (waveforms, spectrograms, log-mel, frequency response and
band energy) to a PNG.

```python
from phonesim import analyze_channel
report = analyze_channel(clean, degraded, sample_rate=24000,
                         metrics={"my_score": my_metric})
print(report["snr_db"], report["my_score_clean"], report["my_score_degraded"])
```

---

## Command-line interface

```bash
python -m phonesim.cli info                              # list profiles & available codecs
python -m phonesim.cli run  --in a.wav --out b.wav --profile pstn_narrowband --seed 7
python -m phonesim.cli analyze --clean a.wav --degraded b.wav --plot out.png --json metrics.json
python -m phonesim.cli batch --in-dir clips/ --out-dir degraded/ --profile voip_opus_wideband
```

`run` and `batch` build the pipeline from `--profile` (default
`voip_to_cellular_narrowband`) or from `--config config.yaml`, which replaces
`--profile`: the file names a profile or lists the stages itself. With
`--config` the file's `input_sr` / `output_sr` (default 24000) set the rates
and `--sr` / `--out-sr` may only repeat them; a different value exits with
status 2. Without `--config`, `--sr` / `--out-sr` set the rates (default
24000). `--deterministic` uses the nominal (non-random) parameters.
`analyze --plot` needs `pip install "phonesim[plot]"`.

---

## Configuration files

Pipelines can be described in YAML/JSON instead of code, either by naming a
profile or by listing stages explicitly:

```yaml
input_sr: 24000
output_sr: 24000
stages:
  - {type: ResampleStage, from_sr: 24000, to_sr: 16000}
  - {type: BandlimitStage, low_hz: 50, high_hz: 7000}
  - {type: CodecStage, codec: amr_wb, bitrate: 12.65k}
  - {type: PacketLossStage, loss_rate: 0.02, burst_probability: 0.2}
  - {type: NoiseStage, snr_db: [25, 40]}
  - {type: ResampleStage, from_sr: 16000, to_sr: 24000}
```

```python
sim = PhoneCallSimulator.from_config("config.yaml")
```

`input_sr` and `output_sr` are optional and default to 24000.

---

## Reproducibility

Every call takes an optional `seed`. The simulator seeds a dedicated
`torch.Generator`, so the same input, seed, profile version and codec build
(ffmpeg, libopus) yield identical output on one machine and torch thread count.
Across machines the G.711 path reproduces to better than 60 dB SNR (the test
suite pins its output); the adaptive coders (G.726, G.722, Opus, AMR) turn
last-bit differences in their input into different bitstreams. The run log
has one line per stage with the parameters drawn and, for each codec, the
build, decoder and mode:

```python
y, log = sim(x, seed=1234, return_log=True)
```

With `per_example=True`, row 0 of a batch uses `seed` and the other rows use
seeds drawn from a generator seeded with it (`phonesim.simulator.row_seeds`).

---

## Shapes at a glance

- **Internal tensor convention**: `[B, C, T]`. The public API accepts `[T]`,
  `[B, T]`, `[B, C, T]`, NumPy or torch, and restores the original rank/type.
- **Output rate**: always `output_sample_rate` (24 kHz by default).

---

## Project layout

```
phonesim/
  phonesim/
    core.py           # Stage / Pipeline / SimContext, shape & dtype handling
    dsp.py            # resampling, FIR filters, level helpers
    ffmpeg_backend.py # codec round-trips through ffmpeg (G.711/G.722/G.726/Opus + AMR-NB/WB if built in)
    opus_backend.py   # Opus through the libopus shared library (PLC, in-band FEC)
    plc.py            # ITU-T G.711 Appendix I packet loss concealment
    stages/           # all signal-path stages
    profiles/         # named profiles + registry
    simulator.py      # PhoneCallSimulator / PhoneCallPipeline
    analysis.py       # metrics + plotting
    config.py         # YAML/JSON pipeline loading
    io_utils.py       # load_audio / save_audio
    cli.py            # command-line interface
  tests/              # pytest suite
  examples/           # runnable scripts
  pyproject.toml      # packaging / install / console script
  README.md           # this documentation
```

## Effective bandwidth per profile

Share of the *output* energy per band, and the −3 dB band relative to 1 kHz,
for a 5 s white-noise probe through each profile (24 kHz in/out, deterministic
parameters, 4096-point Welch spectra):

| Profile | 0–4 kHz | 4–8 kHz | 8–12 kHz | −3 dB band |
|---|---|---|---|---|
| `pstn_narrowband` | 1.00 | ~0 | ~0 | ~300 Hz – 3.4 kHz |
| `pstn_g726` | 1.00 | ~0 | ~0 | ~300 Hz – 3.4 kHz |
| `voip_to_cellular_narrowband` | 1.00 | ~0 | ~0 | ~120 Hz – 3.3 kHz, roll-off to 4 kHz |
| `stress_multi_transcode` | 1.00 | ~0 | ~0 | ~350 Hz – 3 kHz |
| `voip_to_cellular_wideband` | 0.68 | 0.32 | ~0 | ~80 Hz – 6 kHz¹ |
| `voip_opus_wideband` | 0.64 | 0.36 | ~0 | ~80 Hz – 6.9 kHz |
| `voip_g722_wideband` | 0.59 | 0.41 | ~0 | ~70 Hz – 7 kHz |

¹ AMR-WB at 12.65 kbit/s; `profile_params={"second_transcode": True}` adds an
AMR-NB interconnect hop that collapses the call to the narrowband path.

If a downstream system depends on fine spectral detail above 3.4 kHz, only the
wideband profiles keep it; the mobile narrowband path, the common case for a
call to an ordinary number, keeps the telephone band only.

## Assumptions and limitations

- **No EVS, no G.729.** EVS has no open encoder and ffmpeg has no G.729 encoder
  (an open one exists outside ffmpeg), so neither is offered; VoLTE with EVS is
  not modelled.
- **G.722 and G.726 conceal with the G.711 Appendix I algorithm**, scaled to the
  codec's rate; G.722's own Appendix III/IV PLC is not implemented. AMR and Opus
  conceal with their own decoders.
- **G.711, G.722 and G.726 decoder state is never disturbed by an erasure.**
  These codecs are decoded in full and the erased frames are replaced in the
  decoded PCM, so the post-erasure divergence a real ADPCM receiver (G.722,
  G.726) shows is not modelled; frames after an erasure equal a loss-free
  decode.
- **`native` and `ffmpeg` G.711 are not bit-identical.** Same decoder; ffmpeg's
  encoder table rounds differently at the decision levels, so 512 µ-law and 964
  A-law of the 65536 int16 inputs take the adjacent code, one level apart.
- **Transport effects are statistical, not protocol-accurate.** Erasures follow
  a two-state (Gilbert-Elliott) chain per codec hop; jitter-buffer adaptation
  and level control are parametric models, not RTP/WebRTC implementations.
- **No acoustic path is modeled.** Room reverberation, speaker/handset
  acoustics, acoustic echo, and the receiver's re-recording/re-digitization step
  are out of scope; add them upstream if needed.

## Checking a profile against your own path

Each profile is a physically motivated model of its path; no comparison against
recorded calls ships with this release. To check a profile against your own path, record the same clips through a real
call, run `analyze_channel(clean, received)` on both the real and the simulated
output, and compare the distributions. Prefer changing a parameter for a physical
reason over tuning it to a handful of recordings.

## Further reading

- `examples/degrade.py` — degrade a WAV (or a synthetic tone) through a
  profile and print channel metrics.
- `tests/test_phonesim.py` — executable specification: output rate, shape/type
  preservation, determinism, length preservation, band-limiting behaviour, the
  physics of each stage and batch processing are asserted there and double as
  usage examples.
- Each stage and the high-level classes carry docstrings; e.g.
  `help(phonesim.PhoneCallSimulator)`.

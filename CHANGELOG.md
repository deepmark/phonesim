# Changelog

## Unreleased

### Fixed

- Resampling between rates whose reduced ratio has an upsampling factor above
  16 (44.1, 22.05 and 11.025 kHz against 8, 16, 24, 32 and 48 kHz, among
  others) applies the windowed-sinc low-pass of the other rates at exact output
  positions, so a caller at these rates gets the channel a caller at 24 or
  48 kHz gets, and `ResampleStage(zeros=...)` applies at these ratios too. 0.1.0
  used a FIR and linear interpolation there. Every profile resamples between
  the caller's rate and 8 or 16 kHz on entry and exit, and for a 44.1 kHz
  caller:
  - a 44.1 → 8 → 44.1 kHz round trip was 1.9 dB down at 2 kHz, 4.4 dB at 3 kHz
    and 6.5 dB at 3.4 kHz, and 16 → 44.1 kHz was 2.9 dB down at 5 kHz and
    6.3 dB at 7 kHz;
  - each hop shifted the signal by up to 51 µs, earlier going down and later
    going up, so the shifts cancelled only in a profile that leaves through the
    rate it entered at, with matching input and output rates;
    `voip_to_cellular_narrowband` (in at 16 kHz, out from 8 kHz) came out about
    31 µs late even then;
  - the interpolation positions were computed in float32 and drifted: up to
    about 0.2 samples off after a minute of input and up to 4 samples after ten.

  Output changes wherever such a ratio is resampled: the simulator with an
  input or output rate of 44.1, 22.05 or 11.025 kHz, `load_audio` of a file at
  such a rate (including to the default 24 kHz, so `phonesim run` and `batch`
  on such files change), PESQ in `analyze_channel` at such a rate, and custom
  chains that resample or run a codec stage at such a rate. On 44.1 kHz speech
  through `pstn_narrowband` and `voip_g722_wideband` the difference from 0.1.0
  is 16–27 dB below the output with a 44.1 kHz output rate, and 9.5–21 dB with
  a 24 kHz one, where 0.1.0's shifts did not cancel; through
  `voip_to_cellular_narrowband` it is 13–17 dB, and for `load_audio` to 24 kHz
  20–24 dB below the signal. Resampling between 8, 16, 24, 32 and 48 kHz
  is unchanged, bit for bit.
- `ClockDriftStage` and `SpeedDriftStage` compute where each output sample
  falls in the input exactly (in float64, and in integer arithmetic through the
  new `dsp.linear_resize`). 0.1.0 computed those positions in float32, so their
  error grew along the signal: at 16 kHz, `ClockDriftStage` was up to 0.01
  samples off a few seconds in and 0.09 after a minute, and from 2^23 samples
  (8.7 min) on its positions were whole samples; `SpeedDriftStage` was up to
  0.09 samples off after a minute and 1.6 after ten. A run therefore changes,
  at any sample rate, when its drift stage draws a drift: every randomized run
  that draws at least 0.5 ppm of clock drift or any speed drift (only
  `stress_multi_transcode` has a speed-drift stage), and deterministic runs of
  custom drift stages whose deterministic draw is a drift, such as
  `ClockDriftStage(ppm=40)`. Deterministic runs of the built-in profiles
  (`randomize=False`, `--deterministic`) draw no drift and are unchanged at 8,
  16, 24, 32 and 48 kHz. On 24 kHz speech, randomized runs differ from 0.1.0
  by about 57–70 dB below the output for 5 s inputs, 37–46 dB for 60 s and
  15–25 dB for 600 s.
- `NoiseStage(color="hum")` computes its time axis in float64; in float32 the
  hum's error grew from 60 dB below it after one minute to 35–40 dB below
  after ten. No profile uses it.
- `dsp.resample` returns an empty signal for an empty input instead of
  raising.

### Changed

- Profile versions fix the stage chain and its parameters, not the arithmetic
  inside a stage: this release keeps every version although output changes
  (above). Output is reproducible for a given phonesim release, seed, profile
  version and codec build on one machine and torch thread count; phonesim
  0.1.0 still regenerates 0.1.0 output.

### Deprecated

- An unset output rate with an input rate other than 24 kHz:
  `PhoneCallSimulator` and `PhoneCallPipeline` without `output_sample_rate`,
  `build_profile` without `output_sr`, a config without `output_sr`, and
  `phonesim run` / `batch` with `--sr` but without `--out-sr` still resample
  the output to 24 kHz, now with a `FutureWarning`. From 0.3.0 the output rate
  defaults to the input rate; set it to keep 24 kHz.
- `save_audio` without `sr` writes a 24 kHz header whatever rate the signal is
  at, now with a `FutureWarning`; from 0.3.0 `sr` is required.
- `analyze_channel` and `plot_channel` without `sample_rate` take the signals
  to be at 24 kHz, now with a `FutureWarning`; from 0.3.0 `sample_rate` is
  required.
- A `ResampleStage` whose `from_sr` disagrees with the rate the signal arrives
  at. `from_sr` was documented as a sanity check but never checked; the stage
  now warns (`FutureWarning`) and resamples from the signal's rate as before.
  From 0.3.0 this raises `ValueError`. `from_sr=None` takes any rate, and the
  built-in profiles' chains line up at every rate.

## 0.1.0

- Seven profiles, each a seeded model of one call path: `pstn_narrowband`
  (analogue loop into a G.711 exchange), `pstn_g726` (PSTN over G.726 ADPCM),
  `voip_opus_wideband` (WebRTC/VoIP over Opus), `voip_g722_wideband` (SIP HD
  voice over G.722), `voip_to_cellular_wideband` (VoIP into an AMR-WB HD-voice
  call), `voip_to_cellular_narrowband` (VoIP into an ordinary AMR-NB mobile
  call; the default) and `stress_multi_transcode` (a synthetic stress chain,
  not a real route).
- Real codecs only: G.711 with the ITU-T segmented coder in-process; G.722,
  G.726, AMR-NB and AMR-WB through ffmpeg; Opus through ffmpeg or the libopus
  shared library. AMR and Opus decode with `libopencore_amrnb`,
  `libopencore_amrwb` and `libopus`; algorithmic delay is compensated. A
  profile whose codec this machine cannot run raises `CodecUnavailableError`
  when built.
- Frame erasures in `CodecStage`, concealed on the receiving side. AMR and
  Opus frames are removed from the coded stream before the decoder: AMR runs
  its decoder's error concealment, which carries the error into the following
  frames; libopus decodes from the next packet's in-band FEC when it carries
  one, otherwise its PLC. G.711, G.722 and G.726 are decoded in full and the
  erased frames are replaced in the decoded PCM by the ITU-T G.711 Appendix I
  waveform substitution (`phonesim.plc`).
- Send-side speech level and ambient noise, codec- and handset-defined band
  edges, adaptive playout, clock drift in ppm, and the stress chain's
  packet-loss, jitter-buffer, AGC, noise, clipping, speed-drift and
  time-offset stages.
- Profile versions: `name@N`; a bare name is the latest. A version fixes the
  stage chain and its parameters; the codec build and decoder are recorded in
  the run log, not versioned. `phonesim info` lists versions.
- Reproducibility: a seed drives one `torch.Generator`; the same input, seed,
  profile version and codec build give the same output on one machine and
  torch thread count. `per_example=True` gives each row of a batch its own
  call, seeded by `row_seeds`. The run log has one line per stage.
- `analyze_channel` (aligned SNR, band energies, high-frequency energy,
  optional PESQ and STOI, caller-supplied `metrics=`) and `plot_channel`.
- `phonesim` CLI (`info`, `run`, `analyze`, `batch`) and YAML/JSON pipeline
  configs.
- Tests pin every profile version's stage chain, parameters and run log; CI
  runs them on a distro ffmpeg (AMR tests skip) and on a static AMR-capable
  build pinned by sha256.

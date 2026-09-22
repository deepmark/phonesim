# Changelog

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

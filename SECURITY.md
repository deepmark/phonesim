# Security

phonesim reads audio files with libsndfile (through `soundfile`) and runs
`ffmpeg` as a subprocess on WAV files it wrote itself. Treat untrusted input
files as untrusted input to libsndfile. Report vulnerabilities privately to
security@deepmark.me.

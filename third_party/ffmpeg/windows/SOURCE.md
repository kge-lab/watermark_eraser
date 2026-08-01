# FFmpeg 7.1 Windows binary and corresponding source

The Windows application bundle contains this unmodified executable supplied by
`imageio-ffmpeg` 0.6.0 and invokes it as a separate process:

- file: `imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`
- version: `7.1-essentials_build-www.gyan.dev`
- license: GNU General Public License version 3
- SHA-256: `2CE797A0F88D7F067180338FB227F7B1928EA727BD9A4D7A1D022F7C52AF71A3`

The executable is byte-for-byte identical to `bin/ffmpeg.exe` in Gyan Doshi's
original `ffmpeg-7.1-essentials_build.zip` release package:

- package: <https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip>
- package SHA-256: `FA7D4D7E795DB0E2503F49F105F46ED5852386F0CFDD819899BE3B65EBDE24FC`
- release page: <https://github.com/GyanD/codexffmpeg/releases/tag/7.1>

The original package's license text and build README are included beside this
file with line endings and trailing whitespace normalized. `README.txt` records
the complete configure/build feature list and identifies FFmpeg commit
`b08d7969c5` as its source.

Corresponding FFmpeg source is available from both locations below:

- same release server as this application binary:
  <https://github.com/kge-lab/watermark_eraser/releases/download/release-0.1.1/ffmpeg-7.1-b08d7969c5-source.tar.gz>
  (SHA-256: `02FA6D9827DA3B6786E4DF821218CC036DB2B4481E7F48267C2DCDA695633AFC`)
- canonical upstream commit:
  <https://github.com/FFmpeg/FFmpeg/commit/b08d7969c5>

The application project does not modify the FFmpeg binary or FFmpeg source.

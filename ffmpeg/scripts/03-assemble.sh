#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$FFMPEG_DIR/fixtures"

mkdir -p "$OUT_DIR"

SRC1="$OUT_DIR/pattern_720p30.mp4"
SRC2="$OUT_DIR/clip_accurate.mp4"
SRC3="$OUT_DIR/clip_fast.mp4"
SRC4="$OUT_DIR/clip.mkv"
SRC5="$OUT_DIR/av_sync.mp4"
SRC6="$OUT_DIR/audio_av.m4a"
SRC7="$OUT_DIR/tone_440.wav"
SRC8="$OUT_DIR/av_tone.mp4"
SRC8="$OUT_DIR/pitched_av.mp4"

echo "Generating fixture at $SRC2"
ffmpeg -i "$SRC1" -ss 2 -t 2 "$SRC2"

echo "Generating fixture at $SRC3"
ffmpeg -ss 2 -to 4 -i "$SRC1" -c copy "$SRC3"

echo "Generating fixture at $SRC4"
ffmpeg -i "$SRC1" -c copy "$SRC4"

echo "Generating fixture at $SRC6"
ffmpeg -i "$SRC5" -vn -c:a copy "$SRC6"

echo "Generating fixture at $SRC8"
ffmpeg -i "$SRC5" -i "$SRC7" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest "$SRC8"

echo "Generating fixture at $SRC8"
ffmpeg -i "$SRC5" -f lavfi -i "sine=frequency=550:sample_rate=44100" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "$SRC8"

echo "Generation complete"
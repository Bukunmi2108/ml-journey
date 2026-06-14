#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$FFMPEG_DIR/fixtures"

mkdir -p "$OUT_DIR"

SRC1="$OUT_DIR/pattern_720p30.mp4"
SRC2="$OUT_DIR/bars_1080p.mp4"
SRC3="$OUT_DIR/tone_440.wav"

echo "Generating fixture at $SRC1"
ffmpeg -f lavfi -i testsrc=s=1280x720:r=30:d=5 "$SRC1"

echo "Generating fixture at $SRC2"
ffmpeg -f lavfi -i smptebars=s=1920x1080:d=10:r=30 "$SRC2"

echo "Generating fixture at $SRC3"
ffmpeg -f lavfi -i sine=f=440:d=5 "$SRC3"

echo "Generation complete"
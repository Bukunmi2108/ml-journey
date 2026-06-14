#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$FFMPEG_DIR/fixtures"

mkdir -p "$OUT_DIR"

SRC1="$OUT_DIR/pattern_720p30.mp4"
SRC2="$OUT_DIR/bars_1080p.mp4"
SRC3="$OUT_DIR/tone_440.wav"
SRC4="$OUT_DIR/av_sync.mp4"
SRC5="$OUT_DIR/labelled.mp4"
SRC6="$OUT_DIR/one_frame.mp4"
SRC7="$OUT_DIR/odd_dim.mp4"
SRC8="$OUT_DIR/fast_fps.mp4"

echo "Generating fixture at $SRC1"
ffmpeg -y -f lavfi -i testsrc=s=1280x720:r=30:d=5 -pix_fmt yuv420p "$SRC1"

echo "Generating fixture at $SRC2"
ffmpeg -f lavfi -i smptebars=s=1920x1080:d=10:r=30 "$SRC2"

echo "Generating fixture at $SRC3"
ffmpeg -f lavfi -i sine=f=440:d=5 "$SRC3"

echo "Generating fixture at $SRC4"
ffmpeg -f lavfi -i smptebars=s=1920x1080:r=30 -f lavfi -i sine=f=440 -c:v libx264 -c:a aac -t 10 -map 0:v:0 -map 1:a:0 "$SRC4"

echo "Generating fixture at $SRC5"
ffmpeg -i "$SRC2" -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='ffmpeg is chunky':fontcolor=white:fontsize=64:box=1:boxcolor=black@0.5:x=(w-text_w)/2:y=(h-text_h)/2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='%{pts\:hms}':fontcolor=white:fontsize=32:box=1:boxcolor=black@0.5:x=w-text_w-20:y=h-text_h-20" -codec:a copy "$SRC5"

# One frame one sec
echo "Generating fixture at $SRC6"
ffmpeg -f lavfi -i color=c=0x4d1a7f:s=1080x720:r=1:d=1 "$SRC6"

# Odd Dimensions
echo "Generating fixture at $SRC7"
ffmpeg -f lavfi -i color=c=0x4d1a7f:s=641x359:d=5 "$SRC7"

# Fast FPS
echo "Generating fixture at $SRC8"
ffmpeg -f lavfi -i testsrc=s=1080x720:r=120:d=5 "$SRC8"

echo "Generation complete"
#!/bin/bash
# Download the published dataset (RADAR4KIT, doi:10.35097/gycyvxthuw98sfpq), verify and unpack it.
#
# RADAR delivers every record as a BagIt container (a tar holding 10.35097-<id>/data/dataset/<Part>.tar); the archive listed in
# the dataset's SHA256SUMS is the inner tar. This script streams the inner tar out of the container, checks its SHA-256 and
# unpacks it, so that <dest>/<Case>/<Case>.case and <dest>/PIV/ result. RADAR limits the request rate: failed requests are
# retried after a pause. Needs curl, tar, sha256sum and about 15 GB of free space per case (34 GB for Pinhole-L) during the
# download; after unpacking 200 GB in total.
#
# Usage: scripts/download_data.sh <dest> [parts ...]      parts: Drop-S … Pinhole-L, PIV (default: all)
set -euo pipefail
DEST=${1:?usage: download_data.sh <dest> [parts ...]}; shift || true
declare -A ID=( [Drop-S]=pn1e10gzbcbr1es5 [Drop-M]=wx9xndtgc0q99566 [Drop-L]=fmv9u5w8wa7rg8c8 [Drop-XL]=7j69dwbftunf1egw
  [Slot-S]=363nr436dryu1eg3 [Slot-M]=73114d1874t8bqpc [Slot-L]=dwzkhend7r84m5vv [Slot-L-Bent]=230ua5r9974u64xm
  [Pinhole-S]=k1q987zhjun95p0s [Pinhole-M]=syggkzhzzg4ew7gc [Pinhole-L]=2w800mp1jb1rsp8t [PIV]=mcxab6y7kkyucumh )
HEADER=gycyvxthuw98sfpq
URL=https://radar.kit.edu/radar-backend/archives
PARTS=("$@"); [ ${#PARTS[@]} -eq 0 ] && PARTS=(Drop-S Drop-M Drop-L Drop-XL Slot-S Slot-M Slot-L Slot-L-Bent Pinhole-S Pinhole-M Pinhole-L PIV)
mkdir -p "$DEST"; cd "$DEST"

fetch() {   # fetch <id> <file>: download with retries (rate limit answers are a few bytes long)
  for try in $(seq 1 20); do
    curl -sfL -o "$2" "$URL/$1/versions/1/content" || true
    if [ -f "$2" ] && [ "$(stat -c %s "$2")" -gt 100000 ]; then return 0; fi
    echo "  $2: request refused (try $try), waiting 2 min"; sleep 120
  done
  echo "  $2: download failed"; return 1
}

if [ ! -f SHA256SUMS ]; then                                # header record: README, cases.csv, checksums
  fetch $HEADER header.bag.tar
  tar xf header.bag.tar --strip-components=3 --wildcards "*/data/dataset/*" && rm header.bag.tar
fi
for p in "${PARTS[@]}"; do
  if [ -d "$p" ]; then echo "$p: present, skipped"; continue; fi
  echo "$p: downloading"
  fetch "${ID[$p]}" "$p.bag.tar"
  inner=$(tar tf "$p.bag.tar" | grep "/data/dataset/$p.tar$")
  h=$(tar xOf "$p.bag.tar" "$inner" | tee >(tar x) | sha256sum | cut -d' ' -f1); wait
  ref=$(grep " $p.tar$" SHA256SUMS | cut -d' ' -f1)
  if [ "$h" != "$ref" ]; then echo "$p: CHECKSUM MISMATCH ($h, expected $ref)"; exit 1; fi
  rm "$p.bag.tar"; echo "$p: verified and unpacked"
  sleep 30
done

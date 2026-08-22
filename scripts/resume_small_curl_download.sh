#!/usr/bin/env bash
set -u

URL="${1:?URL required}"
SIZE="${2:?size required}"
SHA256="${3:?sha256 required}"
DIR="${4:?parts directory required}"
OUT="${5:?output path required}"
PROXY="${6:-}"
CHUNK=1048576
mkdir -p "$DIR" "$(dirname "$OUT")"

if [[ -n "$PROXY" ]]; then
  ROUTE=(--proxy "$PROXY")
else
  ROUTE=(--noproxy '*')
fi

download_part() {
  local idx="$1"
  local start=$((idx * CHUNK))
  local end=$((start + CHUNK - 1))
  if ((end >= SIZE)); then end=$((SIZE - 1)); fi
  local expected=$((end - start + 1))
  local part="$DIR/part_$(printf '%03d' "$idx")"
  local tmp="$part.tmp"
  while true; do
    local current=0
    [[ -f "$part" ]] && current=$(stat -c '%s' "$part")
    if ((current == expected)); then return 0; fi
    if ((current > expected)); then return 1; fi
    local req_start=$((start + current))
    rm -f "$tmp"
    curl "${ROUTE[@]}" -fL --retry 10 --retry-all-errors \
      --connect-timeout 30 --max-time 300 \
      -r "$req_start-$end" -o "$tmp" "$URL" \
      >"$DIR/part_$(printf '%03d' "$idx").curl.log" 2>&1 || continue
    local got=0
    [[ -f "$tmp" ]] && got=$(stat -c '%s' "$tmp")
    local want=$((end - req_start + 1))
    if ((got != want)); then
      rm -f "$tmp"
      continue
    fi
    cat "$tmp" >> "$part"
    rm -f "$tmp"
  done
}

pids=()
for idx in $(seq 0 $(((SIZE + CHUNK - 1) / CHUNK - 1))); do
  download_part "$idx" &
  pids+=("$!")
  if ((${#pids[@]} >= 16)); then
    wait "${pids[0]}" || exit 1
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done

cat /dev/null > "$OUT"
for idx in $(seq 0 $(((SIZE + CHUNK - 1) / CHUNK - 1))); do
  part="$DIR/part_$(printf '%03d' "$idx")"
  cat "$part" >> "$OUT"
done
[[ "$(stat -c '%s' "$OUT")" == "$SIZE" ]] || exit 1
digest=$(sha256sum "$OUT" | awk '{print $1}')
printf 'sha256=%s\n' "$digest"
[[ "$digest" == "$SHA256" ]]

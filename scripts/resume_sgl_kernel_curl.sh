#!/usr/bin/env bash
set -u

USE_PROXY="${SGL_USE_PROXY:-0}"
if [[ "$USE_PROXY" == "1" ]]; then
  CURL_ROUTE=(--proxy "${SGL_PROXY:?SGL_PROXY is required when SGL_USE_PROXY=1}")
else
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  CURL_ROUTE=(--noproxy '*')
fi

URL='https://files.pythonhosted.org/packages/da/74/d804e757d7dd85aecdd534429d9335ce87b881a1601341c01a39f9b256fd/sgl_kernel-0.1.4-cp39-abi3-manylinux2014_x86_64.whl'
DIR=/root/autodl-tmp/rods-workspace/cache/sgl_kernel_parts
TOTAL=231504246
CHUNK=16777216
mkdir -p "$DIR"

resume_part() {
  local idx="$1"
  local start=$((idx * CHUNK))
  local end=$((start + CHUNK - 1))
  if ((end >= TOTAL)); then end=$((TOTAL - 1)); fi
  local expected=$((end - start + 1))
  local part="$DIR/part_$(printf '%03d' "$idx")"
  local tmp="$part.tmp"
  while true; do
    local current=0
    if [[ -f "$part" ]]; then current=$(stat -c '%s' "$part"); fi
    if ((current == expected)); then
      printf 'complete part=%s bytes=%s\n' "$idx" "$current"
      return 0
    fi
    if ((current > expected)); then
      printf 'invalid oversized part=%s bytes=%s expected=%s\n' "$idx" "$current" "$expected" >&2
      return 1
    fi
    local req_start=$((start + current))
    rm -f "$tmp"
    curl "${CURL_ROUTE[@]}" -fL --retry 8 --retry-all-errors \
      --connect-timeout 30 --max-time 900 \
      -r "$req_start-$end" -o "$tmp" "$URL" \
      >"$DIR/part_$(printf '%03d' "$idx").curl.log" 2>&1 || continue
    local got=$(stat -c '%s' "$tmp" 2>/dev/null || printf '0')
    local want=$((end - req_start + 1))
    if ((got != want)); then
      printf 'short part=%s got=%s want=%s; retrying\n' "$idx" "$got" "$want" >>"$DIR/part_$(printf '%03d' "$idx").curl.log"
      rm -f "$tmp"
      continue
    fi
    cat "$tmp" >> "$part"
    rm -f "$tmp"
  done
}

pids=()
for idx in $(seq 0 13); do
  resume_part "$idx" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

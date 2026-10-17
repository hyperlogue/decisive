#!/usr/bin/env bash
# Raw captures contain process IDs and filesystem paths; do not publish them.
set -euo pipefail
out=${1:?usage: profile_http.sh NEW_OUTPUT_DIRECTORY SERVER_PID PORT INPUT_JSONL ROUNDS}
server_pid=${2:?server pid}
port=${3:?loopback port}
input=${4:?input corpus}
rounds=${5:-60}
[[ "$server_pid" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ && "$rounds" =~ ^[0-9]+$ ]] || exit 2
perf_bin=${PERF:-perf}
client=${DECISIVE_HTTP_PROBE:-/tmp/decisive-http-probe}
mkdir "$out"
"$client" "$port" "$input" 1 1 > "$out/warmup.json"
mkfifo "$out/control" "$out/ack"
capture_pid=
cleanup() {
    if [[ -n "$capture_pid" ]] && kill -0 "$capture_pid" 2>/dev/null; then kill -INT "$capture_pid"; wait "$capture_pid" || true; fi
    rm -f "$out/control" "$out/ack"
}
trap cleanup EXIT
"$perf_bin" record --delay=-1 --control="fifo:$out/control,$out/ack" \
    -p "$server_pid" -e cpu-clock:u -F "${DECISIVE_PERF_HZ:-99}" --call-graph "${DECISIVE_PERF_CALLGRAPH:-dwarf,8192}" \
    -o "$out/perf.data" > "$out/log.txt" 2>&1 &
capture_pid=$!
exec 3> "$out/control"
exec 4< "$out/ack"
printf 'enable\n' >&3
read -r acknowledgement <&4
"$client" "$port" "$input" 1 "$rounds" > "$out/run.json"
printf 'disable\n' >&3
read -r acknowledgement <&4
kill -INT "$capture_pid"
wait "$capture_pid" || true
capture_pid=
# http_probe performs three extra warm requests INSIDE this profiling window.
# The sampled request count is therefore rounds + 3 times the corpus length.
"$perf_bin" report --stdio --no-children -g none --percent-limit 0.1 -i "$out/perf.data" > "$out/self.txt" 2> "$out/report-log.txt"
"$perf_bin" script --no-inline -i "$out/perf.data" > "$out/stacks.txt" 2>> "$out/report-log.txt"
"$perf_bin" script -G --no-inline -F comm,tid,time,event,ip,sym,dso -i "$out/perf.data" > "$out/leaves.txt" 2>> "$out/report-log.txt"

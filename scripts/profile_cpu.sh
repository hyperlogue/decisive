#!/usr/bin/env bash
set -euo pipefail
mode=${1:?usage: profile_cpu.sh stat|record NEW_OUTPUT_DIRECTORY [profile_inference arguments]}
out=${2:?new output directory required}
shift 2
perf_bin=${PERF:-perf}
driver=${DECISIVE_PROFILE_DRIVER:-target/benchmarks/profile_inference}
[[ "$mode" == stat || "$mode" == record ]] || { echo 'expected stat or record' >&2; exit 2; }
mkdir "$out"
mkfifo "$out/control" "$out/ack"
trap 'rm -f "$out/control" "$out/ack"' EXIT
common=(--delay=-1 --control="fifo:$out/control,$out/ack")
if [[ "$mode" == stat ]]; then
    "$perf_bin" stat "${common[@]}" -x, -o "$out/counters.csv" \
        -e "${DECISIVE_PERF_EVENTS:-task-clock,context-switches,cpu-migrations,page-faults,cycles:u,instructions:u}" \
        -- "$driver" "$@" --control "$out/control" --ack "$out/ack" > "$out/run.json" 2> "$out/log.txt"
else
    "$perf_bin" record "${common[@]}" -e cpu-clock:u -F "${DECISIVE_PERF_HZ:-99}" \
        --call-graph "${DECISIVE_PERF_CALLGRAPH:-dwarf,8192}" -o "$out/perf.data" \
        -- "$driver" "$@" --control "$out/control" --ack "$out/ack" > "$out/run.json" 2> "$out/log.txt"
    "$perf_bin" report --stdio --no-children -g none --percent-limit 0.1 -i "$out/perf.data" > "$out/self.txt" 2> "$out/report-log.txt"
    "$perf_bin" script --no-inline -i "$out/perf.data" > "$out/stacks.txt" 2>> "$out/report-log.txt"
    "$perf_bin" script -G --no-inline -F comm,tid,time,event,ip,sym,dso -i "$out/perf.data" > "$out/leaves.txt" 2>> "$out/report-log.txt"
fi

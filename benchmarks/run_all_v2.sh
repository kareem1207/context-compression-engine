#!/usr/bin/env bash
# Runs the v2 benchmarks end to end on local models (needs E:/llm with llama-server and the two GGUF files).
# Takes roughly two hours on a laptop CPU. Usage: bash benchmarks/run_all_v2.sh
set -u
cd "$(dirname "$0")/.."
LLAMA=${LLAMA:-/e/llm/bin/llama-server.exe}
DEEPSEEK=${DEEPSEEK:-E:/llm/models/deepseek.gguf}
GEMMA=${GEMMA:-E:/llm/models/gemma4-e4b/google_gemma-4-E4B-it-Q4_K_M.gguf}
PY=.venv/Scripts/python.exe
RUNS=${RUNS:-"fit overflow gemma"}   # choose which runs to do, e.g. RUNS="overflow gemma"
want() { [[ " $RUNS " == *" $1 "* ]]; }

serve() {  # model ctx
  taskkill //IM llama-server.exe //F > /dev/null 2>&1 || true
  sleep 3
  "$LLAMA" -m "$1" --port 8080 --ctx-size "$2" --threads 8 --no-warmup > /tmp/llama_v2.log 2>&1 &
  for _ in $(seq 1 60); do curl -s localhost:8080/health | grep -q '"ok"' && return 0; sleep 3; done
  echo "server did not start" >&2; return 1
}

want fit && serve "$DEEPSEEK" 8192 && $PY benchmarks/run_benchmark_v2.py --mode fit --label deepseek-coder-6.7b-fit --variants cce,cce_hash --conversations 3 --questions 8
want overflow && serve "$DEEPSEEK" 4096 && $PY benchmarks/run_benchmark_v2.py --mode overflow --label deepseek-coder-6.7b-overflow --variants cce,cce_hash --conversations 3 --questions 8
want gemma && serve "$GEMMA" 4096 && $PY benchmarks/run_benchmark_v2.py --mode overflow --label gemma4-e4b-overflow --variants cce,cce_hash --conversations 2 --questions 8
taskkill //IM llama-server.exe //F > /dev/null 2>&1 || true
echo ALL_DONE

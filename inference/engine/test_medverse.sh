#!/usr/bin/env bash
# MedVerse Inference Engine Test Script
# Usage: bash test_medverse.sh [start|test|stop|all]

set -e

MODEL_PATH="${MODEL_PATH:-/path/to/MedVerse-7B}"
PORT="${PORT:-30000}"
HOST="${HOST:-localhost}"
GPU="${GPU:-0}"
CONDA_ENV="${CONDA_ENV:-medverse-engine}"
LOG_FILE="/tmp/medverse_server.log"
PID_FILE="/tmp/medverse_server.pid"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Start server ──────────────────────────────────────────────────────────────
start_server() {
    info "Starting MedVerse server on GPU $GPU, port $PORT ..."
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        warn "Server already running (PID=$(cat $PID_FILE)). Skipping."
        return 0
    fi

    CUDA_VISIBLE_DEVICES=$GPU \
    conda run -n "$CONDA_ENV" --no-capture-output \
        python -m sglang.srt.entrypoints.medverse_server \
            --model-path "$MODEL_PATH" \
            --tp-size 1 \
            --port "$PORT" \
            --trust-remote-code \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    info "Server PID=$(cat $PID_FILE). Logs: $LOG_FILE"

    info "Waiting for server to be ready ..."
    for i in $(seq 1 120); do
        if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            info "Server is ready! (${i}s)"
            return 0
        fi
        sleep 1
        printf "."
    done
    echo
    error "Server did not become healthy within 120 s. Check $LOG_FILE"
    exit 1
}

# ── Stop server ───────────────────────────────────────────────────────────────
stop_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        kill -9 "$PID" 2>/dev/null && info "Killed PID=$PID" || true
        rm -f "$PID_FILE"
    fi
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "medverse_scheduler"   2>/dev/null || true
    for i in $(seq 1 15); do
        ss -tlnp 2>/dev/null | grep -q ":${PORT}\b" || { info "Port $PORT is free."; return 0; }
        sleep 1
    done
    warn "Port $PORT may still be in use."
}

# ── Single generate request ───────────────────────────────────────────────────
generate() {
    local label="$1"
    local prompt="$2"
    local max_tokens="${3:-4096}"
    local temp="${4:-0.1}"

    echo
    echo "════════════════════════════════════════════════════════"
    info "Test: $label"
    echo "════════════════════════════════════════════════════════"

    RESPONSE=$(curl -sf --max-time 300 "http://$HOST:$PORT/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"text\": $(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
            \"sampling_params\": {
                \"max_new_tokens\": $max_tokens,
                \"temperature\": $temp,
                \"stop\": [\"<|im_end|>\"]
            }
        }") || { error "curl failed (timeout or connection error)"; return 1; }

    echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
text = d.get('text', d.get('generated_text', str(d)))
print(text)
print()
meta = d.get('meta_info', {})
if meta:
    print(f'[tokens: prompt={meta.get(\"prompt_tokens\",\"?\")}, completion={meta.get(\"completion_tokens\",\"?\")}]')
"
}

# ── Build standard MedVerse prompt ───────────────────────────────────────────
medverse_prompt() {
    local user_msg="$1"
    printf '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n' "$user_msg"
}

# ── Run all tests ─────────────────────────────────────────────────────────────
run_tests() {
    info "Running MedVerse Phase I + Phase II tests ..."

    # ── Test 1: Chest pain (classic ACS) ─────────────────────────────────────
    generate "Chest Pain - ACS" \
        "$(medverse_prompt "A 45-year-old male presents with chest pain radiating to the left arm, diaphoresis, and shortness of breath for the past 2 hours. Vitals: BP 150/95, HR 110, SpO2 94%.")"

    # ── Test 2: Sepsis workup ─────────────────────────────────────────────────
    generate "Sepsis Workup" \
        "$(medverse_prompt "A 68-year-old female presents with fever 39.2°C, HR 118, RR 24, BP 88/56, confusion. She has a foley catheter placed 3 days ago. WBC 22k, lactate 3.8 mmol/L.")"

    # ── Test 3: Stroke ───────────────────────────────────────────────────────
    generate "Acute Stroke" \
        "$(medverse_prompt "A 72-year-old male with atrial fibrillation presents with sudden onset right-sided weakness and aphasia that started 90 minutes ago. Last known normal was 2 hours ago.")"

    info "All tests complete."
}

# ── Health check ─────────────────────────────────────────────────────────────
check_health() {
    if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        info "Server is healthy at http://$HOST:$PORT"
    else
        error "Server not responding at http://$HOST:$PORT"
        exit 1
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
CMD="${1:-all}"
case "$CMD" in
    start)  start_server ;;
    stop)   stop_server ;;
    test)   check_health && run_tests ;;
    health) check_health ;;
    all)
        start_server
        run_tests
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage: $0 [start|stop|test|health|logs|all]"
        echo
        echo "  start   Start the MedVerse server (background)"
        echo "  stop    Kill the server"
        echo "  test    Run inference tests (server must already be running)"
        echo "  health  Check if server is up"
        echo "  logs    Tail server logs"
        echo "  all     Start server then run tests (default)"
        exit 1
        ;;
esac

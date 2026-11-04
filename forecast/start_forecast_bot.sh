#!/usr/bin/env bash
# Start the prod forecast bot + health watchdog in tmux session "tempBot"
# (windows "bot" and "health"). Safe to re-run: a running session is left alone.
#
#   forecast/start_forecast_bot.sh          # start (or no-op if running)
#   forecast/start_forecast_bot.sh --restart
#
# Logs: $BOT_DIR/../loop_prod.log, $BOT_DIR/../health.log
set -euo pipefail

FORECAST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FORECAST_DIR/bot_env.sh"
LOG_DIR="$(dirname "$BOT_DIR")"

SESSION=tempBot

if [ "${1:-}" = "--restart" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$SESSION already running (attach: tmux attach -t $SESSION)"
    exit 0
fi

# Rotate the log once it passes 50 MB so it can't grow without bound.
rotate() {
    local f="$1"
    if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 52428800 ]; then
        mv -f "$f" "$f.1"
    fi
}

start() {
    local window="$1" log="$2" cmd="$3"
    local tmux_cmd=(new-window -t "$SESSION" -n "$window")
    tmux has-session -t "$SESSION" 2>/dev/null || tmux_cmd=(new-session -d -s "$SESSION" -n "$window")
    # Outer while-loop: if python dies (crash, OOM, bug), restart after 60 s
    # instead of leaving the window dead.
    tmux "${tmux_cmd[@]}" bash -c "
        source '$FORECAST_DIR/bot_env.sh'
        cd '$FORECAST_DIR'
        while true; do
            $(declare -f rotate); rotate '$log'
            echo \"=== [\$(date)] starting: $cmd ===\" >> '$log'
            $cmd >> '$log' 2>&1
            echo \"=== [\$(date)] exited \$?, restarting in 60 s ===\" >> '$log'
            sleep 60
        done"
    echo "started $SESSION:$window -> $log"
}

start bot "$LOG_DIR/loop_prod.log" \
    "FORECAST_ENV=slac \$FORECAST_PY -u run_forecast_loop_dev.py"
start health "$LOG_DIR/health.log" \
    "\$FORECAST_PY -u check_forecast_health.py --watch"

# Environment for the forecast bot. Sourced by start_forecast_bot.sh.
#
# Everything writable goes under BOT_DIR on the data volume, never $HOME:
# a full home dir once broke numba's cache (spectractor @njit(cache=True),
# pulled in by lsst.summit.utils) -> EFD fetch failed for 18 days unnoticed.

export BOT_DIR=/sdf/data/rubin/user/esteves/forecast/bot
export FORECAST_PY=/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python

# Pinned LSST stack paths (PYTHONPATH + LD_LIBRARY_PATH for lsst.summit.utils).
# Do NOT follow w_latest: it moves (now scipipe-14.0.0, where numpy.long breaks
# imports). Regenerate from a known-good `source ~/setup.sh` shell with:
#   { printf 'export PYTHONPATH=%q\n' "$PYTHONPATH";
#     printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"; } > $BOT_DIR/stack_env.sh
if [ ! -f "$BOT_DIR/stack_env.sh" ]; then
    echo "[FATAL] missing $BOT_DIR/stack_env.sh (see bot_env.sh for how to make it)" >&2
    return 1 2>/dev/null || exit 1
fi
source "$BOT_DIR/stack_env.sh"

# lsst.summit.utils getSite() reads $HOSTNAME to pick the EFD; bash sets it but
# only a login profile exports it, so a bare tmux/env shell gets site=UNKOWN.
export HOSTNAME="${HOSTNAME:-$(hostname)}"

mkdir -p "$BOT_DIR"/{cache,tmp,pycache}
export NUMBA_CACHE_DIR="$BOT_DIR/cache/numba"
export XDG_CACHE_HOME="$BOT_DIR/cache"
export MPLCONFIGDIR="$BOT_DIR/cache/matplotlib"
export TMPDIR="$BOT_DIR/tmp"            # cmdstanpy/prophet temp files
export PYTHONPYCACHEPREFIX="$BOT_DIR/pycache"

# See run_weathernbeats_forecast.py: avoid UCX ibverbs retry storm.
export UCX_TLS=tcp,self
export NCCL_IB_DISABLE=1
export PL_TORCH_DISTRIBUTED_BACKEND=gloo

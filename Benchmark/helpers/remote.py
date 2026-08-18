"""
helpers/remote.py
Utilities for rsync transfers and SSH remote execution between Mac (data collection)
and the Linux training server.
"""

import subprocess
import sys
from pathlib import Path
from global_var import *


# ---------------------------------------------------------------------------
# Configuration — override these or import from global_var
# ---------------------------------------------------------------------------

class RemoteConfig:
    """All connection/path details in one place. Edit to match your setup."""

    # SSH target — can use an alias from ~/.ssh/config (recommended)
    ssh_host: str = "davide@fibonacci-brairlab"          # or "user@192.168.x.x:2302"
    ssh_key:  str = "~/.ssh/fibo_key_davide"        # path to private key

    # Absolute paths on the LINUX SERVER
    server_data_root:   str = f"/home/davide/github/rl_learning/Benchmark/{CURRENT_ENV['img_dir']}"
    server_models_root: str = f"/home/davide/github/rl_learning/Benchmark/{CURRENT_ENV['models']}"
    server_script:      str = "/home/davide/github/rl_learning/Benchmark/policy/train_server.py"
    server_python:      str = "/home/davide/github/rl_learning/rl_env/bin/python"

    # Absolute paths on the MAC
    local_data_root:   str = str(Path.home() / f"Documents/github/rl_learning/Benchmark/{CURRENT_ENV['img_dir']}")
    local_models_root: str = str(Path.home() / f"Documents/github/rl_learning/Benchmark/{CURRENT_ENV['models']}")

    # rsync options
    # --archive  : recursive + preserve permissions/timestamps
    # --compress : compress during transfer (good for checkpoints)
    # --partial  : resume interrupted transfers
    # --info=progress2 : single-line progress (quiet but informative)
    rsync_opts: list = ["--archive", "--compress", "--partial"]


CFG = RemoteConfig()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess and stream its output. Raises on non-zero exit."""
    print(f"\n[remote] {label}")
    print(f"[remote] $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[remote] ERROR: '{label}' exited with code {result.returncode}", file=sys.stderr)
        raise RuntimeError(f"Remote command failed (code {result.returncode}): {label}")
    print(f"[remote] ✓ {label} done\n")


# ---------------------------------------------------------------------------
# rsync helpers
# ---------------------------------------------------------------------------

def rsync_push_data() -> None:
    """
    Push collected data from Mac → Server.

    Only the relevant env/experiment subdirectory is transferred so the
    transfer stays small even when multiple experiments exist locally.
    """
    local_src  = str(Path(CFG.local_data_root)) + "/"
    remote_dst = f"{CFG.ssh_host}:{CFG.server_data_root}"

    _run(
        ["rsync"] + CFG.rsync_opts + [
            "-e", f"ssh -i {CFG.ssh_key}",
            local_src,
            remote_dst,
        ],
        label=f"push data  {CURRENT_ENV['img_dir']}  →  server",
    )


def rsync_pull_models() -> None:
    """
    Pull trained model checkpoints from Server → Mac.

    Pulls the entire models directory for this experiment so the Mac has
    the latest VQ-VAE, LSTM, and PPO agent checkpoints.
    """
    remote_src = f"{CFG.ssh_host}:{CFG.server_models_root}/"
    local_dst  = str(Path(CFG.local_models_root)) + "/"

    Path(local_dst).mkdir(parents=True, exist_ok=True)
    _run(
        ["rsync"] + CFG.rsync_opts + [
            "-e", f"ssh -i {CFG.ssh_key}",
            remote_src,
            local_dst,
        ],
        label=f"pull models  server  →  {CURRENT_ENV['models']}",
    )


# ---------------------------------------------------------------------------
# SSH remote training trigger
# ---------------------------------------------------------------------------

def ssh_train_on_server(round_idx: int, exp_id: str) -> None:
    """
    SSH into the server and run the training script for one round.

    The call BLOCKS until training is complete (or raises on failure).
    stdout/stderr from the server are streamed directly to the local terminal.

    The server script receives --round and --exp_id so it knows which
    checkpoint to load / where to save results.
    """
    remote_cmd = (
        f"{CFG.server_python} {CFG.server_script} "
        f"--round {round_idx} "
        f"--exp_id {exp_id}"
    )

    _run(
        [
            "ssh",
            "-i", CFG.ssh_key,
            "-o", "StrictHostKeyChecking=accept-new",   # first-connect safety
            "-o", "ServerAliveInterval=60",              # keepalive for long training
            "-o", "ServerAliveCountMax=30",              # 30 min max silence
            CFG.ssh_host,
            remote_cmd,
        ],
        label=f"server training  round={round_idx}  exp={exp_id}",
    )

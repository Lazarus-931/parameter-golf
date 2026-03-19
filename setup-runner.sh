#!/bin/bash
set -euo pipefail

# =============================================================
# Setup a Mac Mini as a distributed training node.
#
# From your laptop, SSH into the Mini via Tailscale and run:
#
#   ssh user@<tailscale-ip>
#   curl -sL https://raw.githubusercontent.com/<you>/parameter-golf/main/setup-runner.sh | \
#       bash -s -- --token <RUNNER_TOKEN> --repo https://github.com/<you>/parameter-golf
#
# The runner token comes from:
#   GitHub repo → Settings → Actions → Runners → New self-hosted runner
#
# That's it. The Mini will:
#   1. Register as a GitHub Actions self-hosted runner
#   2. Install Python deps (MLX, etc.)
#   3. Pre-cache the training dataset
#   4. Setup SSH keys for distributed training between Minis
#   5. Print its Tailscale IP for you to add to cluster/hosts.txt
# =============================================================

RUNNER_TOKEN=""
REPO_URL=""
RUNNER_DIR="$HOME/actions-runner"
VENV_DIR="$HOME/parameter-golf-venv"
DATA_DIR="$HOME/parameter-golf-data"
WORK_DIR="$HOME/parameter-golf-work"
SSH_KEY="$HOME/.ssh/parameter_golf_cluster"

usage() {
  echo "Usage: $0 --token <GITHUB_RUNNER_TOKEN> --repo <REPO_URL>"
  echo ""
  echo "  --token   Runner registration token from GitHub Settings → Actions → Runners"
  echo "  --repo    Full GitHub repo URL (e.g. https://github.com/youruser/parameter-golf)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --token) RUNNER_TOKEN="$2"; shift 2 ;;
    --repo)  REPO_URL="$2"; shift 2 ;;
    *) usage ;;
  esac
done

if [[ -z "$RUNNER_TOKEN" || -z "$REPO_URL" ]]; then
  usage
fi

RUNNER_NAME="$(hostname -s)"

echo "=== Setting up Mac Mini: $RUNNER_NAME ==="
echo ""

# ---------------------------------------------------------
# 1. GitHub Actions self-hosted runner
# ---------------------------------------------------------
echo "--- [1/5] Installing GitHub Actions runner ---"
mkdir -p "$RUNNER_DIR" && cd "$RUNNER_DIR"

if [[ ! -f ./run.sh ]]; then
  LATEST=$(curl -s https://api.github.com/repos/actions/runner/releases/latest \
    | grep -o '"tag_name": "v[^"]*"' | head -1 | cut -d'"' -f4)
  VERSION="${LATEST#v}"
  echo "Downloading runner ${VERSION} for macOS ARM64..."
  curl -sL "https://github.com/actions/runner/releases/download/${LATEST}/actions-runner-osx-arm64-${VERSION}.tar.gz" \
    -o runner.tar.gz
  tar xzf runner.tar.gz
  rm runner.tar.gz
fi

./config.sh \
  --url "$REPO_URL" \
  --token "$RUNNER_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "self-hosted,macOS,ARM64,parameter-golf" \
  --work _work \
  --replace

echo "--- Installing runner as launchd service ---"
./svc.sh install 2>/dev/null || true
./svc.sh start 2>/dev/null || true

# ---------------------------------------------------------
# 2. Python venv + MLX dependencies
# ---------------------------------------------------------
echo ""
echo "--- [2/5] Setting up Python environment ---"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm -q

# ---------------------------------------------------------
# 3. Pre-cache dataset
# ---------------------------------------------------------
echo ""
echo "--- [3/5] Pre-caching FineWeb dataset ---"
if [[ ! -d "$DATA_DIR/fineweb10B_sp1024" ]]; then
  mkdir -p "$WORK_DIR"
  TMPDIR=$(mktemp -d)
  git clone --depth 1 "$REPO_URL" "$TMPDIR/repo"
  cd "$TMPDIR/repo"
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
  mkdir -p "$DATA_DIR"
  mv data/datasets/fineweb10B_sp1024 "$DATA_DIR/"
  cp -r data/tokenizers "$DATA_DIR/" 2>/dev/null || true
  rm -rf "$TMPDIR"
  cd "$HOME"
else
  echo "Dataset already cached at $DATA_DIR"
fi

# ---------------------------------------------------------
# 4. SSH keys for distributed training (mlx.launch uses SSH)
# ---------------------------------------------------------
echo ""
echo "--- [4/5] Setting up SSH for distributed training ---"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"

if [[ ! -f "$SSH_KEY" ]]; then
  ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "parameter-golf-cluster"
fi

# Authorize this key so other Minis can SSH in
if ! grep -qf "$SSH_KEY.pub" "$HOME/.ssh/authorized_keys" 2>/dev/null; then
  cat "$SSH_KEY.pub" >> "$HOME/.ssh/authorized_keys"
  chmod 600 "$HOME/.ssh/authorized_keys"
fi

# Use this key by default for cluster connections
if ! grep -q "parameter_golf_cluster" "$HOME/.ssh/config" 2>/dev/null; then
  cat >> "$HOME/.ssh/config" <<SSHEOF

# Parameter Golf cluster — auto-generated
Host 100.* fd7a:*
  IdentityFile $SSH_KEY
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
SSHEOF
  chmod 600 "$HOME/.ssh/config"
fi

echo "SSH key generated at $SSH_KEY"
echo ""
echo "IMPORTANT: Copy this public key to ALL other Minis in the cluster:"
echo ""
echo "  $(cat "$SSH_KEY.pub")"
echo ""
echo "On each other Mini, run:"
echo "  echo '$(cat "$SSH_KEY.pub")' >> ~/.ssh/authorized_keys"
echo ""

# ---------------------------------------------------------
# 5. Detect Tailscale IP
# ---------------------------------------------------------
echo "--- [5/5] Detecting Tailscale IP ---"
TS_IP=""
if command -v tailscale &>/dev/null; then
  TS_IP=$(tailscale ip -4 2>/dev/null || echo "")
fi

echo ""
echo "========================================"
echo " SETUP COMPLETE: $RUNNER_NAME"
echo "========================================"
echo ""
echo " Runner:  $RUNNER_DIR (registered + running)"
echo " Venv:    $VENV_DIR"
echo " Dataset: $DATA_DIR"
echo ""
if [[ -n "$TS_IP" ]]; then
  echo " Tailscale IP: $TS_IP"
  echo ""
  echo " NEXT STEP: Add this IP to cluster/hosts.txt in the repo:"
  echo ""
  echo "   echo '$TS_IP' >> cluster/hosts.txt"
  echo "   git add cluster/hosts.txt && git commit -m 'Add $RUNNER_NAME to cluster' && git push"
  echo ""
else
  echo " WARNING: Could not detect Tailscale IP."
  echo " Find it with: tailscale ip -4"
  echo " Then add it to cluster/hosts.txt in the repo."
fi
echo " Once hosts.txt is updated, every push triggers distributed training"
echo " across ALL listed Minis automatically."
echo "========================================"

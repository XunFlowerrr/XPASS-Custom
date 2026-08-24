#!/usr/bin/env bash
# ==============================================================================
# scripts/bootstrap_studio.sh <commit-sha>
#
# Prepares a Lightning Studio to run a fold. Safe to run on every start.
#
# Only $HOME survives a Studio stop/start. Anything installed into /usr/bin is
# gone on the next boot -- rclone installed there once and vanished -- and .venv
# is a symlink into /system/conda, which goes with it. So this script installs
# tools under $HOME and rebuilds the environment unconditionally; the steps that
# are already satisfied cost a second each.
#
# Exit codes:
#   0  ready to run
#   2  dataset or weights missing  -> caller should run setup_data.sh
#   3  rclone not configured       -> needs an interactive OAuth flow
# ==============================================================================
set -euo pipefail

COMMIT="${1:?usage: bootstrap_studio.sh <commit-sha>}"
REPO_DIR="${REPO_DIR:-$HOME/XPASS-Custom}"
REMOTE_NAME="${RCLONE_REMOTE:-Google Drive}"
FOLDER_ID="${GDRIVE_FOLDER_ID:-1WfoO2zszob9rAe7ya8CkmBt6Ci7p070L}"

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
mkdir -p "$HOME/bin" "$HOME/fleet/status"

say() { echo "▶ $*"; }

# ─── 1. Reclaim disk ──────────────────────────────────────────────────────────
# setup_data.sh used to leave its archives behind: 3x5 GB of model zips plus a
# 1.7 GB sample zip, all pure duplication once extracted. On the source studio
# this halves what every duplicate has to copy.
say "Reclaiming disk from extracted archives"
for d in "$HOME" "$REPO_DIR"; do
  rm -f "$d"/sample.zip "$d"/xpass_v4_essentials.zip \
        "$d"/art_models_v4.zip "$d"/fashion_models_v4.zip "$d"/scenery_models_v4.zip 2>/dev/null || true
done

# ─── 2. rclone into $HOME/bin ─────────────────────────────────────────────────
# Not `curl … | sudo bash`: that installs to /usr/bin, which does not persist.
if command -v rclone >/dev/null 2>&1 && [ -x "$HOME/bin/rclone" ]; then
  say "rclone present ($(rclone version | head -1))"
else
  say "Installing rclone into \$HOME/bin"
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/rclone.zip" https://downloads.rclone.org/rclone-current-linux-amd64.zip
  unzip -q -j -o "$tmp/rclone.zip" '*/rclone' -d "$HOME/bin"
  chmod +x "$HOME/bin/rclone"
  rm -rf "$tmp"
  say "rclone installed ($(rclone version | head -1))"
fi

# ─── 3. uv into $HOME/.local/bin ──────────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  say "uv present ($(uv --version))"
else
  say "Installing uv into \$HOME/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$HOME/.local/bin" sh
fi

# ─── 4. PATH for interactive shells ───────────────────────────────────────────
# Every command the orchestrator sends carries its own export, because
# Studio.run() may not source .bashrc. This is for humans who ssh in.
LINE='export PATH="$HOME/bin:$HOME/.local/bin:$PATH"'
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  [ -f "$rc" ] || continue
  grep -qF "$LINE" "$rc" || echo "$LINE" >> "$rc"
done

# ─── 5. rclone credentials ────────────────────────────────────────────────────
# The OAuth flow is interactive and must not be automated. The config lives in
# $HOME, so it does survive restarts -- but a duplicated Studio may not have it.
if [ ! -f "$HOME/.config/rclone/rclone.conf" ]; then
  echo "❌ ~/.config/rclone/rclone.conf is missing." >&2
  echo "   Copy it from a configured machine:" >&2
  echo "     scp ~/.config/rclone/rclone.conf <studio>:$HOME/.config/rclone/rclone.conf" >&2
  echo "   (scp on Lightning does NOT expand ~ -- use the absolute path.)" >&2
  exit 3
fi
chmod 700 "$HOME/.config/rclone" 2>/dev/null || true
chmod 600 "$HOME/.config/rclone/rclone.conf" 2>/dev/null || true
if rclone lsd "${REMOTE_NAME}:" --drive-root-folder-id "$FOLDER_ID" >/dev/null 2>&1; then
  say "rclone remote '${REMOTE_NAME}' authenticated"
else
  echo "❌ rclone remote '${REMOTE_NAME}' did not authenticate." >&2
  echo "   The OAuth token may have expired; re-run 'rclone config' interactively." >&2
  exit 3
fi

# ─── 6. Pin the repo ──────────────────────────────────────────────────────────
# Detached HEAD on an exact SHA, so every machine in the fleet runs identical code.
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "❌ No git repository at $REPO_DIR" >&2
  exit 2
fi
say "Pinning $REPO_DIR to $COMMIT"
git -C "$REPO_DIR" fetch --all --prune --quiet
git -C "$REPO_DIR" checkout --quiet --detach "$COMMIT"
say "HEAD is now $(git -C "$REPO_DIR" rev-parse --short HEAD)"

# ─── 7. Python environment ────────────────────────────────────────────────────
# Unconditional: .venv points into /system/conda and is gone after a restart.
say "Syncing Python environment"
( cd "$REPO_DIR" && uv sync --quiet )

# ─── 8. Verify data ───────────────────────────────────────────────────────────
# Same probes run_all.sh's check_dataset uses. A non-zero exit tells the
# orchestrator this machine needs the setup_data.sh fallback.
missing=0
for probe in "Dataset/sample/art_extracted/art" \
             "Dataset/sample/fashion_extracted/fashion" \
             "Dataset/sample/scenery_extracted/scenery_image" \
             "Dataset/split/v4_fold1/art" \
             "models_pth/v4_fold1/art"; do
  if [ ! -e "$REPO_DIR/$probe" ]; then
    echo "   missing: $probe" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "❌ Dataset or weights incomplete at $REPO_DIR" >&2
  exit 2
fi
say "Dataset and weights verified"

# ─── 9. Marker ────────────────────────────────────────────────────────────────
# An audit record, not a short-circuit: steps 2, 3 and 7 must re-run on every
# start regardless, because the root filesystem was reset underneath them.
{
  echo "commit=$COMMIT"
  echo "rclone=$(rclone version | head -1)"
  echo "uv=$(uv --version)"
  echo "at=$(date -Is)"
} > "$HOME/.xpass_bootstrap_ok"

echo "✅ Studio ready ($(nproc) vCPU, $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU'))"

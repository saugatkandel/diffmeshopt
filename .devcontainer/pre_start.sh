#!/bin/bash

# --- 1. SILENCE VS CODE LOCATOR (Machine Level) ---
# We use /root/ because the RunPod PyTorch image runs as root.
VSCODE_MACHINE_DIR="/root/.vscode-server/data/Machine"
mkdir -p "$VSCODE_MACHINE_DIR"

# Note: Using 'js' locator and setting persistent indices to true.
# This prevents the "Conda metadata" error and speeds up future boots.
cat <<EOF > "$VSCODE_MACHINE_DIR/settings.json"
{
    "python.locator": "js",
    "python.condaPath": "",
    "python.defaultInterpreterPath": "/workspace/diffmeshopt/.pixi/envs/default/bin/python",
    "python.analysis.persistAllIndices": true
}
EOF

# --- 2. PIXI INSTALL OPTIMIZATION ---
if [ -d "/workspace/diffmeshopt" ]; then
    cd /workspace/diffmeshopt
    if [ -f "pixi.toml" ]; then
        echo "📦 Syncing Pixi Environment..."
        # --locked: Faster than standard install because it assumes the lockfile is correct.
        # --all: Ensures all environments are ready for both Python and Jupyter.
        pixi install --all --locked --skip pytorch3d
    fi
fi

echo "🚀 Pre-start configuration complete."
#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_to_github.sh  —  One-click GitHub deployment for PhyLSTM
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── 0. Prerequisites check ───────────────────────────────────────────────────
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found. Install git first."; exit 1; }
command -v gh  >/dev/null 2>&1 && HAS_GH=true || HAS_GH=false

echo ""
echo "═══════════════════════════════════════════════"
echo "  PhyLSTM — GitHub Deployment Script"
echo "═══════════════════════════════════════════════"

# ── 1. Collect user inputs ───────────────────────────────────────────────────
read -rp "GitHub username          : " GH_USER
read -rp "Repository name          [PhyLSTM]: " REPO_NAME
REPO_NAME=${REPO_NAME:-PhyLSTM}
read -rp "Repository visibility    (public/private) [public]: " VISIBILITY
VISIBILITY=${VISIBILITY:-public}
read -rp "Git email (for commits)  : " GIT_EMAIL
read -rp "Git name  (for commits)  : " GIT_NAME

# ── 2. Configure git identity ────────────────────────────────────────────────
git config --global user.email "$GIT_EMAIL"
git config --global user.name  "$GIT_NAME"

# ── 3. Init local repo ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".git" ]; then
    git init
    echo "  git init done."
fi

mkdir -p figures
git add -A
git commit -m "Initial commit: PhyLSTM physics-guided sequential learning" \
    --allow-empty

# ── 4. Create GitHub repo & push ─────────────────────────────────────────────
if $HAS_GH; then
    echo ""
    echo "Using GitHub CLI (gh) to create the repository …"
    gh repo create "$REPO_NAME" \
        --"$VISIBILITY" \
        --description "PhyLSTM: Physics-Guided Sequential Learning for Ecological Restoration Decision Stability" \
        --source=. \
        --remote=origin \
        --push
    echo ""
    echo "✅  Repository created and pushed via gh CLI."
    echo "    URL: https://github.com/$GH_USER/$REPO_NAME"
else
    echo ""
    echo "GitHub CLI (gh) not found — using manual git remote."
    echo ""
    echo "Step 1: Create a new EMPTY repository on GitHub:"
    echo "        https://github.com/new"
    echo "        Name : $REPO_NAME"
    echo "        Visibility: $VISIBILITY"
    echo "        ⚠️  Do NOT add README, .gitignore, or license (already included)."
    echo ""
    read -rp "Press ENTER once the GitHub repo is created …"

    REMOTE_URL="https://github.com/$GH_USER/$REPO_NAME.git"
    git remote remove origin 2>/dev/null || true
    git remote add origin "$REMOTE_URL"

    git branch -M main
    git push -u origin main

    echo ""
    echo "✅  Pushed to: $REMOTE_URL"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  Deployment complete!"
echo "  Repository: https://github.com/$GH_USER/$REPO_NAME"
echo ""
echo "  Clone with:"
echo "    git clone https://github.com/$GH_USER/$REPO_NAME"
echo "═══════════════════════════════════════════════"

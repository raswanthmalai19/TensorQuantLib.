#!/bin/bash
set -e

echo "🔄 Starting git history rewrite..."
echo "This will replace all 'claude' commits with 'raswanthmalai19'"
echo ""

cd /Users/raswanthmalaisamy/Desktop/library

# Create backup
echo "📦 Creating backup of main branch..."
git branch backup-main main

# Rewrite history - replace claude with raswanthmalai19
echo "✏️ Rewriting commit history..."
git filter-branch -f --env-filter '
if [ "$GIT_AUTHOR_NAME" = "claude" ] || [ "$GIT_COMMITTER_NAME" = "claude" ]; then
    export GIT_AUTHOR_NAME="raswanthmalai19"
    export GIT_COMMITTER_NAME="raswanthmalai19"
fi
if [ "$GIT_AUTHOR_EMAIL" = "" ] || [ "$GIT_AUTHOR_EMAIL" = "claude" ]; then
    export GIT_AUTHOR_EMAIL="raswanthmalai19@users.noreply.github.com"
    export GIT_COMMITTER_EMAIL="raswanthmalai19@users.noreply.github.com"
fi
' -- --all

# Force push to GitHub
echo "📤 Pushing updated history to GitHub..."
git push origin --force-with-lease main
git push origin --force-with-lease --tags

# Verify
echo ""
echo "✅ Done! Checking recent commits..."
git log --oneline -10

echo ""
echo "✅ History rewritten successfully!"
echo "📌 Backup branch 'backup-main' created if needed"

#!/bin/bash

echo "=== Contextual Bandit - Push to GitHub ==="
echo "==========================================="

# Navigate to script directory
cd "$(dirname "$0")"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed! Please install git first."
    exit 1
fi

echo "✅ Git is installed"

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add remote if not already added
if ! git remote get-url origin &> /dev/null; then
    echo "Adding GitHub remote..."
    git remote add origin https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm.git
fi

# Check status
echo ""
git status

# Stage all files
echo ""
echo "Staging files..."
git add .

# Commit
echo ""
read -p "Enter commit message (default: 'Update project'): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Update project"
fi

git commit -m "$commit_msg"

# Check branch name
current_branch=$(git branch --show-current)
echo ""
echo "Current branch: $current_branch"

# Push
echo ""
echo "Pushing to GitHub..."
git push -u origin "$current_branch"

echo ""
echo "✅ Done! Check your repository at:"
echo "https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm"

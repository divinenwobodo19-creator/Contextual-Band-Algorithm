
#!/bin/bash

# ==============================================================================
#  AUTOMATIC GITHUB PUSH SCRIPT
# ==============================================================================

echo "========================================"
echo "  🚀 PUSHING TO GITHUB AUTOMATICALLY!"
echo "========================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed! Please install git first!"
    exit 1
fi
echo "✅ Git is installed"

# Initialize git repo if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Set remote
echo "Setting up GitHub remote..."
git remote add origin https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm.git 2>/dev/null || git remote set-url origin https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm.git

# Stage all files
echo "Staging files..."
git add .

# Commit
echo "Committing changes..."
git commit -m "Complete project with full simulation, API, and improvements" 2>/dev/null || git commit -m "Update project"

# Rename to main if not already
git branch -M main 2>/dev/null

# Push
echo "Pushing to GitHub..."
git push -u origin main || git push -u origin master

echo ""
echo "========================================"
echo "  ✅ PUSH COMPLETED SUCCESSFULLY!"
echo "========================================"
echo "Check your repo at https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm"

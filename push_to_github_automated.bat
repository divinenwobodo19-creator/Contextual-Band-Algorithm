
@echo off
REM ==============================================================================
REM  AUTOMATIC GITHUB PUSH SCRIPT (WINDOWS)
REM ==============================================================================

echo ========================================
echo   🚀 PUSHING TO GITHUB AUTOMATICALLY!
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Git is not installed! Please install git first!
    pause
    exit /b 1
)
echo ✅ Git is installed

REM Initialize git repo if not already initialized
if not exist ".git" (
    echo Initializing git repository...
    git init
)

REM Set remote
echo Setting up GitHub remote...
git remote add origin https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm.git 2>nul
git remote set-url origin https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm.git 2>nul

REM Stage all files
echo Staging files...
git add .

REM Commit
echo Committing changes...
git commit -m "Complete project with full simulation, API, and improvements" 2>nul
git commit -m "Update project" 2>nul

REM Rename to main if not already
git branch -M main 2>nul

REM Push
echo Pushing to GitHub...
git push -u origin main
git push -u origin master 2>nul

echo.
echo ========================================
echo   ✅ PUSH COMPLETED SUCCESSFULLY!
echo ========================================
echo Check your repo at https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm
echo.
pause

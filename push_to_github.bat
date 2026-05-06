@echo off
echo === Contextual Bandit - Push to GitHub ===
echo ===========================================
echo.

cd /d "%~dp0"

where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Git is not installed! Please install git first.
    pause
    exit /b 1
)

echo ✅ Git is installed

if not exist ".git" (
    echo Initializing git repository...
    git init
)

git remote get-url origin >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Adding GitHub remote...
    git remote add origin https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm.git
)

echo.
git status

echo.
echo Staging files...
git add .

echo.
set /p commit_msg="Enter commit message (default: 'Update project'): "
if "%commit_msg%"=="" set commit_msg="Update project"

git commit -m "%commit_msg%"

for /f "tokens=*" %%i in ('git branch --show-current') do set current_branch=%%i
echo.
echo Current branch: %current_branch%

echo.
echo Pushing to GitHub...
git push -u origin %current_branch%

echo.
echo ✅ Done! Check your repository at:
echo https://github.com/divinenwobodo19-creator/Contextual-Band-Algorithm
echo.
pause

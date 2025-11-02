# Competition Task Setup Script for Windows
# Requires PowerShell 5.0 or higher

$ErrorActionPreference = "Stop"

Write-Host "`n[SETUP] Competition Task Setup for Windows" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

# Check PowerShell version
$psVersion = $PSVersionTable.PSVersion.Major
if ($psVersion -lt 5) {
    Write-Host "[ERROR] PowerShell 5.0 or higher is required. You have version $psVersion" -ForegroundColor Red
    exit 1
}

# Function to check if command exists
function Test-CommandExists {
    param($command)
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# Step 1: Install pyenv-win
Write-Host "[STEP 1] Checking pyenv-win installation..." -ForegroundColor Yellow

if (Test-CommandExists pyenv) {
    Write-Host "[OK] pyenv-win is already installed" -ForegroundColor Green
    pyenv --version
} else {
    Write-Host "[INSTALL] Installing pyenv-win..." -ForegroundColor Yellow
    
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"
        & "./install-pyenv-win.ps1"
        Remove-Item "./install-pyenv-win.ps1" -ErrorAction SilentlyContinue
        
        # Add to PATH for current session
        $env:PYENV = "$env:USERPROFILE\.pyenv\pyenv-win"
        $env:PYENV_ROOT = "$env:USERPROFILE\.pyenv\pyenv-win"
        $env:PYENV_HOME = "$env:USERPROFILE\.pyenv\pyenv-win"
        $env:PATH = "$env:PYENV\bin;$env:PYENV\shims;$env:PATH"
        
        Write-Host "[OK] pyenv-win installed successfully" -ForegroundColor Green
        Write-Host "[WARNING] Please restart your terminal after this script completes" -ForegroundColor Yellow
        
    } catch {
        Write-Host "[ERROR] Failed to install pyenv-win: $_" -ForegroundColor Red
        Write-Host "Please install manually from: https://github.com/pyenv-win/pyenv-win" -ForegroundColor Yellow
        exit 1
    }
}

# Step 2: Install Python 3.11.2
Write-Host "`n[STEP 2] Installing Python 3.11.2..." -ForegroundColor Yellow

try {
    # Refresh environment
    $env:PYENV = "$env:USERPROFILE\.pyenv\pyenv-win"
    $env:PYENV_ROOT = "$env:USERPROFILE\.pyenv\pyenv-win"
    $env:PYENV_HOME = "$env:USERPROFILE\.pyenv\pyenv-win"
    $env:PATH = "$env:PYENV\bin;$env:PYENV\shims;$env:PATH"
    
    $installedVersions = pyenv versions 2>&1 | Out-String
    if ($installedVersions -match "3.11.2") {
        Write-Host "[OK] Python 3.11.2 is already installed" -ForegroundColor Green
    } else {
        Write-Host "[INSTALL] Installing Python 3.11.2 (this may take a few minutes)..." -ForegroundColor Yellow
        pyenv install 3.11.2
        Write-Host "[OK] Python 3.11.2 installed successfully" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERROR] Failed to install Python 3.11.2: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Create virtual environment with Python 3.11.2
Write-Host "`n[STEP 3] Creating virtual environment with Python 3.11.2..." -ForegroundColor Yellow

if (Test-Path "venv") {
    Write-Host "[WARNING] Virtual environment already exists, removing..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

try {
    # Use the specific Python 3.11.2 from pyenv to create the venv
    $python311Path = "$env:USERPROFILE\.pyenv\pyenv-win\versions\3.11.2\python.exe"
    
    if (-not (Test-Path $python311Path)) {
        Write-Host "[ERROR] Python 3.11.2 executable not found at: $python311Path" -ForegroundColor Red
        exit 1
    }
    
    & $python311Path -m venv venv
    Write-Host "[OK] Virtual environment created with Python 3.11.2" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to create virtual environment: $_" -ForegroundColor Red
    exit 1
}

# Step 4: Activate and install dependencies
Write-Host "`n[STEP 4] Installing dependencies..." -ForegroundColor Yellow

try {
    & ".\venv\Scripts\Activate.ps1"
    
    # Verify Python version in venv
    $venvPythonVersion = python --version
    Write-Host "Virtual environment Python version: $venvPythonVersion" -ForegroundColor Gray
    
    Write-Host "Upgrading pip..." -ForegroundColor Gray
    python -m pip install --upgrade pip --quiet
    
    if (Test-Path "requirements.txt") {
        Write-Host "Installing requirements..." -ForegroundColor Gray
        pip install -r requirements.txt
        Write-Host "[OK] Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] No requirements.txt found, skipping dependency installation" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[ERROR] Failed to install dependencies: $_" -ForegroundColor Red
    exit 1
}

# Step 5: Verify setup
Write-Host "`n[STEP 5] Verifying setup..." -ForegroundColor Yellow

if (Test-Path "check_setup.py") {
    try {
        python check_setup.py
    } catch {
        Write-Host "[WARNING] Setup verification failed, but environment is created" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARNING] check_setup.py not found, skipping verification" -ForegroundColor Yellow
    Write-Host "Python version in virtual environment:" -ForegroundColor Gray
    python --version
}

# Final instructions
Write-Host "`n[SUCCESS] Setup Complete!" -ForegroundColor Green
Write-Host "==================`n" -ForegroundColor Green
Write-Host "Virtual environment created with Python 3.11.2" -ForegroundColor Cyan
Write-Host "Your system default Python version remains unchanged.`n" -ForegroundColor Cyan
Write-Host "To activate the environment, run:" -ForegroundColor Cyan
Write-Host "    .\venv\Scripts\Activate.ps1`n" -ForegroundColor White
Write-Host "[IMPORTANT] If this was a fresh pyenv installation:" -ForegroundColor Yellow
Write-Host "    1. Close this terminal" -ForegroundColor Yellow
Write-Host "    2. Open a new terminal" -ForegroundColor Yellow
Write-Host "    3. Navigate to this directory" -ForegroundColor Yellow
Write-Host "    4. Run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
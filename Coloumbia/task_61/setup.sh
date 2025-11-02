#!/bin/bash

# Competition Task Setup Script for macOS/Linux
# Requires: bash, curl

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_NAME="macOS"
elif [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS_NAME="Linux ($ID)"
else
    OS_NAME="Linux"
fi

echo -e "${CYAN}\n[SETUP] Competition Task Setup for $OS_NAME${NC}"
echo -e "${CYAN}========================================${NC}\n"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Install prerequisites (macOS: Homebrew, Linux: build tools)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}[STEP 1] Checking Homebrew installation...${NC}"
    
    if command_exists brew; then
        echo -e "${GREEN}[OK] Homebrew is already installed${NC}"
    else
        echo -e "${YELLOW}[INSTALL] Installing Homebrew...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ $(uname -m) == 'arm64' ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        
        echo -e "${GREEN}[OK] Homebrew installed successfully${NC}"
    fi
else
    echo -e "${YELLOW}[STEP 1] Installing system dependencies...${NC}"
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        
        case $ID in
            ubuntu|debian|pop)
                sudo apt-get update -qq
                sudo apt-get install -y \
                    build-essential libssl-dev zlib1g-dev libbz2-dev \
                    libreadline-dev libsqlite3-dev curl git libncursesw5-dev \
                    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
                    >/dev/null 2>&1
                ;;
            fedora|rhel|centos)
                sudo dnf groupinstall -y "Development Tools" >/dev/null 2>&1
                sudo dnf install -y zlib-devel bzip2 bzip2-devel readline-devel \
                    sqlite sqlite-devel openssl-devel tk-devel libffi-devel \
                    xz-devel curl git >/dev/null 2>&1
                ;;
            arch|manjaro)
                sudo pacman -S --noconfirm base-devel openssl zlib xz tk curl git >/dev/null 2>&1
                ;;
        esac
    fi
    
    echo -e "${GREEN}[OK] System dependencies installed${NC}"
fi

# Step 2: Install pyenv
echo -e "\n${YELLOW}[STEP 2] Checking pyenv installation...${NC}"

if command_exists pyenv; then
    echo -e "${GREEN}[OK] pyenv is already installed${NC}"
    pyenv --version
else
    echo -e "${YELLOW}[INSTALL] Installing pyenv...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install pyenv
    else
        curl https://pyenv.run | bash
    fi
    
    # Determine shell configuration file
    if [[ "$SHELL" == */zsh ]]; then
        SHELL_CONFIG="$HOME/.zshrc"
    else
        SHELL_CONFIG="$HOME/.bashrc"
    fi
    
    # Add pyenv to shell configuration
    echo -e "${YELLOW}Configuring shell...${NC}"
    {
        echo ''
        echo '# pyenv configuration'
        echo 'export PYENV_ROOT="$HOME/.pyenv"'
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
        echo 'eval "$(pyenv init --path)"'
        echo 'eval "$(pyenv init -)"'
    } >> "$SHELL_CONFIG"
    
    # Load pyenv for current session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    
    echo -e "${GREEN}[OK] pyenv installed successfully${NC}"
    echo -e "${YELLOW}[WARNING] Shell configuration updated. Changes will take effect in new terminals.${NC}"
fi

# Step 3: Install Python 3.11.2
echo -e "\n${YELLOW}[STEP 3] Installing Python 3.11.2...${NC}"

if pyenv versions | grep -q "3.11.2"; then
    echo -e "${GREEN}[OK] Python 3.11.2 is already installed${NC}"
else
    echo -e "${YELLOW}[INSTALL] Installing Python 3.11.2 (this may take a few minutes)...${NC}"
    pyenv install 3.11.2
    echo -e "${GREEN}[OK] Python 3.11.2 installed successfully${NC}"
fi

# Step 4: Create virtual environment with Python 3.11.2
echo -e "\n${YELLOW}[STEP 4] Creating virtual environment with Python 3.11.2...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}[WARNING] Virtual environment already exists, removing...${NC}"
    rm -rf venv
fi

# Use the specific Python 3.11.2 from pyenv to create the venv
PYTHON_311_PATH="$HOME/.pyenv/versions/3.11.2/bin/python"

if [ ! -f "$PYTHON_311_PATH" ]; then
    echo -e "${RED}[ERROR] Python 3.11.2 executable not found at: $PYTHON_311_PATH${NC}"
    exit 1
fi

"$PYTHON_311_PATH" -m venv venv
echo -e "${GREEN}[OK] Virtual environment created with Python 3.11.2${NC}"

# Step 5: Activate and install dependencies
echo -e "\n${YELLOW}[STEP 5] Installing dependencies...${NC}"

source venv/bin/activate

# Verify Python version in venv
VENV_PYTHON_VERSION=$(python --version)
echo -e "${NC}Virtual environment Python version: $VENV_PYTHON_VERSION${NC}"

echo "Upgrading pip..."
pip install --upgrade pip --quiet

if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo -e "${GREEN}[OK] Dependencies installed successfully${NC}"
else
    echo -e "${YELLOW}[WARNING] No requirements.txt found, skipping dependency installation${NC}"
fi

# Step 6: Verify setup
echo -e "\n${YELLOW}[STEP 6] Verifying setup...${NC}"

if [ -f "check_setup.py" ]; then
    python check_setup.py || echo -e "${YELLOW}[WARNING] Setup verification failed, but environment is created${NC}"
else
    echo -e "${YELLOW}[WARNING] check_setup.py not found, skipping verification${NC}"
    echo "Python version in virtual environment:"
    python --version
fi

# Final instructions
echo -e "\n${GREEN}[SUCCESS] Setup Complete!${NC}"
echo -e "${GREEN}==================${NC}\n"
echo -e "${CYAN}Virtual environment created with Python 3.11.2${NC}"
echo -e "${CYAN}Your system default Python version remains unchanged.${NC}\n"
echo -e "${CYAN}To activate the environment, run:${NC}"
echo -e "${NC}    source venv/bin/activate${NC}\n"
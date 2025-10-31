#!/usr/bin/env bash
#
# GPU COVID ABM Runner with tmux
# Usage: ./run_gpu_sweep.sh [runs] [agents]
# Example: ./run_gpu_sweep.sh 50 10000
#

set -e  # Exit on error

# Configuration
RUNS=${1:-50}
AGENTS=${2:-10000}
SESSION_NAME="gpu_sweep_${RUNS}r_${AGENTS}a"
SCRIPT_NAME="covid_abm_gpu_claude.py"
LOG_DIR="logs"
ENV_DIR="env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo -e "${RED}Error: $SCRIPT_NAME not found!${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$ENV_DIR" ]; then
    echo -e "${RED}Error: Virtual environment '$ENV_DIR' not found!${NC}"
    echo "Create it with: python -m venv env"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Warning: tmux session '$SESSION_NAME' already exists!${NC}"
    echo "Options:"
    echo "  1) Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2) Kill and restart: tmux kill-session -t $SESSION_NAME"
    read -p "Kill existing session and restart? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo -e "${GREEN}Killed existing session${NC}"
    else
        echo "Exiting..."
        exit 0
    fi
fi

# Generate log filename
LOG_FILE="$LOG_DIR/sweep_${RUNS}r_${AGENTS}a_$(date +%F_%H-%M-%S).log"

# Print configuration
echo -e "${GREEN}=== GPU COVID ABM Sweep ===${NC}"
echo "Runs:       $RUNS"
echo "Agents:     $AGENTS"
echo "Session:    $SESSION_NAME"
echo "Log file:   $LOG_FILE"
echo "Backend:    $(python -c 'import jax; print(jax.default_backend())' 2>/dev/null || echo 'unknown')"
echo ""

# Create tmux session with improved command
tmux new-session -s "$SESSION_NAME" -d bash -c "
    set -e
    echo '=== Starting GPU sweep at \$(date) ==='
    echo 'Activating virtual environment...'
    source $ENV_DIR/bin/activate
    
    echo 'Python: \$(which python)'
    echo 'JAX version: \$(python -c \"import jax; print(jax.__version__)\" 2>/dev/null || echo \"not installed\")'
    echo 'JAX backend: \$(python -c \"import jax; print(jax.default_backend())\" 2>/dev/null || echo \"unknown\")'
    echo ''
    
    echo 'Starting simulation...'
    python -u $SCRIPT_NAME sweep $RUNS $AGENTS 2>&1 | tee -a $LOG_FILE
    
    EXIT_CODE=\$?
    echo ''
    echo '=== Finished at \$(date) with exit code '\$EXIT_CODE' ==='
    
    if [ \$EXIT_CODE -eq 0 ]; then
        echo -e '\n${GREEN}✓ SUCCESS: Check results in gpu_sweep_results.csv${NC}'
    else
        echo -e '\n${RED}✗ FAILED: Check log file for errors${NC}'
    fi
    
    echo 'Press ENTER to close this session or Ctrl+C to keep it open'
    read
"

# Check if session started successfully
sleep 1
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${GREEN}✓ tmux session started successfully!${NC}"
    echo ""
    echo "Commands:"
    echo -e "  ${YELLOW}Attach:${NC}      tmux attach -t $SESSION_NAME"
    echo -e "  ${YELLOW}Detach:${NC}      Ctrl+b, then d"
    echo -e "  ${YELLOW}Kill:${NC}        tmux kill-session -t $SESSION_NAME"
    echo -e "  ${YELLOW}Monitor log:${NC} tail -f $LOG_FILE"
    echo ""
    
    # Optionally attach immediately
    read -p "Attach to session now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        tmux attach -t "$SESSION_NAME"
    fi
else
    echo -e "${RED}✗ Failed to start tmux session${NC}"
    exit 1
fi
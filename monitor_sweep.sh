#!/usr/bin/env bash
#
# Monitor GPU COVID ABM Sweep Progress
# Usage: ./monitor_sweep.sh [log_file]
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Find most recent log file if not specified
if [ -z "$1" ]; then
    LOG_FILE=$(ls -t logs/sweep_*.log 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo "No log files found in logs/"
        echo "Usage: $0 [log_file]"
        exit 1
    fi
else
    LOG_FILE="$1"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file '$LOG_FILE' not found!"
    exit 1
fi

clear
echo -e "${GREEN}=== GPU Sweep Monitor ===${NC}"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to exit"
echo ""

# Function to extract and display progress
show_progress() {
    echo -e "${BLUE}--- Latest Progress ---${NC}"
    
    # Show last few parameter sweeps
    echo "Recent parameter sweeps:"
    grep "🔧 Sweeping:" "$LOG_FILE" | tail -3
    echo ""
    
    # Show last value being tested
    echo "Current value:"
    grep "→ Value:" "$LOG_FILE" | tail -1
    echo ""
    
    # Show latest simulation status
    echo "Latest simulation day:"
    grep "Day.*Inf=.*Imm=.*LC=" "$LOG_FILE" | tail -3
    echo ""
    
    # Check for completion messages
    if grep -q "SWEEP COMPLETE" "$LOG_FILE"; then
        echo -e "${GREEN}✓ SWEEP COMPLETE!${NC}"
        grep "Total time:" "$LOG_FILE" | tail -1
        grep "Avg per sim:" "$LOG_FILE" | tail -1
        return 1
    fi
    
    # Estimate progress
    TOTAL_PARAMS=$(grep -c "🔧 Sweeping:" "$LOG_FILE" || echo "0")
    COMPLETED_RUNS=$(grep -c "✓ Complete:" "$LOG_FILE" || echo "0")
    
    if [ "$TOTAL_PARAMS" -gt 0 ]; then
        echo "Completed simulations: $COMPLETED_RUNS"
    fi
    
    return 0
}

# GPU monitoring function
show_gpu() {
    echo -e "${BLUE}--- GPU Status ---${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
            awk -F', ' '{printf "GPU %s: %s | Temp: %s°C | GPU: %s%% | Mem: %s%% (%s/%s MB)\n", $1, $2, $3, $4, $5, $6, $7}'
    else
        echo "nvidia-smi not available"
    fi
    echo ""
}

# Main monitoring loop
while true; do
    clear
    echo -e "${GREEN}=== GPU Sweep Monitor ===${NC}"
    echo "Log: $LOG_FILE"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Show GPU status
    show_gpu
    
    # Show progress
    if ! show_progress; then
        # Sweep complete, show final stats and exit
        echo ""
        echo -e "${YELLOW}Monitoring ended - sweep complete${NC}"
        break
    fi
    
    echo ""
    echo -e "${YELLOW}Refreshing in 10 seconds... (Ctrl+C to exit)${NC}"
    sleep 10
done

# Offer to show full results
echo ""
read -p "Show tail of log file? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tail -50 "$LOG_FILE"
fi
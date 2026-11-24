#!/bin/bash
# =============================================================================
# Join the running Isaac Sim container from another terminal
# =============================================================================

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/.env"

# Check if the container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Start it first with: ./run.sh"
    exit 1
fi

echo "Joining container '${CONTAINER_NAME}'..."
docker exec -it ${CONTAINER_NAME} bash

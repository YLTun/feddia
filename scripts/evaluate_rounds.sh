#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Batch-evaluate multiple saved checkpoints from one training run.

Two checkpoint layouts are supported:
  --run-dir     federated runs   -> <dir>/global_agent_states/round_XXX.pt
  --client-dir  isolated clients -> <dir>/round_artifacts/round_XXX/agent_state.pt

Usage:
  bash scripts/evaluate_rounds.sh --run-dir <run_dir> [options]
  bash scripts/evaluate_rounds.sh --client-dir <client_dir> [options]

Required (exactly one):
  --run-dir <path>          Training run directory, e.g.
                            logs/reward_weighted/run_20260411_080739_seed_52
  --client-dir <path>       One client directory of an isolated run, e.g.
                            logs/isolated/run_.../clients/client_0
                            The run must have been trained with
                            --no_cleanup_client_agent_states, or no per-round
                            checkpoints will exist.

Options:
  --rounds <csv|all>        Comma-separated rounds to evaluate (default: 5,10,15,20,25,30)
                            Use 'all' to evaluate every saved checkpoint.
  --gpu-index <id>          GPU index to use (optional).
  --eval-episodes <n>       Episodes per mode (default: 1000).
  --num-envs <n>            Parallel envs per eval run (default: 256).
  --eval-modes <csv>        Modes to evaluate
                            (default: none,random,dense,random_dense,gate_palisade_multi).
  --isaaclab-script <path>  Isaac Lab launcher path (default: /opt/IsaacLab/isaaclab.sh).
  --dry-run                 Print commands without executing them.
  -h, --help                Show this help message.

Reports are written under <run_root>/evaluations/<scope>/<label>/. For --client-dir
the scope is forced to 'isolated' and the label to <client>_<round>, so clients and
rounds never overwrite each other.

Examples:
  bash scripts/evaluate_rounds.sh \
      --run-dir logs/fedavg/run_20260423_081705_seed_42 \
      --rounds 5,10,15,20,25,30 \
      --gpu-index 0

  bash scripts/evaluate_rounds.sh \
      --run-dir logs/fedavg/run_20260411_080739_seed_52 \
      --rounds all \
      --gpu-index 0

  # Isolated run: one client at a time
  for c in client_0 client_1 client_2; do
      bash scripts/evaluate_rounds.sh \
          --client-dir logs/isolated/run_20260728_101500_seed_32/clients/"$c" \
          --rounds 20,40,60 \
          --gpu-index 0
  done
EOF
}

RUN_DIR=""
CLIENT_DIR=""
ROUNDS_SPEC="5,10,15,20,25,30"
GPU_INDEX=""
EVAL_EPISODES="1000"
NUM_ENVS="256"
EVAL_MODES="none,random,dense,random_dense,gate_palisade_multi"
ISAACLAB_SCRIPT="/opt/IsaacLab/isaaclab.sh"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --client-dir)
            CLIENT_DIR="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS_SPEC="$2"
            shift 2
            ;;
        --gpu-index)
            GPU_INDEX="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --eval-modes)
            EVAL_MODES="$2"
            shift 2
            ;;
        --isaaclab-script)
            ISAACLAB_SCRIPT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo
            usage
            exit 1
            ;;
    esac
done

if [[ -n "$RUN_DIR" && -n "$CLIENT_DIR" ]]; then
    echo "Error: pass either --run-dir or --client-dir, not both."
    echo
    usage
    exit 1
fi
if [[ -z "$RUN_DIR" && -z "$CLIENT_DIR" ]]; then
    echo "Error: one of --run-dir or --client-dir is required."
    echo
    usage
    exit 1
fi

# Isolated runs have no aggregated global model, so per-round checkpoints live in each
# client's round_artifacts/ instead. Only the discovery differs; everything below is shared.
if [[ -n "$CLIENT_DIR" ]]; then
    CLIENT_DIR="${CLIENT_DIR%/}"
    CLIENT_NAME="$(basename "$CLIENT_DIR")"
    SOURCE_LABEL="$CLIENT_DIR"
    CHECKPOINT_DIR="$CLIENT_DIR/round_artifacts"
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        echo "Error: client checkpoint directory not found: $CHECKPOINT_DIR"
        echo "Isolated runs must be trained with --no_cleanup_client_agent_states,"
        echo "otherwise per-round agent states are deleted during training."
        exit 1
    fi
else
    SOURCE_LABEL="$RUN_DIR"
    CHECKPOINT_DIR="$RUN_DIR/global_agent_states"
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        echo "Error: global checkpoint directory not found: $CHECKPOINT_DIR"
        exit 1
    fi
fi

round_files=()
if [[ "$ROUNDS_SPEC" == "all" ]]; then
    if [[ -n "$CLIENT_DIR" ]]; then
        while IFS= read -r file; do
            round_files+=("$file")
        done < <(find "$CHECKPOINT_DIR" -mindepth 2 -maxdepth 2 -type f -name 'agent_state.pt' | sort)
    else
        while IFS= read -r file; do
            round_files+=("$file")
        done < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'round_*.pt' | sort)
    fi
else
    IFS=',' read -r -a requested_rounds <<< "$ROUNDS_SPEC"
    for round in "${requested_rounds[@]}"; do
        trimmed="$(echo "$round" | xargs)"
        [[ -z "$trimmed" ]] && continue
        formatted_round=$(printf "%03d" "$trimmed")
        if [[ -n "$CLIENT_DIR" ]]; then
            checkpoint="$CHECKPOINT_DIR/round_${formatted_round}/agent_state.pt"
        else
            checkpoint="$CHECKPOINT_DIR/round_${formatted_round}.pt"
        fi
        if [[ -f "$checkpoint" ]]; then
            round_files+=("$checkpoint")
        else
            echo "[WARN] Missing checkpoint, skipping: $checkpoint"
        fi
    done
fi

if [[ ${#round_files[@]} -eq 0 ]]; then
    echo "No checkpoints found to evaluate."
    exit 1
fi

echo "Evaluating ${#round_files[@]} checkpoint(s) from: $SOURCE_LABEL"
echo "Rounds: $ROUNDS_SPEC"
echo "Eval modes: $EVAL_MODES"
echo "Episodes per mode: $EVAL_EPISODES"
echo "Num envs: $NUM_ENVS"
echo

for checkpoint in "${round_files[@]}"; do
    if [[ -n "$CLIENT_DIR" ]]; then
        # Every client checkpoint is named agent_state.pt, so the round comes from its
        # parent directory (round_XXX) and the label must also carry the client id --
        # evaluate_sequential.py would otherwise derive the label "agent_state" for all
        # of them and write every report into the same directory.
        round_label="$(basename "$(dirname "$checkpoint")")"
        echo "===== Evaluating $CLIENT_NAME / $round_label ====="
    else
        round_label="$(basename "$checkpoint" .pt)"
        echo "===== Evaluating $round_label ====="
    fi

    cmd=(
        "$ISAACLAB_SCRIPT"
        -p scripts/evaluate_sequential.py
        --model_path "$checkpoint"
        --model_format pt
        --eval_modes "$EVAL_MODES"
        --num_envs "$NUM_ENVS"
        --eval_episodes "$EVAL_EPISODES"
        --headless
    )

    if [[ -n "$CLIENT_DIR" ]]; then
        cmd+=(--eval_scope isolated --eval_label "${CLIENT_NAME}_${round_label}")
    fi

    if [[ -n "$GPU_INDEX" ]]; then
        cmd+=(--gpu-index "$GPU_INDEX")
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'DRY RUN: '
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi

    echo

done

echo "Batch evaluation finished."

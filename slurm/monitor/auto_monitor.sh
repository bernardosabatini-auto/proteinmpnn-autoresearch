#!/bin/bash
# =============================================================================
# Autonomous ProteinMPNN production-run monitor.
#
# Driven by user crontab (every 20 min). For each registered experiment in
# monitor/jobs/<EXP_NAME>.sbatch.sh it:
#   - tracks the current SLURM JobID in monitor/state/<EXP_NAME>.state
#   - follows the train_production.sh self-resubmit chain via --job-name
#   - classifies terminal outcomes (OK, OOM, NCCL timeout, CUDA OOM, other)
#   - auto-resubmits recoverable failures with the right fix flag
#   - emails bernardo_sabatini@hms.harvard.edu on state transitions and
#     progress heartbeats (every ~6h or 20 additional epochs).
#
# Side effects are capped: MAX_RETRIES=3 resubmits per experiment, beyond
# that it only emails for human intervention.
# =============================================================================
set -u
umask 0027

# Code (jobs/) tracks with the script in the repo; runtime state + log
# live on netscratch so repo stays clean and state survives repo updates.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="$SCRIPT_DIR/jobs"
RUNTIME_DIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/monitor
STATE_DIR=$RUNTIME_DIR/state
OUT_DIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs
LOG_DIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs
MLOG=$RUNTIME_DIR/auto_monitor.log
EMAIL_TO=bernardo_sabatini@hms.harvard.edu

MAX_RETRIES=3
HEARTBEAT_SEC=$((6 * 3600))
HEARTBEAT_EPOCH_DELTA=20

mkdir -p "$STATE_DIR"
touch "$MLOG"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$MLOG"; }

email() {
  local subj="$1"
  /usr/bin/mail -s "AUTO: $subj" "$EMAIL_TO"
}

# sacct state for a JobID. Returns e.g. COMPLETED, FAILED, OUT_OF_MEMORY,
# TIMEOUT, CANCELLED, RUNNING, PENDING. Empty on lookup failure.
sacct_state() {
  local jid="$1"
  /usr/bin/sacct -j "$jid" -X -n -P -o State 2>/dev/null | head -1 | awk '{print $1}'
}

# Highest epoch and its valid_acc from the experiment log.
latest_epoch_vacc() {
  local f="$1"
  [ -f "$f" ] || { echo "0 -"; return; }
  awk -F'[,:]' '/^epoch/ {
    for(i=1;i<=NF;i++) {
      if ($i ~ /epoch/)     e=$(i+1)+0
      if ($i ~ /valid_acc/) v=$(i+1)
    }
  } END { if (e=="") print "0 -"; else printf "%d %s\n", e, v }' "$f" | tail -1
}

# Classify a FAILED job: read its slurm .err and return one of:
#   oom, cuda_oom, nccl_timeout, python_exc, unknown
classify_failure() {
  local jid="$1"
  local errf="$LOG_DIR/prod_${jid}.err"
  [ -f "$errf" ] || { echo "unknown"; return; }
  if grep -q "oom_kill event\|Out Of Memory" "$errf"; then
    # system-RAM OOM from cgroup
    echo "oom"; return
  fi
  if grep -q "CUDA out of memory\|torch.cuda.OutOfMemoryError" "$errf"; then
    echo "cuda_oom"; return
  fi
  if grep -q "Watchdog caught collective operation timeout\|NCCL.*timeout" "$errf"; then
    echo "nccl_timeout"; return
  fi
  if grep -qE "Traceback|Error|Exception" "$errf"; then
    echo "python_exc"; return
  fi
  echo "unknown"
}

# State-file I/O. Format is one KEY=VALUE per line; missing keys default.
load_state() {
  local f="$1"
  CURRENT_JOBID=0; LAST_STATE=""; LAST_EPOCH_EMAILED=0
  LAST_EMAIL_UNIX=0; RETRY_COUNT=0
  [ -f "$f" ] && source "$f" || true
}
save_state() {
  local f="$1"
  cat > "$f" <<EOF
CURRENT_JOBID=$CURRENT_JOBID
LAST_STATE=$LAST_STATE
LAST_EPOCH_EMAILED=$LAST_EPOCH_EMAILED
LAST_EMAIL_UNIX=$LAST_EMAIL_UNIX
RETRY_COUNT=$RETRY_COUNT
EOF
}

# Find an active (RUNNING or PENDING) job with the given --job-name.
# Covers train_production.sh chain-resubmits: the resubmitted job inherits
# the same SBATCH --job-name, so this returns the current chain step.
#
# Derives the job-name by reading the sbatch script; expects a line
#   --job-name=<NAME>
derive_jobname() {
  awk -F'=' '/--job-name=/ { gsub(/[ \\]/,"",$2); print $2; exit }' "$1"
}

find_active_by_name() {
  local name="$1"
  /usr/bin/squeue -u "$USER" -h -o "%i %j %t" 2>/dev/null \
    | awk -v n="$name" '$2==n && ($3=="R"||$3=="PD") { print $1; exit }'
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
log "=== tick ==="

shopt -s nullglob
for relaunch in "$JOBS_DIR"/*.sbatch.sh; do
  EXP_NAME=$(basename "$relaunch" .sbatch.sh)
  STATE_FILE=$STATE_DIR/${EXP_NAME}.state
  JOB_NAME=$(derive_jobname "$relaunch")
  LOG_FILE=$OUT_DIR/$EXP_NAME/log.txt

  load_state "$STATE_FILE"

  # If we don't yet have a JobID for this exp, try to discover one by name.
  if [ "$CURRENT_JOBID" = "0" ] || [ -z "$CURRENT_JOBID" ]; then
    CURRENT_JOBID=$(find_active_by_name "$JOB_NAME")
    [ -z "$CURRENT_JOBID" ] && CURRENT_JOBID=0
  fi

  # Follow the chain: if a newer job with the same name is active, adopt it.
  ACTIVE_JID=$(find_active_by_name "$JOB_NAME")
  if [ -n "$ACTIVE_JID" ] && [ "$ACTIVE_JID" != "$CURRENT_JOBID" ]; then
    log "$EXP_NAME: chain advanced $CURRENT_JOBID -> $ACTIVE_JID"
    CURRENT_JOBID=$ACTIVE_JID
    # Reset retry counter when chain legitimately advances to a new step.
    RETRY_COUNT=0
  fi

  if [ "$CURRENT_JOBID" = "0" ] || [ -z "$CURRENT_JOBID" ]; then
    log "$EXP_NAME: no active or tracked job, skipping"
    save_state "$STATE_FILE"
    continue
  fi

  STATE=$(sacct_state "$CURRENT_JOBID")
  [ -z "$STATE" ] && STATE="UNKNOWN"
  log "$EXP_NAME: jobid=$CURRENT_JOBID state=$STATE"

  case "$STATE" in
    RUNNING|PENDING|REQUEUED|RESIZING|SUSPENDED|CONFIGURING)
      # Live job. Maybe send heartbeat.
      NOW=$(date +%s)
      read EP VACC <<< "$(latest_epoch_vacc "$LOG_FILE")"
      DELTA_T=$((NOW - LAST_EMAIL_UNIX))
      DELTA_E=$((EP - LAST_EPOCH_EMAILED))
      SHOULD_EMAIL=0
      # Always send the first heartbeat once training actually starts.
      if [ "$LAST_STATE" != "RUNNING" ] && [ "$STATE" = "RUNNING" ]; then
        SHOULD_EMAIL=1
      fi
      if [ "$DELTA_T" -ge "$HEARTBEAT_SEC" ] && [ "$EP" -gt 0 ]; then
        SHOULD_EMAIL=1
      fi
      if [ "$DELTA_E" -ge "$HEARTBEAT_EPOCH_DELTA" ] && [ "$EP" -gt 0 ]; then
        SHOULD_EMAIL=1
      fi
      if [ "$SHOULD_EMAIL" = "1" ]; then
        {
          echo "Experiment: $EXP_NAME"
          echo "JobID:      $CURRENT_JOBID"
          echo "SLURM:      $STATE"
          echo "Epoch:      $EP"
          echo "valid_acc:  $VACC"
          echo "Retries:    $RETRY_COUNT / $MAX_RETRIES"
          echo
          echo "Last 10 log lines:"
          tail -10 "$LOG_FILE" 2>/dev/null || echo "(no log yet)"
        } | email "$EXP_NAME $STATE ep=$EP vacc=$VACC"
        LAST_EPOCH_EMAILED=$EP
        LAST_EMAIL_UNIX=$NOW
      fi
      LAST_STATE=$STATE
      ;;

    COMPLETED)
      if [ "$LAST_STATE" != "COMPLETED" ]; then
        read EP VACC <<< "$(latest_epoch_vacc "$LOG_FILE")"
        {
          echo "Experiment: $EXP_NAME"
          echo "JobID:      $CURRENT_JOBID"
          echo "Final epoch: $EP"
          echo "Final valid_acc: $VACC"
          echo
          echo "Last 20 log lines:"
          tail -20 "$LOG_FILE" 2>/dev/null
        } | email "$EXP_NAME COMPLETED vacc=$VACC"
        LAST_STATE=COMPLETED
        LAST_EMAIL_UNIX=$(date +%s)
      fi
      ;;

    TIMEOUT)
      # train_production.sh is supposed to self-resubmit on timeout. Give
      # it one monitor tick to show up under the same job-name; if nothing
      # appears we resubmit manually.
      NEW_JID=$(find_active_by_name "$JOB_NAME")
      if [ -n "$NEW_JID" ] && [ "$NEW_JID" != "$CURRENT_JOBID" ]; then
        log "$EXP_NAME: TIMEOUT handled by built-in chain -> $NEW_JID"
        CURRENT_JOBID=$NEW_JID
        LAST_STATE=PENDING
      elif [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; then
        log "$EXP_NAME: TIMEOUT without chain successor; manual resubmit"
        NEW_JID=$(bash "$relaunch" 2>>"$MLOG") || NEW_JID=""
        if [ -n "$NEW_JID" ]; then
          CURRENT_JOBID=$NEW_JID
          RETRY_COUNT=$((RETRY_COUNT+1))
          LAST_STATE=PENDING
          echo "Resubmitted $EXP_NAME after TIMEOUT: new JobID $NEW_JID (retry $RETRY_COUNT/$MAX_RETRIES)" \
            | email "$EXP_NAME TIMEOUT resubmitted as $NEW_JID"
        fi
      fi
      ;;

    FAILED|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|PREEMPTED|CANCELLED*)
      if [ "$LAST_STATE" = "$STATE" ]; then
        # Already acted on this failure last tick; wait.
        :
      else
        FAIL_CLASS=$(classify_failure "$CURRENT_JOBID")
        read EP VACC <<< "$(latest_epoch_vacc "$LOG_FILE")"
        log "$EXP_NAME: $STATE class=$FAIL_CLASS ep=$EP vacc=$VACC retries=$RETRY_COUNT"

        MEM_OVERRIDE=""; EXTRA=""; RESUBMIT=0
        case "$FAIL_CLASS" in
          oom)
            # Bump memory further. 1200G still fits in 1.5TB node.
            MEM_OVERRIDE="--mem=1200G"
            RESUBMIT=1
            ;;
          nccl_timeout)
            # Often transient (noisy fabric); retry same config.
            RESUBMIT=1
            ;;
          cuda_oom)
            # Training script isn't configured to take a smaller batch via
            # monitor easily; flag for human review.
            RESUBMIT=0
            ;;
          *)
            RESUBMIT=0
            ;;
        esac

        if [ "$RESUBMIT" = "1" ] && [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; then
          NEW_JID=$(MEM_OVERRIDE="$MEM_OVERRIDE" EXTRA="$EXTRA" bash "$relaunch" 2>>"$MLOG") || NEW_JID=""
          if [ -n "$NEW_JID" ]; then
            RETRY_COUNT=$((RETRY_COUNT+1))
            {
              echo "Experiment: $EXP_NAME"
              echo "Failed JobID:    $CURRENT_JOBID ($STATE, class=$FAIL_CLASS)"
              echo "Resubmitted as:  $NEW_JID   (MEM_OVERRIDE='$MEM_OVERRIDE')"
              echo "Retry:           $RETRY_COUNT / $MAX_RETRIES"
              echo "Progress at failure: epoch=$EP valid_acc=$VACC"
              echo
              echo "Last 30 lines of $LOG_DIR/prod_${CURRENT_JOBID}.err:"
              tail -30 "$LOG_DIR/prod_${CURRENT_JOBID}.err" 2>/dev/null
            } | email "$EXP_NAME $STATE/$FAIL_CLASS resubmitted as $NEW_JID"
            CURRENT_JOBID=$NEW_JID
            LAST_STATE=PENDING
          else
            echo "Resubmit attempt FAILED (sbatch returned nothing)." \
              | email "$EXP_NAME $STATE resubmit FAILED - human needed"
            LAST_STATE=$STATE
          fi
        else
          {
            echo "Experiment: $EXP_NAME"
            echo "JobID:      $CURRENT_JOBID"
            echo "Outcome:    $STATE (class=$FAIL_CLASS)"
            echo "Retries used: $RETRY_COUNT / $MAX_RETRIES"
            echo "Progress at failure: epoch=$EP valid_acc=$VACC"
            echo
            echo "Not auto-resubmitting (class unhandled or retries exhausted)."
            echo
            echo "Last 30 .err lines:"
            tail -30 "$LOG_DIR/prod_${CURRENT_JOBID}.err" 2>/dev/null
          } | email "$EXP_NAME $STATE/$FAIL_CLASS - HUMAN NEEDED"
          LAST_STATE=$STATE
        fi
      fi
      ;;

    *)
      log "$EXP_NAME: state=$STATE (no action)"
      LAST_STATE=$STATE
      ;;
  esac

  save_state "$STATE_FILE"
done

log "=== tick done ==="

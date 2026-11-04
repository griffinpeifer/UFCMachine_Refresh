#!/bin/bash
# Monthly UFC model refresh — runs on the 1st of each month via launchd.
# Full pipeline: fighter scrape → match scrape → clean → retrain.
# Backs up prior artifacts to backups/YYYY-MM-DD/ before overwriting.

set -uo pipefail  # NOTE: no -e so a late step failure still leaves earlier artifacts intact

PROJECT_DIR="$HOME/Documents/Personal_Projects/UFCMachine_Refresh"
LOG_DIR="$PROJECT_DIR/logs"
BACKUP_DIR="$PROJECT_DIR/backups/$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/refresh-$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR" "$BACKUP_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== UFC Model Refresh started ==="

cd "$PROJECT_DIR" || { log "ERROR: project dir not found"; exit 1; }

# ---- Back up current artifacts ----
log "Backing up current model/data files to $BACKUP_DIR"
for f in \
    ufc_prediction_model.h5 \
    ufc_model_best.h5 \
    ufc_model_logistic.h5 \
    calibration.json \
    model_metrics.txt \
    training_pit.csv \
    ufc_model_exp1_simple.h5 \
    ufc_model_exp2_dropout.h5 \
    ufc_model_exp3_wide.h5 \
    scaler.pkl \
    fighters_clean.csv \
    matches_clean.csv \
    ufc_fighters_raw.csv \
    training_model.csv
do
    [ -f "$f" ] && cp "$f" "$BACKUP_DIR/"
done

# ---- Activate venv ----
# shellcheck source=/dev/null
source .venv/bin/activate
log "Python: $(python --version 2>&1), venv: $VIRTUAL_ENV"

# ufcstats.com now blocks scripts with a browser check, so data comes from the
# public Greco1899/scrape_ufc_stats dataset (updated near-daily) instead of
# scraping. See scripts/build_from_greco.py.
DATA_REPO="$PROJECT_DIR/data/scrape_ufc_stats"
STEP1_RC=0; STEP2_RC=0; STEP3_RC=0

# ---- Step 1: download latest dataset ----
log "Step 1/3: updating Greco1899/scrape_ufc_stats dataset"
if [ -d "$DATA_REPO/.git" ]; then
    git -C "$DATA_REPO" pull --ff-only --depth 1 >> "$LOG_FILE" 2>&1 \
      || { rm -rf "$DATA_REPO" && git clone --depth 1 https://github.com/Greco1899/scrape_ufc_stats.git "$DATA_REPO" >> "$LOG_FILE" 2>&1; }
else
    mkdir -p "$PROJECT_DIR/data"
    git clone --depth 1 https://github.com/Greco1899/scrape_ufc_stats.git "$DATA_REPO" >> "$LOG_FILE" 2>&1
fi
STEP1_RC=$?
log "Step 1 exit code: $STEP1_RC"
if [ $STEP1_RC -ne 0 ]; then
    log "ABORT: could not download dataset. Model left untouched."
    osascript -e 'display notification "Could not download UFC dataset — refresh aborted, model untouched." with title "UFC Machine Refresh"' 2>/dev/null || true
    exit 1
fi

# ---- Step 2: build fighters_clean.csv + matches_clean.csv ----
log "Step 2/3: build_from_greco.py"
python scripts/build_from_greco.py --data-dir "$DATA_REPO" >> "$LOG_FILE" 2>&1
STEP2_RC=$?
log "Step 2 exit code: $STEP2_RC"
if [ $STEP2_RC -eq 2 ]; then
    log "ABORT: dataset looks stale (no new events in 45+ days). Model left untouched."
    osascript -e 'display notification "UFC dataset has not updated in 45+ days — refresh skipped. Check the source repo." with title "UFC Machine Refresh"' 2>/dev/null || true
    exit 1
elif [ $STEP2_RC -ne 0 ]; then
    log "ABORT: build failed. Restoring backed-up CSVs; model left untouched."
    cp "$BACKUP_DIR/fighters_clean.csv" "$BACKUP_DIR/matches_clean.csv" . 2>/dev/null
    osascript -e 'display notification "Building UFC data failed — refresh aborted, model untouched." with title "UFC Machine Refresh"' 2>/dev/null || true
    exit 1
fi

# ---- Step 3: train ----
# Point-in-time training: each fight uses only stats from before it.
# (The notebook is kept for experimenting but its features include the fight
# being predicted, so it's no longer used here.)
log "Step 3/3: train_model.py (point-in-time training, a few minutes)"
python scripts/train_model.py >> "$LOG_FILE" 2>&1
STEP4_RC=$?
[ -f model_metrics.txt ] && sed 's/^/    /' model_metrics.txt | tee -a "$LOG_FILE"
log "Step 3 exit code: $STEP4_RC"
if [ $STEP4_RC -ne 0 ]; then
    log "Training failed — restoring previous model, scaler and CSVs from $BACKUP_DIR"
    cp "$BACKUP_DIR"/* . 2>/dev/null
fi

# ---- Report ----
if [ $STEP1_RC -eq 0 ] && [ $STEP2_RC -eq 0 ] && [ $STEP3_RC -eq 0 ] && [ $STEP4_RC -eq 0 ]; then
    log "=== UFC Model Refresh COMPLETED successfully ==="
    MSG="UFC model refresh completed successfully."
else
    log "=== UFC Model Refresh FINISHED with errors (exit codes: $STEP1_RC/$STEP2_RC/$STEP3_RC/$STEP4_RC) ==="
    MSG="UFC model refresh had errors — check $LOG_FILE"
fi

# macOS notification
osascript -e "display notification \"$MSG\" with title \"UFC Machine Refresh\"" 2>/dev/null || true

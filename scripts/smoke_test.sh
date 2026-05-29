#!/usr/bin/env bash
# Smoke test: run the end-to-end pipeline on a tiny synthetic dataset under
# the mock backend. Should finish in well under a minute and leaves artefacts
# under runs/<task>/.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== running math smoke test (mock backend) ==="
python scripts/06_run_all.py --task math --n 60 --no-probe --seed 0

echo "=== running qa smoke test (mock backend) ==="
python scripts/06_run_all.py --task qa --n 60 --no-probe --seed 0

echo "=== running code smoke test (mock backend) ==="
python scripts/06_run_all.py --task code --n 60 --no-probe --seed 0

echo "=== smoke test OK ==="
echo "see runs/<task>/tables/main.csv for the tables"

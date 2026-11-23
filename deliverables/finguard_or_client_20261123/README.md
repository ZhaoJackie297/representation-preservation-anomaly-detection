# FinGuard-OR Client Package (2026-11-23)

This directory contains the packaged `智御内控（FinGuard-OR）` client shell.

## Archive

- `finguard_or_client_20261123.zip`
- `SHA256SUMS.txt`

The archive contains:

- FastAPI backend, SQLAlchemy persistence, model runner, and synthetic-data generator.
- React/TypeScript/Vite source and the current production `frontend/dist` build.
- Backend tests, pytest configuration, and contract tests.
- The two bank-client UI reference files.
- Synthetic bank drill data and the current smoke SQLite database.
- The frozen SoftInst-Rand checkpoint under `artifacts/frozen_model.pth`.

The archive excludes `node_modules`, `__pycache__`, `.pytest_cache`, user uploads,
and runtime smoke artifacts.

## Run

After extracting the archive:

```powershell
Set-Location .\bank_ops_risk
pip install -r requirements.txt
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

The bundled `launch_finguard_or.bat` contains machine-specific absolute paths
from the development host. Edit those paths before using the launcher on a
different machine.

## Data and model boundary

- All bundled business data is synthetic drill data.
- `unlabeled_live` inference must not read labels or incident truth columns.
- Watchlist state is display-only and must not enter the encoder, memory bank, or scoring path.
- `-1` timing sentinels map to `NaN`; midnight `0` remains a valid time.
- The checkpoint is included for delivery convenience; deployments should pin and verify it explicitly.

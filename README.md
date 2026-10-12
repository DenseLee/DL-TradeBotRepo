# DL Trade Bot Archive

This repository is a source archive for two experimental MT5 trading runners.
It is provided to inspect their behavior on a **demo account only**.

## Safety disclaimer

**Do not use this repository with real money.** These are research experiments,
not validated trading systems or financial advice. The archived MT5 adapter
rejects every non-demo account before any quote or order operation. Use a
separate demo account, small test conditions, and your own independent review.

No credentials, account identifiers, terminal paths, logs, market data, or
broker configuration are included in this repository.

## Layout

- `terminal-1/`: the archived C1 sequence diagnostic runner.
- `terminal-2/`: the archived T9/C1 one-position portfolio experiment.

Each folder intentionally contains its own imported support modules. This keeps
the two archived runners independently readable and prevents a later bot from
silently changing their behavior through shared source code.

## What you need

1. Windows with Python 3.10 or newer.
2. MetaTrader 5 installed and logged into a broker **demo** account.
3. The `MetaTrader5`, `numpy`, and `pandas` Python packages.
4. BTCUSD available and visible in the demo terminal.
5. A local synchronized input-feed directory for the runner you want to inspect.
   Raw market data is deliberately excluded from this archive.

Install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Confirm the archived source parses before connecting it to a terminal:

```powershell
python -m compileall terminal-1 terminal-2
python .\terminal-1\models\trend\c1_sequence_demo_runner.py --help
python .\terminal-2\models\trend\t9_c1_portfolio_live_runner.py --help
```

## Demo-only usage

Supply your own demo login and local terminal path at runtime. Do not save them
in the repository, command history, screenshots, or logs that you later commit.
Both runners require an explicit demo-order switch and reject real-money modes.

Terminal 1 consumes its synchronized probe feed through `--sync-dir`:

```powershell
python .\terminal-1\models\trend\c1_sequence_demo_runner.py run `
  --enable-demo-orders `
  --account <your-demo-login> `
  --terminal <path-to-your-demo-terminal> `
  --sync-dir <path-to-your-synchronized-demo-feed>
```

Terminal 2 needs the same synchronized feed plus its T9 reference inputs:

```powershell
python .\terminal-2\models\trend\t9_c1_portfolio_live_runner.py `
  --enable-demo-orders `
  --account <your-demo-login> `
  --terminal <path-to-your-demo-terminal> `
  --sync-dir <path-to-your-synchronized-demo-feed>
```

The commands above are setup examples, not a claim of profitability or a
recommendation to trade. Inspect the source, verify the data contract, and run
only against a disposable demo environment.

## Archive scope

This is a static archive, not a supported product. It does not include a
watchdog, process manager, historical datasets, performance claims, or a live
deployment path. Future bot archives should be added as new top-level folders.

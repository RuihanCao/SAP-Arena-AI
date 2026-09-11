# SAP-Arena-AI

A Super Auto Pets agent that combines a behavior-cloned policy, search, and a
learned value model. It plays the Turtle pack in Arena mode using a simulator
based on version 45 of the game.

You can evaluate the agent, view its recorded games, and play against it
locally.

The Python import package is `sap_ppo`.

## Installation

Requires Python 3.11 or newer, Node.js, and Git.

The agent runs on CPU; no GPU is required. Search speed depends on your CPU
and the selected search settings.

From the repository root, create a virtual environment and install the Python
dependencies:

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[agents]"
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[agents]"
```

Run all commands below from the repository root. The Linux / macOS commands
assume the virtual environment is activated; the Windows commands use its
Python executable directly, so activation is not needed.

Battles are simulated by [SAP-Calculator](https://github.com/robertley/SAP-Calculator),
which runs on Node.js. Clone the following version into `third_party/`:

Linux / macOS:

```bash
git clone https://github.com/robertley/SAP-Calculator.git third_party/SAP-Calculator
git -C third_party/SAP-Calculator checkout 4d03d89b0c8df346bed9e9f504aefc0494f81155
```

Windows PowerShell:

```powershell
git clone https://github.com/robertley/SAP-Calculator.git third_party\SAP-Calculator
git -C third_party\SAP-Calculator checkout 4d03d89b0c8df346bed9e9f504aefc0494f81155
```

Replay images use the original [sap-replay-bot renderer](https://github.com/RuihanCao/sap-replay-bot/tree/ca06ba34bf647b7876d15922c9b24ab0d1dd16a3),
pinned to `ca06ba34bf647b7876d15922c9b24ab0d1dd16a3`.
The setup script downloads it into `third_party/sap-replay-bot/` and installs
Canvas 3.1.2 locally. Run this before evaluation or the local demo.

Linux / macOS:

```bash
python tools/setup_renderer.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tools\setup_renderer.py
```

This installs the renderer only; it does not start a Discord bot.

## Model weights

Download the pretrained BC policy and value model using the download script.
The files are saved to `models/`.

Linux / macOS:

```bash
python tools/fetch_models.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tools\fetch_models.py
```

- `bc_attn_v4.zip`: the behavior-cloned policy, trained on human gameplay and
  used to propose candidate action sequences.
- `vgame_heads_w3b.pt`: the learned value model, used to score end-of-turn
  states during search.

The pretrained weights are provided under the [MIT License](LICENSE).

## Evaluation

Run the agent against the included Arena opponent pool:

Linux / macOS:

```bash
python tools/evaluate.py --policy search --games 1000 --out evaluation.json
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tools\evaluate.py --policy search --games 1000 --out evaluation.json
```

The default search configuration uses a root proposal budget of 72, completion
width `w=4`, and `k=12` chance samples.

The command saves:

- `evaluation.json`: summary statistics and configuration.
- `evaluation.games.jsonl`: individual game records.
- `evaluation.gallery/index.html`: a gallery of the first five games.

Use `--gallery-games N` to change the number of games shown, or
`--gallery-games 0` to disable gallery generation.

Open the generated gallery:

Linux desktop:

```bash
xdg-open evaluation.gallery/index.html
```

macOS:

```bash
open evaluation.gallery/index.html
```

Windows PowerShell:

```powershell
Invoke-Item .\evaluation.gallery\index.html
```

Reference results on this opponent pool:

| Agent / human reference | Games | Mean trophies |
|---|---:|---:|
| Agent | 1,000 | 4.7150 |
| Human reference, Elo ≥ 1700 | 182 | 4.9176 |
| Human reference, Elo 1501–1699 | 303 | 4.7030 |
| Human reference, Elo ≤ 1500 | 60 | 3.6500 |

The agent score comes from complete games, with a 95% confidence interval of
4.582–4.853 trophies.

Human references replay recorded teams against the same opponent pool. Their
scores may be underestimated because a recording can run out before the
simulated game ends.

## Local demo

Prepare the opponent pool for the demo:

Linux / macOS:

```bash
python tools/build_demo_snapshot.py \
    data/opponents/arena_val_pool_deidentified.json.gz \
    data/opponents/demo_snapshot.json.gz
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tools\build_demo_snapshot.py `
    data\opponents\arena_val_pool_deidentified.json.gz `
    data\opponents\demo_snapshot.json.gz
```

Start the local server:

Linux / macOS:

```bash
python -m sap_ppo.tools.play_web \
    --host 127.0.0.1 --port 8765 \
    --snapshot-path data/opponents/demo_snapshot.json.gz \
    --game-mode arena
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m sap_ppo.tools.play_web `
    --host 127.0.0.1 --port 8765 `
    --snapshot-path data\opponents\demo_snapshot.json.gz `
    --game-mode arena
```

Open `http://127.0.0.1:8765/play` to play against the agent. You can also visit
`http://127.0.0.1:8765/sandbox` to play Arena games against the included opponent
pool.

The home page at `http://127.0.0.1:8765/` has Play, Sandbox, and Replays buttons.
Play is a direct duel against the agent using versus rules by default;
evaluation and the Arena sandbox use the opponent pool instead.

### Search settings

Choose a search mode in the game setup menu:

- **Fixed-all:** Uses the same search configuration as evaluation.
- **Grow k (default):** Keeps the same root and completion widths, starts at
  `k=12`, and uses additional thinking time to sample more chance outcomes.

The default turn budget for Grow k is 105 seconds and can be adjusted in the
setup menu.

## Recorded games

Generate a gallery from the included game records:

Linux / macOS:

```bash
python tools/gallery.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tools\gallery.py
```

Open `gallery/index.html` in your browser to view each game turn by turn,
including the agent’s team, the opponent’s team, and the battle result.
Each game also has a PNG replay image in the same folder.

Linux desktop:

```bash
xdg-open gallery/index.html
```

macOS:

```bash
open gallery/index.html
```

Windows PowerShell:

```powershell
Invoke-Item .\gallery\index.html
```

## Data

The included Arena opponent pool contains 3,096 recorded turns from 268 human
games. It is the validation split used for the evaluation results above, and
also supplies opponents for the local Arena sandbox.

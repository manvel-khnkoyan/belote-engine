## Environment setup

### Virtualenv

Create a virtual environment to isolate dependencies.   
```bash
python3 -m venv bin
``` 

Activate the environment and set up the path.

```bash
source bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Test

Run all tests

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/src/models && ./bin/python -m pytest src/models/
``` 

Test individual file

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/src/models && ./bin/python -m pytest src/models/card_test.py
```

## Universal Models
The system uses a set of constant, universal models (**Card**, **Deck**, **Trump**, **Set**, **Probability**) to represent the fundamental entities of the game. These models encapsulate the static rules and properties of Belote.

## Phase 1: Play
The architecture separates the **Game Logic (Core)** from the **Decision Making (Agents)**.

### Core Logic
The simulation follows a strict cycle:

*   **Rules**: Determines valid moves from the current state (`Rules(State) -> [Action]`).
*   **Agent**: Selects an action from the valid options (`Agent.choose_action([Action]) -> Action`).
*   **State**: Updates itself based on actions (`State.observe(player, Action)`).
*   **Simulator**: Orchestrates the game loop using Rules and Agents to produce a final outcome (`Simulator(Rules, Agents)->simulate(States) -> Result`).

The **Result** serves as a complete record for **replication** or **training**.

### Implementation Strategy
*   **Core**: Contains the immutable logic of the game (Rules, State definitions, Simulator).
*   **Specific Implementations (e.g., PPO)**: Build upon the Core. A PPO Agent, for example, wraps the Core State with training-specific logic (rewards, policy networks) to learn optimal strategies without modifying the underlying game rules.

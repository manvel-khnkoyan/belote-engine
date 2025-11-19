

## Environment setup

## virtualenv

Create a virtual environment to isolate dependencies.   
```bash
python3 -m venv bin
``` 

Set the PYTHONPATH environment variable to the root of the project in your ".env" file.

```bash
source bin/activate
source .env
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Test

Run all tests

```bash
pytest -v ./src/
``` 

Test individual file

```bash
pytest src/states/probability_test.py -v
```

### State is personal

{
    "probs": its a probability matrix player/rank/suit -> 0-1 float
    "trump": {mode(0-3), suit (0-3/None)} - current trump
    "hands": List[{i,j}] - list of hands for current player
    "table": List[{i,j}] - list of cards on the table
    "score": [team0_score, team1_score] - current score
    "trick": {
        "winner": 0-3/None - current trick winner player index
        "points": 0-... - current trick points
    }
}

### Actions

Actions are depending on the game type. Each action has a type and optional suit.

{
    "type": ActionType,
    "logs": Any - Agent specific logs, thats going to be saved and passed back to the agent for learning purposes
    ...
    params based on action type {suit, suits}
}


### Env

env.choose_action(i, state, []actions) -> Action


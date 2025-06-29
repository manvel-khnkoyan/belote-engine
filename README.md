

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

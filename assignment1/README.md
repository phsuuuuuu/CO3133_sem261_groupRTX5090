##just  for store command to run smth  
##terminal  = powershell

1/prepare environment
    make virtual env:
    python  -m venv .venv

    activate virtual env:
    ./.venv/Scripts/activate.ps1

    upgrade pip: to get smth
    python -m pip  install --upgrade  pip

    using pip install smth:
    pip  install -r requirements.txt

    deactivate
2/ run the  code:
python src/data.py

python src/eda.py

python -m src.train --model linear --epochs 1 --max-batches 1

python -m src.train --model mlp --epochs 1 --max-batches 1


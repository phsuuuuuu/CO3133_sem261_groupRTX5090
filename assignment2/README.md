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

2/ prepare data
    Download dataset from [text](https://www.cityscapes-dataset.com/downloads/) (just download 2 files: gtFine_trainvaltest.zip (241MB) and leftImg8bit_trainvaltest.zip (11GB))
    After that extract all the files into assignment2/dataset/

    Then run the prepare_data.py by:
    python assignment2/scripts/prepare_data.py
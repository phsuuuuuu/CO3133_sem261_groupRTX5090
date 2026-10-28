# Assignment 1 — Linear, MLP, CNN, LSTM, Transformer on Fashion-MNIST

**Group RTX5090** · CO3133 Semester-261 · Instructor: Dr. Le Thanh Sach

- Web page: https://phsuuuuuu.github.io/CO3133_sem261_groupRTX5090/assignment1/assignment1.html
- Report: [Assignment1_M2_RTX5090.pdf](../docs/reports/Assignment1_M2_RTX5090.pdf) · Slides: [Assignment1_Slides_RTX5090.pdf](../docs/reports/Assignment1_Slides_RTX5090.pdf)
- Video: https://youtu.be/6xHOHH6awPw · AI usage: [AI_USAGE.md](../AI_USAGE.md)

## Environment
Windows 11, Python 3.12, PyTorch 2.13 (CUDA), NVIDIA GeForce RTX 5070 Ti. Dependency versions are in `requirements.txt`.

```powershell
cd assignment1
python -m venv .venv; ./.venv/Scripts/activate.ps1
pip install -r requirements.txt
```

## Data
Fashion-MNIST is downloaded automatically by torchvision into `data/`.
The split is 54,000 train / 6,000 val (from the official train set) / 10,000 test, with **seed 666**. The indices are saved in `results/split.npz` and reused by all models.

```powershell
python src/data.py    # download + create split
python src/eda.py     # EDA figures -> results/figures/
```

## Train, evaluate and plot
Defaults: 10 epochs, Adam lr 1e-3, batch size 200, seed 666 (see `python -m src.train -h`).

```powershell
python -m src.train --model linear       # linear | mlp | cnn | lstm | transformer
python -m src.train --model transformer --epochs 30   # extra experiment
python -m src.evaluate                    # confusion matrices + prediction examples
python -m src.plot_curves                 # training curves
```

Per-epoch logs go to `results/<model>/<model>_ep<E>_history.csv`, and the final metrics are appended to `results/summary.csv`.

## Checkpoints
The `.pt` files are not committed (they are too large). To recreate them, run the training commands above. Each run saves:

- `checkpoints/<model>_ep<E>/best.pt`: the epoch with the best **validation macro-F1** (this is the checkpoint used for the test results)
- `checkpoints/<model>_ep<E>/epoch_XXX.pt`: one file per epoch

Training all five models takes about 10 minutes on an RTX 5070 Ti.

## Main results (test set, 10 epochs)
| Model | Acc (%) | Macro-F1 | Params |
|---|---|---|---|
| Linear | 83.91 | 0.8371 | 7,850 |
| MLP | 88.12 | 0.8816 | 203,530 |
| CNN | 93.08 | 0.9310 | 3,397,578 |
| LSTM | 90.04 | 0.9004 | 17,043,466 |
| Transformer | 84.81 (87.33 @30 ep) | 0.8492 | 28,202 |

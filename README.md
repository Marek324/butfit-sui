# butfit-sui

butfit-sui holds three projects for the SUI course (Artificial Intelligence and Machine Learning) at FIT BUT, 2025/2026.

| Project | What it does |
|---|---|
| [`proj1`](proj1) | Binary classification of 6-feature data in NumPy |
| [`proj2`](proj2) | FreeCell solver with BFS, depth-limited DFS and A* |
| [`proj3`](proj3) | Conv1D and MaxPool1D layers for a tiny NumPy autograd library |

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/) for proj3, and a C++17 compiler with make for proj2.

## proj1: classifiers in NumPy

A notebook (`zadani.ipynb`) that compares a baseline, a Gaussian generative classifier (with and without class priors) and logistic regression trained by gradient descent.

```bash
cd proj1
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
jupyter lab zadani.ipynb
```

It ends with the test-set accuracies:

```
Baseline: 0.504
Generative classifier (w/o prior): 0.9615
Generative classifier (correct): 0.964
Logistic regression: 0.9645
logistic regression all features: 0.9195
```

## proj2: FreeCell search

The game engine comes from the course and the solvers are in `sui-solution.cc`.
All three solvers keep an explored set and walk parent pointers back to rebuild the solution.
A* orders its frontier by `depth + heuristic`.
Each solver gives up 50 MB below `--mem-limit` instead of getting killed.

```bash
cd proj2
make
./fc-sui 5 42 --solver bfs --easy-mode 10
./fc-sui 5 42 --solver a_star --easy-mode 20
make test
```

```
Solved 5 / 5 [ 100 % ]. Avg solution length 1.6 steps, ...
Solved 5 / 5 [ 100 % ]. Avg solution length 4 steps, ...
All tests passed (163 assertions in 21 test cases)
```

## proj3: conv and maxpool layers

`sui_torch.py` implements Conv1D and MaxPool1D on top of the provided `primitives.py`.
Conv1D does the forward pass with im2col and computes gradients with `np.convolve` / `np.correlate`.
`test.py` checks both layers against PyTorch, and `training.py` trains a small classifier to tell cosine, square and noise signals apart.

```bash
cd proj3
uv sync
uv run python -m unittest test
uv run python training.py          # add --do-plots to see predictions
```

## License

`proj2` (GPL-3.0) and `proj3` (BSD-3-Clause) are built on course code under the licenses in their `LICENSE` files.

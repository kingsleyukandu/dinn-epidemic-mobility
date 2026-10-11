# Recorded environment

The October 11, 2026 experiments used Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, PyTorch 2.14.1+cpu and Matplotlib 3.10.8. Notebook construction
used nbformat 5.11.1. The root `requirements.txt` consolidates these dependency pins;
the original dependency files remain in the historical source archive.

Use a separate Python 3.12 environment and install `requirements.txt`. The recorded
experiments used CPU execution with one thread per process; the constrained models
used float64. The original notebook uses its preserved float32 implementation.

Read the experiment protocols for configurations and runtime details. Fixed seeds
and dependency pins do not guarantee bitwise equality on other hardware or platforms.

Run the documented commands from the project root. New outputs belong in `runs/`
or another fresh directory. Recorded evidence folders are protected by the current runners.

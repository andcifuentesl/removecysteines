# RemoveCysteines
Remove cysteines from a protein sequences using evolutionary scale modeling (ESM) to account for conservation.

## Run on Google Colab 
<a href="https://colab.research.google.com/github/ckinzthompson/removecysteines/blob/main/remove_cysteines.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Make sure you select a (free) GPU instance, because it will be very slow on a CPU.


<video src='https://github.com/ckinzthompson/removecysteines/assets/17210418/f33df108-f6b2-46e7-9440-7fcd13a955f3' width=100/>






## Run locally (Python)
Requires either:
1. A new Apple Silicon computer
2. A computer with a CUDA-capable GPU

Steps:
* Download `remove_cysteine.py`
* Install torch, fair-esm, numpy, and matplotlib
* Run from the terminal:

```bash
> python remove_cysteines.py --help
usage: remove_cysteines.py [-h] [-n N_ROUNDS] [-p PCA] sequence

Remove Cysteines

positional arguments:
  sequence              WT protein sequence to alter

optional arguments:
  -h, --help            show this help message and exit
  -n N_ROUNDS, --n_rounds N_ROUNDS
                        Maximum Number of Polishing Rounds
  -p PCA, --pca PCA     Show embedding PCA?
```

## Example
![](https://github.com/ckinzthompson/removecysteines/assets/17210418/51e16705-c925-4da5-b1ea-e21b24bba5a1)

# Code to reproduce the results of the "Quantum Advantage in Sensing and Imaging" paper.
## Relational Fourier Sampling for Quantum Telescope Imaging

Minimal research code for reproducing three numerical results using **relational Fourier sampling (RFS)** under correlated atmospheric noise. 

## Files

| Figure | Python functions | Jupyter notebook |
|---|---|---|
| Binary-source Fisher information vs. interarrival time and optical dimension | `binary_fi.py` | `binary_time_dimension.ipynb` |
| Einstein-cross image reconstruction using RFS algorithm | `imaging_povm.py` | `Einstein_Cross_imaging.ipynb` |
| Moving-object tracking: 1D and 2D, 8-second prediction | `object_tracking.py` | `object_tracking_8s.ipynb` |

## Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebook for the desired figure and run all cells. Keep each notebook and its corresponding `.py` module in the same directory. The notebooks contain parameters, reproducibility settings, and plotting options.

## License

MIT License; see [`LICENSE`](LICENSE).

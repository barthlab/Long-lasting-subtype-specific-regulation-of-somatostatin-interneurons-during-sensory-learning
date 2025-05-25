# Long-lasting-subtype-specific-regulation-of-somatostatin-interneurons-during-sensory-learning


Analysis codes for paper: 

#### Long-lasting, subtype-specific regulation of somatostatin interneurons during sensory learning


## Dataset access

Please download two-photon imaging data from the following link:  [Link](https://doi.org/10.1184/R1/27098272.v1).

You can also find the dataset with the following DOI: 10.1184/R1/27098272.

## Reproduce the results

1. Unzip the downloaded dataset in the `data` directory.
2. Run /scripts/main.py

The folder structure should look like this:

```
.
├── data
│   ├── Calcium imaging
│   │   ├── Ai148_PSE
│   │   ├── Ai148_SAT
│   │   ├── Calb2_SAT
├── figures
├── src
│   ├── data_manager.py
│   ├── ...
├── script1_overview.py
├── script2_...
├── ...

```

## Python Dependencies

- `Python 3.11`
- `numpy`
- `colorist`
- `matplotlib`
- `scipy`
- `pandas`
- `scikit-learn`
- `seaborn`
- `tqdm`
- `xlwt`
- `xlrd`



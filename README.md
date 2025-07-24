# Long-lasting, subtype-specific regulation of somatostatin interneurons during sensory learning

Data analysis codes for the research paper on somatostatin interneuron regulation during sensory learning.

## Dataset Access

Please download the two-photon imaging data from the following link: [Two-photon imaging dataset (KiltHub)](https://doi.org/10.1184/R1/27098272.v3).

You can also find the dataset with the following DOI: **10.1184/R1/27098272**

## Quick Start

1. Unzip the downloaded dataset in the `data` directory
2. Run scripts 1-9 in sequential order

### Project Structure

The folder structure should look like this:

```text
.
├── data/
│   ├── Calcium imaging/
│   │   ├── Ai148_PSE/
│   │   ├── Ai148_SAT/
│   │   ├── Calb2_SAT/
│   │   └── Calb2_PSE/
│   ├── Behavior/
│   ├── Feature/
│   ├── Extracted Feature/
│   ├── Best Clustering/
│   └── Clustering Result/
├── figures/
│   ├── 1_raw_data/
│   ├── 2_overview/
│   ├── 3_plasticity_manifold/
│   ├── 4_diagram/
│   ├── 5_features/
│   ├── 6_main_figure/
│   ├── 7_examples/
│   ├── 8_justification/
│   └── 9_behavior/
├── src/
│   ├── basic/
│   ├── feature/
│   ├── ploter/
│   ├── behavior/
│   ├── data_manager.py
│   └── config.py
├── script1_overview.py
├── script2_plasticity_manifold.py
├── script3_diagram.py
├── script4_feature_candidate.py
├── script5_prediction.py
├── script6_example.py
├── script7_clustering_distance.py
├── script8_confusion_matrix.py
└── script9_behavior.py
```

## Analysis Scripts Documentation

### Main Analysis Scripts

1. **`script1_overview.py`** - Raw data visualization and overview analysis
   - **Purpose**: Generates overview visualizations of calcium imaging data including heatmaps and peak response analysis
   - **Outputs**: Raw data plots, heatmap overviews by cell type, peak complex visualizations
   - **Dependencies**: All experiments (Calb2_PSE, Calb2_SAT, Ai148_SAT, Ai148_PSE)

2. **`script2_plasticity_manifold.py`** - Plasticity manifold analysis
   - **Purpose**: Analyzes plasticity changes between Acclimation period and learning periods (SAT/PSE)
   - **Outputs**: Plasticity manifold plots comparing baseline to learning periods
   - **Dependencies**: Calb2_SAT and Calb2_PSE experiments only

3. **`script3_diagram.py`** - Diagram generation for figures
   - **Purpose**: Creates specific diagram visualizations for publication figures
   - **Outputs**: Overview diagrams, large view plots, trial diagrams
   - **Dependencies**: Calb2_SAT experiment, specific example FOV

4. **`script4_feature_candidate.py`** - Feature extraction and ranking
   - **Purpose**: Extracts features from calcium imaging data and ranks them by statistical significance
   - **Outputs**: Sorted feature names JSON files, feature hierarchy plots, distribution plots
   - **Dependencies**: Calb2_SAT experiment for feature ranking

5. **`script5_prediction.py`** - Main clustering and prediction analysis
   - **Purpose**: Performs dimensionality reduction, clustering, and generates main figure visualizations
   - **Outputs**: Embedding plots, clustering visualizations, feature summaries, fold-change analysis
   - **Dependencies**: All experiments, requires pre-computed feature rankings

6. **`script6_example.py`** - Example cell visualizations
   - **Purpose**: Generates example visualizations of individual cells and clusters
   - **Outputs**: Individual cell examples colored by cluster ID and cell type
   - **Dependencies**: Ai148_PSE and Calb2_PSE experiments

7. **`script7_clustering_distance.py`** - Clustering validation analysis
   - **Purpose**: Analyzes clustering quality and distance distributions
   - **Outputs**: Neighbor distribution plots for clustering validation
   - **Dependencies**: Calb2_SAT experiment, top features

8. **`script8_confusion_matrix.py`** - Classification performance evaluation
   - **Purpose**: Generates confusion matrix for cell type classification performance
   - **Outputs**: Confusion matrix heatmap with F1 score
   - **Dependencies**: None, just for revision figures

9. **`script9_behavior.py`** - Behavioral data analysis
   - **Purpose**: Analyzes behavioral performance data and correlates with imaging data
   - **Outputs**: Daily behavior summary plots, performance bars by cell clusters
   - **Dependencies**: Ai148_SAT experiment, requires clustering results

### Supporting Modules (`src/` Directory)

- **`src/config.py`** - Configuration parameters and experimental settings
- **`src/data_manager.py`** - Core data structures and experiment management
- **`src/data_status.py`** - Data status tracking and validation
- **`src/basic/`** - Basic utilities, terminology, and data operations
- **`src/feature/`** - Feature extraction, clustering, and dimensionality reduction
- **`src/ploter/`** - Visualization and plotting functions
- **`src/behavior/`** - Behavioral data processing and analysis

### Additional Scripts (`other_scripts/` Directory)

1. **`matlab_pertrial_analysis.m`** - MATLAB per-trial calcium imaging analysis
   - Processes Suite2p output for per-trial analysis of calcium signals
   - Requires MATLAB with Statistics and Signal Processing Toolboxes

2. **`sat_cage_code_arduino/sat_cage_code_arduino.ino`** - Arduino behavioral control code
   - Controls behavioral apparatus for somatosensory learning experiments
   - Requires Arduino IDE with FileIO library

## Software Requirements

### Python Environment

- **Python Version**: 3.11
- **Required Packages**:

  ```text
  numpy>=1.21.0
  matplotlib>=3.5.0
  scipy>=1.7.0
  pandas>=1.3.0
  scikit-learn>=1.0.0
  seaborn>=0.11.0
  umap-learn>=0.5.0
  tqdm>=4.62.0
  colorist>=1.4.0
  xlwt>=1.3.0
  xlrd>=2.0.0
  openpyxl>=3.0.0
  ```

### MATLAB Environment (for `matlab_pertrial_analysis.m`)

- **MATLAB Version**: R2019b or later
- **Required Toolboxes**:
  - Statistics and Machine Learning Toolbox
  - Signal Processing Toolbox
- **Input Files**: Suite2p output files (F.npy, Fneu.npy, iscell.npy, ops.npy)
- **Additional Files**: Arduino timing data in Excel format

### Arduino Environment (for behavioral control)

- **Arduino IDE**: 1.8.0 or later
- **Required Libraries**: FileIO library
- **Hardware**: Arduino-compatible board with relay control capabilities

## Expected Outputs

Each script generates specific outputs in the `figures/` directory:

- **`1_raw_data/`** - Raw data visualizations
- **`2_overview/`** - Overview heatmaps and peak analysis
- **`3_plasticity_manifold/`** - Plasticity analysis plots
- **`4_diagram/`** - Publication diagrams
- **`5_features/`** - Feature analysis plots
- **`6_main_figure/`** - Main clustering and embedding results
- **`7_examples/`** - Individual cell examples
- **`8_justification/`** - Clustering validation plots
- **`9_behavior/`** - Behavioral analysis results

## Notes and Support

- Full analysis workflow may take several hours depending on system specifications
- Consider running scripts individually for debugging
- README.md generated by LLM, manually checked and modified
- For any issues or questions, please contact Max at [xma3@andrew.cmu.edu](mailto:xma3@andrew.cmu.edu)
[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20-blue.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)
[![tests](https://github.com/erfanhamdi/pfm_dataset/workflows/tests/badge.svg)](https://github.com/erfanhamdi/pfm_dataset/actions) 
# Phase-field fracture benchmark dataset for solid mechanics
In this work, we have implemented multiple modeling choices for PFMs, specifically multiple energy decomposition methods, and have created a diverse, and challenging dataset of crack propagation inside a domain with varying initial crack locations and boundary conditions. Specifically, the dataset includes simulations from three energy decomposition methods, two distinct boundary conditions, and 1000 sets of initial conditions. For each sample, we publish 100 time steps, capturing the temporal evolution of crack propagation and providing rich data for training and evaluating machine learning models.
![dataset](Figs/dataset-nolabel.png)
This repository contains the code to reproduce the phase-field fracture benchmark dataset and the verification tests.

![dataset](Figs/github_readme.gif)


## Installation
* Clone the repository and create a conda environment from the environment.yml file
```bash
git clone https://github.com/erfanhamdi/pfm_dataset.git

cd pfm_dataset

conda env create --file environment.yml 

conda activate pfm-env

pip install -e .
```
## Usage
### With singularity container
* You can use the singularity container to run the code, to do that:
    1. Build the container
    ```bash
    singularity build --fakeroot pfm_dataset.sif pfm_dataset.def
    ```
    2. Shell into the container
    ```bash
    singularity shell pfm_dataset.sif
    ```
    3. clone the repository if needed and install the dependencies
    ```bash
    git clone https://github.com/erfanhamdi/pfm_dataset.git
    cd pfm_dataset
    pip install -e .
    pip install -r requirements.txt
    ```
    4. run the tests to check if the all the dependencies are installed correctly
    ```bash
    pytest
    ```
    5. Run the main script to generate the dataset
    ```bash
    python3 src/main.py
    ```
### Without singularity container
* Run the main script to generate the dataset
```bash
python3 src/main.py
```
## Citation
If you use this code or the datasets in your research, please cite:
```
@misc{hamdi2025robust,
      title={Towards Robust Surrogate Models: Benchmarking Machine Learning Approaches to Expediting Phase Field Simulations of Brittle Fracture}, 
      author={Erfan Hamdi and Emma Lejeune},
      year={2025},
      eprint={2507.07237},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.07237}, 
}
```
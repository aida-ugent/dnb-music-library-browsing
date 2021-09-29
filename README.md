# MIR tools for music library navigation

This repository contains some work-in-progress tools for navigating a personal (drum'n'bass) music library using MIR tools.

## Installation

Using conda, install the following dependencies:

```
conda create -n musicbrowsing python==3.7
pip install jupyter
pip install ipyfilechooser
pip install matplotlib
pip install plotly
pip install git+https://github.com/lenvdv/dnb-autodj-3
conda install -c conda-forge ffmpeg libsndfile
pip install spleeter
pip install -e drum-rhythm-browser/util_nmf_experiment/
```

Also install the NMFToolbox:  
First download from [this site](https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/), then install using:

```
pip install -e /path/to/NMFtoolbox/python
```

## Usage

Run `conda activate musicbrowsing`, then `jupyter notebook` in this directory. Then navigate to the subdirectories and run the demo notebook.
More info and instructions for each demo in the respective subfolder `README.md`s.

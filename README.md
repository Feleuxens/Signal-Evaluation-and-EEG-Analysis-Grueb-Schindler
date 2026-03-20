# Signal processing and Analysis of human brain potentials (EEG)

**Lynx-Lab-Rats**

Felix Grüb 3530441 st172408@stud.uni-stuttgart.de

Simon Schindler 3538650 st173013@stud.uni-stuttgart.de

## Setup 
For reproducibility and interoperability, `python-venv` and fixed library versions are used.

### EEG Data
The provided [data](https://nemar.org/dataexplorer/detail?dataset_id=ds004347) from our chosen [paper](https://nemar.org/dataexplorer/detail?dataset_id=ds004347) should be placed into `./data/` for our pipeline to be able to access everything.

### Pipenv
We used `pipenv` to manage the `python-venv` and libraries.

To install everything:
```
pipenv install
```

To run scripts:
```
pipenv run ./script.py
```

## Usage
To run a pipeline, use the scripts `./src/main.py` or `./src/blink_detection.py`.
After starting, you can select the specific pipeline config via the command line.


### Our Pipeline
To run our pipeline, trying to recreate what the original authors did, can be done by:
```
cd ./src/
pipenv run python ./main.py
```
And use config `1` on all subjects.


### Different configs
In `./config/` are all the configs to be used for our analysis of the impact of different pipeline steps.


### Blink analysis
All analyses about our deep dive into ASR and the impact of Blinks are separated into `./src/blink_detection.py`.
This is mainly because the pipeline is slightly different to be used for the needed comparisons.


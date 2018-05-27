# Environment Setup

## Anaconda Environment

[environment.yml](environment.yml) contains a list of the packages needed to recreate the conda environment.

```
conda env create -f environment.yml
```

## To install Box2D:

* clone the repo
    ```
    git clone https://github.com/pybox2d/pybox2d.git
    ```
* activate the conda environment
    ```
    activate gym-env
    ```
* change directory to pybox2d
    ```
    cd 'path/to/pybox2d'
    ```
* install pybox2d
    ```
    pip install -e .
    ```

## To install gym-vertical-landing:

* clone the repo
    ```
    git clone https://github.com/bbueno5000/gym-vertical-landing
    ```
* activate the conda environment
    ```
    activate gym-env
    ```
* change directory to pybox2d
    ```
    cd 'path/to/gym-vertical-landing'
    ```
* install gym-vertical-landing
    ```
    pip install -e .
    ```

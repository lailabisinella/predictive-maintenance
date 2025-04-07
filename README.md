# Dissertation Project 

## Setup

### Conda Environment
In order to run the following repository, it is necessary to set up the Python environment. The recommendation is to 
create a **virtual environment** which will not affect the base installation of Python.

If, as the project has been set up, you are using **conda** follow along the instructions, otherwise lookup the 
alternatives commands for your virtual environment manager.

After cloning the following repository into your machine, enter the folder and start by creating a virtual environment.
```shell
conda env create --file requirements.yml
```

This should create your conda environment named `predictive-failure` without any error. However, in order to make sure
the virtual environment has been created, you can run the following command and make sure the conda environment is
returned among the available ones.
```shell
conda info --envs
```

The last step is to finally activate the conda environment with all the packages already installed through the
`requirements.yml`.
```shell
conda activate predictive-failure
```

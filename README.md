[EAEE4000] Machine Learning for Environmental Engineers & Scientists

Andrew Fagerheim, Bernard Wang course project

Our project will involve working with ML models to study the submesoscale structure of buoyancy fluxes. We will be using a dataset provided by Dhruv Balwada of high-resolution inputs of physical 
terms related to buoyancy from the MITgcm-LLC4320 model. The output will be predictions of buoyancy flux.

Our goal is to test different models, input combinations, and hyperparameters to create an efficient output of buoyancy flux. Since performance is important to sub-grid model parametrization, it'll be an
important consideration in our project too. 

This repo contains the following files and directories: 
- funcs: directory of functions to create the CNN architecture for the pytorch model
- models: directory to save models created from the pytorch model
- diagnostics: notebook to visualize the output of the pytorch model and compares validation loss, R2 value, and # parameters
- get_data: notebook to load and visualize the input and output data
- submeso_ml_model: notebook to create and visualize the output of tensorflow model
- sweeps: notebook that attempted to run detailed hyperparameter sweeps with wandb
- torch_model: notebook to train pytorch CNN model

# DIRECTIONS:
1. run get_data.ipynb to retrieve the data and save it to a data folder.
2. run submeso_ml_model.ipynb to load the data, run the model, and evaluate the model.

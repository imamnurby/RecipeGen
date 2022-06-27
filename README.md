# *RecipeGen++* 

This is a replication package for our paper titled **RecipeGen++: An Automated Trigger Action Programs Generator**. 

# Important Files
- `train_merged_interactive.py` is a script to train a model in Interactive mode
- `train_merged_oneshot.py` is a script to train a model in One-Click mode
- `inference.ipynb` is a script to perform inference using the trained model and compute the metrics
- `gradio_app/app.py` contains *RecipeGen++* implementation

# Setting Environment
We provide a Dockerfile to instatiate the environment that we use. You can set up the environment by running `docker build Dockerfile --tag <name:tag>`.

# Training
To train a model (either Interactive or One-Click), you can simply run `python3 <script-name>`. The training settings can be changed by modifying the `args` initialization in the beginning of the script.

# Inference
Follow the instructions in `inference.ipynb` to perform inference using the trained model and compute the metrics. 
<br>*Do not forget to check the inference parameter in the beginning of the notebook.*

# Checkpoints and Result Artefacts
We release our model checkpoints and the corresponding inference results [here](https://zenodo.org/record/6668462#.YrAMh6hByUk).

# Prior Work
This tool is created based on our prior work that was accepted at ICPC 2022. For those who are interested in more comprehensive explanations and experiments, you can check the repo [here](https://github.com/imamnurby/RecipeGen-IFTTT-RP) and the paper [here](https://imamnurby.github.io/files/ICPC_CR_Version%20(4).pdf). 

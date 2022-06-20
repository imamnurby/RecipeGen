# *RecipeGen++* 

This is a replication package for our paper titled **RecipeGen++: An Automated Trigger Action Programs Generator**.
We release our model checkpoints and the corresponding inference results [here](https://zenodo.org/record/6668462#.YrAMh6hByUk).

# Important Files
- `train_merged_interactive.py` is a script to train a model in Interactive mode
- `train_merged_oneshot.py` is a script to train a model in One-Click mode
- `inference.ipynb` is a script to perform inference using the trained model and computing the metrics
- `gradio_app/app.py` contains *RecipeGen++* implementation

# Setting up the environment
We provide a Dockerfile to instatiate the environment that we use. You can setting the environment by running `docker build Dockerfile --tag <name:tag>`.

# Training
To train a model (either Interactive or One-Click), you can simply run `python3 <script-name>`. The training setting is can be changed by modifying `args` initialization in the beginning of the script

# Inference
- Follow the instruction in `inference.ipynb`. Do not forget to check the inference parameter in the beginning of the notebook.
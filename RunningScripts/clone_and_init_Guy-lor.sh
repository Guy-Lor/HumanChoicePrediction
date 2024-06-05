# cloning your code from github:
git clone https://github.com/Guy-Lor/HumanChoicePrediction.git
cd HumanChoicePrediction

# conda env create -f final_project_env_Guy-lor.yml
# conda activate final_project_env_Guy-lor
# wandb login
# Your main sweep:
python final_sweep_Guy-lor.py

# More runs appear in your report:
python sweep_1_HPT_ON_Guy-lor.py
python sweep_2_HPT_ON_Guy-lor.py
# python sweep_3.py

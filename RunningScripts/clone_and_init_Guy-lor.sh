# cloning your code from github:
git clone https://github.com/Guy-Lor/HumanChoicePrediction.git
cd HumanChoicePrediction
conda env create -f final_project_env.yml
conda activate final_project_env
# Your main sweep:
python final_sweep_Guy-lor.py

# More runs appear in your report:
# python sweep_1.py
# python sweep_2.py
# python sweep_3.py

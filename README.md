# Enhancing Human Choice Prediction by Improving Human Decision Simulation

This project is done as a part of NLP course and is based on the paper "Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation" (Eilam Shapira and Reut Apel and Moshe Tennenholtz and Roi Reichart).

Our project aims to enhance the accuracy of predicting human choices by improving the simulation of human decision-making processes. Current models use three primary simulation strategies, plus an “Oracle,” but these cannot fully capture the complexity of human decision-making. We reviewed scholarly articles on cognitive processes and decision-making behaviors to develop updated simulation strategies. By employing real decision-making strategies like loss aversion, we aim to create more accurate simulations. Experiments will validate our approach, expected to improve user choice prediction. we added 3 new strategies and the notebooks output the needed data and hyperparameters for creating the new strategies.

Strategy 1: Sentiment Ratio Analysis Using TripAdvisor Data
We calulated the positive to negative ratio using sentiment scores to reflect the DM's loss aversion behavior.

Strategy 2: Semantic Counter Strategy
We calculated the ratio of positive to total words and negative to total words to analyze the sentiment balance.

Strategy 3: Aspect-Based Sentiment Analysis
We divided reviews into eight aspects, generated binary vectors for positive and negative mentions, and scored reviews based on DM-specific preferences

The code is forked from the papers repository: https://github.com/eilamshapira/HumanChoicePrediction

1. Getting Started
2. Reproduce Results
3. Citation
   
## 1. Getting Started 


### Prerequisites

Before you begin, ensure you have the following tools installed on your system:
- Git
- Anaconda or Miniconda

### Installation

To install and run the code on your local machine, follow these steps:

1. **Clone the repository**

   First, clone the repository to your local machine using Git. Open a terminal and run the following command:
   ```bash
   git clone https://github.com/Guy-lor/HumanChoicePrediction
    ```
2. **Create and activate the conda environment**

    After cloning the repository, navigate into the project directory:

    ```bash
    cd HumanChoicePrediction
    ```

    Then, use the following command to create a conda environment from the requirements.yml file provided in the project:
    ```bash
    conda env create -f final_project_env_Guy-lor.yml
    ```
3. **Log in to Weights & Biases (W&B)**

   Weights & Biases is a machine learning platform that helps you track your experiments, visualize data, and share your findings. Logging in to W&B is essential for tracking the experiments in this project. If you haven't already, you'll need to create a W&B account. 
   Use the following command to log in to your account:
    ```bash
    wandb login
    ```
## 2. Reproduce Results
### Create Strategies Data

To reproduce the results, run the three added Jupyter notebooks to generate the necessary strategies data. The required outputs for a sweep run are already included in the repository, so you can skip this stage if you prefer to run the sweep using the existing data.

   1. **Sentiment and length ratio analysis**

      This Jupyter notebook processes hotel reviews to generate sentiment scores and analyze their characteristics. It outputs a CSV file, *combined_reviews_with_sentiment_scores.csv*, which includes each review's positive and negative sentiment scores, review               lengths, and the ratios of sentiment and length between positive and negative reviews. Additionally, it determines the threshold ratios for sentiment and review length based on an analysis of TripAdvisor hotel reviews, by splitting the data into training and        testing sets and examining the precision-recall curve.
 
            
   3. **Create reviews categories vectors**
      
      This Jupyter notebook analyzes customer reviews by sending them to OpenAI's ChatGPT-3.5 turbo model, which assesses the presence of 8 predefined topics: cleanliness, comfort, service, food, location, facilities, maintenance, and value for money. It takes as          input *combined_reviews_with_sentiment_scores.csv*, which includes each hotel review's positive and negative sentiment scores, review lengths, and sentiment and review length ratios. The notebook outputs *df_per_hotel_reviews_scores.csv*, containing all the data          from the input file along with a vector for each review representing the predefined topics.
      
   5. **Convert score df to dict pickle**
         
      This Jupyter notebook optimizes project runtime by processing the *df_per_hotel_reviews_scores.csv* file to extract and save only the IDs, ratios and topic vectors of each review. It converts this data into a dictionary format and stores it as a pickle file named *data_scores_dict.pkl* for efficient access.
### Run Sweep

   To run the sweep, execute *final_sweep_Guy-lor.py*. This will create a project in Weights & Biases. Copy the printed command(s) and run them in the terminal.

### Analyze Results

   Execute the modified *analyze_results.ipynb* and input your Weights & Biases (wandb) username, project name, and sweep ID. The notebook functions will then analyze the results and output the mean accuracy along with the confidence interval for the chosen hyper-parameters.

## 3. Citation

If you find this work useful, please cite our paper:

    @misc{shapira2024human,
          title={Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation}, 
          author={Eilam Shapira and Reut Apel and Moshe Tennenholtz and Roi Reichart},
          year={2024},
          eprint={2305.10361},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

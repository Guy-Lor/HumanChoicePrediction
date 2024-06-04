import wandb
YOUR_WANDB_USERNAME = "guylororg"
project = "NLP2024_PROJECT_Guy-lor"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "All 3 strategies added. check simulation_user_improve, basic_nature",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [True]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(1, 6))},
        # "seed": {"values": list(range(1, 2))},
        # "online_simulation_factor": {"values": [0, 4]},
        "online_simulation_factor": {"values": [4]},
        # "features": {"values": ["EFs", "GPT4", "BERT"]},
        "features": {"values": ["EFs"]},
        "basic_nature": {"values": [17, 18, 19, 20, 21, 22]},
        "simulation_user_improve": {"values":[0.01, 0.05]}
    },
    "command": command
}

sweep_config = {
    "name": "All 3 strategies added. Check double weight of added strategies",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [True]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(1, 6))},
        # "seed": {"values": list(range(1, 2))},
        # "online_simulation_factor": {"values": [0, 4]},
        "online_simulation_factor": {"values": [4]},
        # "features": {"values": ["EFs", "GPT4", "BERT"]},
        "features": {"values": ["EFs"]},
        "basic_nature": {"values": [23, 24, 25, 26]},
        "simulation_user_improve": {"values":[0.01, 0.05]}
    },
    "command": command
}

sweep_config = {
    "name": "Final sweep with HPT=False",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(1, 6))},
        # "seed": {"values": list(range(1, 2))},
        # "online_simulation_factor": {"values": [0, 4]},
        "online_simulation_factor": {"values": [4]},
        # "features": {"values": ["EFs", "GPT4", "BERT"]},
        "features": {"values": ["EFs"]},
        "basic_nature": {"values": [17, 20, 26]},
        "simulation_user_improve": {"values":[0.01]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)


print("Run these lines to run your agent in a screen:")
parallel_num = 6

if parallel_num > 10:
    print('Are you sure you want to run more than 10 agents in parallel? It would result in a CPU bottleneck.')
for i in range(parallel_num):
    print(f"screen -dmS \"final_sweep_agent_{i}\" nohup wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")

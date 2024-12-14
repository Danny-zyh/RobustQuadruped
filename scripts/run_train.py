from tqdm import tqdm
import subprocess

train_tasks = ["Isaac-Velocity-Flat-Base-Go2-v0", "Isaac-Velocity-Flat-RandForce-Go2-v0"] + [f"Isaac-Velocity-Flat-Entropy-p{i}-Go2-v0" for i in ["1", "05", "01", "2"]]

script_dir = "/home/danny/Documents/IsaacLab/source/standalone/workflows/rsl_rl/train.py"

for task in tqdm(train_tasks[1:]):
    result = subprocess.run(
        [
            "python", script_dir,
            "--task", task,
            "--num_envs", "4096",
            "--headless"
        ],
        capture_output=True,
        text=True,
        check=True,
    )

import os
import subprocess

model_checkpoint_dir = "/home/danny/Documents/IsaacLab/logs/rsl_rl/go2_flat_rand_force_full/2024-12-05_21-36-51"

test_task = "Isaac-Velocity-Flat-Test-Go2-v0"

output_dir = "/home/danny/Documents/sim2real/data/rsl_rand_full"

log_dir = "/home/danny/Documents/sim2real/log/rsl_rand_full"

script_dir = "/home/danny/Documents/IsaacLab/source/standalone/workflows/rsl_rl/test.py"

# Ensure log directory exists
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(model_checkpoint_dir):
    output_file_path = os.path.join(output_dir, file.split(".pt")[0] + ".json")
    if file.startswith("model") and not os.path.isfile(output_file_path):
        checkpoint_path = os.path.join(model_checkpoint_dir, file)
        log_file_path = os.path.join(log_dir, f"{file.split('.pt')[0]}.log")
        print("[INFO]: Evaluating policy at", checkpoint_path)

        try:
            print(" ".join([
                    "python", script_dir,
                    "--task", test_task,
                    "--checkpoint", checkpoint_path,
                    "--output", os.path.join(output_dir, file.split(".pt")[0]),
                    "--num_envs", "300",
                ]))
            result = subprocess.run(
                [
                    "python", script_dir,
                    "--task", test_task,
                    "--checkpoint1", checkpoint_path,
                    "--output", output_file_path,
                    "--num_envs", "300",
                    "--headless"
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Save stdout to the log file
            with open(log_file_path, "w") as log_file:
                log_file.write(result.stdout)

        except subprocess.CalledProcessError as e:
            print(e)
            # Log error details and stop execution
            error_log_file_path = os.path.join(log_dir, "error.log")
            with open(error_log_file_path, "w") as error_log_file:
                error_log_file.write(f"Command failed for checkpoint: {checkpoint_path}\n")
                error_log_file.write(f"Return Code: {e.returncode}\n")
                error_log_file.write(f"Stdout:\n{e.stdout}\n")
                error_log_file.write(f"Stderr:\n{e.stderr}\n")
            print(f"[ERROR]: Command failed for checkpoint {checkpoint_path}. Details logged in {error_log_file_path}.")
            raise  # Stop execution

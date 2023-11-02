import subprocess
import yaml
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(current_directory, "config", "config.yaml")
with open(config_file_path, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)
    
twist_angle = config["twist_angle"]
z_strain = config["z_strain"]
ktop_x = config["top_x_strain"]
ktop_y = config["top_y_strain"]
kbot_x = config["bot_x_strain"]
kbot_y = config["bot_y_strain"]


for i in range(30,33):
    z = float(i/10)
    config["z_strain"] = z
    with open(config_file_path, 'w') as yaml_file:
        yaml.safe_dump(config, yaml_file)
    z_strain = config["z_strain"]
    print(f"Now running with ktop_x = {ktop_x}, ktop_y = {ktop_y}, kbot_x = {kbot_x}, kbot_y = {kbot_y}, z_strain = {z_strain}, twist_angle = {twist_angle}")
    subprocess.run(["python", "main.py"])
    
    
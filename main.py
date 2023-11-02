import numpy as np
import yaml
import time, math, os, sys, random
import progressbar
import ctypes

from twist_strain_0827 import make_strain_twisted_sample
from disposition import tensor_matrix
from plot_tools import plot_tools
from lammps_setup import lammps_setup
import subprocess

start_time = time.time()
#setup directory info and config
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
MPI_threads = config["MPI_threads"]

folder_path = os.path.join(current_directory, f"./TBG_top_x_{ktop_x}_topy_{ktop_y}_botx_{kbot_x}_boty_{kbot_y}_zstrain_{z_strain}_twist_{twist_angle}")
config["folder_path"] = folder_path

if os.path.exists(folder_path):
    print("The folder path exists!")
else:
    os.mkdir(f"./TBG_top_x_{ktop_x}_topy_{ktop_y}_botx_{kbot_x}_boty_{kbot_y}_zstrain_{z_strain}_twist_{twist_angle}")

#make TBG sample
#updating sample_path_dir
sample_path = os.path.join(folder_path, "Strain.data")
config["sample_path"] = sample_path
with open(config_file_path, 'w') as yaml_file:
    yaml.safe_dump(config, yaml_file)
    
if os.path.exists(sample_path):
    print("The sample is made already.")
else:
    print("The sample hasn't been made yet.")
    make_strain_twisted_sample(ktop_x, ktop_y, kbot_x, kbot_y, twist_angle, z_strain)
    print("The TBG sample is made.")

#lammps setup
#updating relaxed_path_dir
relaxed_path = os.path.join(folder_path, "zig_TBG_relaxed.atom")
config["relaxed_path"] = relaxed_path
with open(config_file_path, 'w') as yaml_file:
    yaml.safe_dump(config, yaml_file)

if os.path.exists(relaxed_path):
    print("The sample has been relaxed!")
else:
    print("The sample hasn't been relaxed, now relax the sample.")
    lammps_setup(MPI_threads)
    
    
#determining the strain matrix
sample_freq = config["sample_freq"]
tm = tensor_matrix()
tm.strain_matrix(sample_freq)
print("The strain matrix is calculated.")


#ploting the tensor strength w.r.t. coordinate
os.chdir(folder_path)
pt = plot_tools()
strain_type = "Real part of lambda 1"
pt.plot_strain(strain_type, tm.coord_x_list, tm.lambda1_re_list)
strain_type = "Real part of lambda 2"
pt.plot_strain(strain_type, tm.coord_x_list, tm.lambda2_re_list)
strain_type = "Imagine part of lambda1"
pt.plot_strain(strain_type, tm.coord_x_list, tm.lambda1_img_list)
strain_type = "Imagine part of lambda2"
pt.plot_strain(strain_type, tm.coord_x_list, tm.lambda2_img_list)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")
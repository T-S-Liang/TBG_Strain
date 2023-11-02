import os
import yaml
from lammps import lammps
import subprocess

def lammps_setup(MPI_threads):
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
    
    folder_path = config["folder_path"]
    
    #conducting LAMMPS simulation
    os.chdir(folder_path)
    print("Writing in.relax file.")
    #write in.relax file (lammps relaxation file)
    content = f"""# Initialization
box tilt large
units           metal
boundary        p p f
atom_style      atomic
processors      * * 1    # domain decomposition over x and y

# System and atom definition
# we use different molecule ids for each layer
# so that inter- and intra-layer
# interactions can be specified separately 

read_data       Strain.data

mass       1 12.011  # C mass (g/mole) |
mass       2 12.011  # C mass (g/mole) |
mass       3 12.011  # C mass (g/mole) |
mass       4 12.011  # C mass (g/mole) |

# Separate atom groups
group layer1  type 1   # set A1  C   layer1
group layer1  type 2   # set B1  C   layer1
group layer2  type 3   # set A2  C   layer2
group layer2  type 4   # set B2  C   layer2

fix horiz1 layer1 setforce NULL NULL 0.0
fix horiz2 layer2 setforce NULL NULL 0.0

######################## Potential defition ########################
pair_style  hybrid/overlay  rebo  rebo kolmogorov/crespi/z 14.0
####################################################################
pair_coeff  * * none 
# C-C intralayer
pair_coeff  * * rebo 1 {current_directory}/CH.rebo C C NULL NULL
pair_coeff  * * rebo 2 {current_directory}/CH.rebo NULL NULL C C

# C-C interlayer
pair_coeff 1*2 3*4 kolmogorov/crespi/z {current_directory}/CC.KC   C C C C


# Neighbor update settings
neighbor        2.0 bin
neigh_modify    every 5 delay 0 check yes

# Output
timestep    0.001
dump xyzdump all custom 100 zig_TBG_relaxed.atom id type x y z                                         
dump_modify xyzdump format 3 "%14.10f" &
		    format 4 "%14.10f" &
		    format 5 "%14.10f"

dump forcedump all custom 100 dump_file.atom id type x y z fx fy fz
                                                    
thermo 100                                                                        
thermo_style custom step time etotal pe temp lx ly lz pxx pyy pzz
thermo_modify line one flush yes          &                                       
                       format  1 "%-6d"   &                                       
                       format  2 "%6.2f"  &                                       
                       format  3 "%14.4f" &                                       
                       format  4 "%14.4f" &                                       
                       format  5 "%8.2f"  &                                                                             
                       format  6 "%9.4f"  &                                       
                       format  7 "%9.4f"  &                                       
                       format  8 "%9.4f"  &                                       
                       format  9 "%12.4f" &                                       
                       format 10 "%12.4f" &                                       
                       format 11 "%12.4f" &

fix   thermos all nvt temp 0.01 0.01 0.01  

# Minimize settings ####
min_style cg                                                                   
minimize 1e-11 1e-7 1024 2048"""
# writing in.relax
    with open('./in.relax', 'w') as file:
        file.write(content)
        file.close()
    lammps_file_path = os.path.join(current_directory, "lammps_mpi.py")
    print("Now relax the sample.")
    subprocess.run(["mpiexec", "-n", str(MPI_threads), "python", str(lammps_file_path)])
    
    print("LAMMPS relax done!")
    os.chdir(current_directory)


if __name__ == "__main__":
    lammps_setup()
from cProfile import label
import math
from math import sqrt, pi
from typing import Tuple
from ase import io
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import same_color
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla
from scipy import  interpolate
import progressbar
import os
import yaml

coord_list_o = []
coord_list_f = []
#coord_list_o_lower = []
    
def init_atom():
    #setup directory info and config
    current_directory = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(current_directory, "config", "config.yaml")
    with open(config_file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    folder_path = config["folder_path"]
    relaxed_path = config["relaxed_path"]
    
    atoms = io.read(relaxed_path, format="lammps-dump-text", index=0)
    atoms_relaxed = io.read(relaxed_path, format="lammps-dump-text", index=-1)
    print("number of atoms:",len(atoms))

    lattice_vectors = atoms.cell
    a1= atoms.cell[0,:]              #: [Angstroms] Supercell vector
    a2= atoms.cell[1,:]              #: [Angstroms] Supercell vector
    print(a1,a2)

    '''
    Note that here we selected the bottom layer as study object.
    '''
    atom_size = int(len(atoms))
    bar1 = progressbar.ProgressBar(max_value=atom_size)
    for i in range(int(len(atoms))):
        z_coor = atoms[i].scaled_position[2]*lattice_vectors[2,2]       # z-coor in Angstroms
        if  3> z_coor > -3:
            coord_o = atoms[i].scaled_position
            coord_f = atoms_relaxed[i].scaled_position
            coords_o = np.zeros(3)
            coords_f = np.zeros(3)
            for j in range(3):
                coords_o += coord_o[j]*lattice_vectors[j,:]
                coords_f += coord_f[j]*lattice_vectors[j,:]
            carbon_dict_origin=np.array(coords_o)
            carbon_dict_final=np.array(coords_f)
            coord_list_o.append(carbon_dict_origin)
            coord_list_f.append(carbon_dict_final)
            
        #if  0> z_coor > -3:  # select layer
            #coord_o_lower = atoms[i].scaled_position
            #coords_o_lower = np.zeros(3)
            #for j in range(3):
                #coords_o_lower += coord_o_lower[j]*lattice_vectors[j,:]
            #carbon_dict_lower_origin=np.array(coords_o_lower)
            #coord_list_o_lower.append(carbon_dict_lower_origin)
        bar1.update(i + 1)

    kd_tree_o = KDTree(coord_list_o)

    pairs_o = kd_tree_o.query_pairs(r=2) #we condiser atoms within the radius of a c-c bond length
    print("n_pair:",len(pairs_o))
    #print(len(coord_list_o))
    #print(len(coord_list_o_lower))
    return pairs_o

#calculating the angle between vectors
def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cos_angle = dot_product / (norm_v1 * norm_v2)
    
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

#use this function to recover the vector to unrelaxed ones
def recover_to_unrelaxed(original_pos_vec):
    v_unrelax_1 = [0, -1.42, 0]
    v_unrelax_2 = [1.229756073, 0.71, 0]
    v_unrelax_3 = [-1.229756073, 0.71, 0]
    v_unrelax_4 = [0,1.42,0]
    v_unrelax_5 = [1.229756073, -0.71, 0]
    v_unrelax_6 = [-1.229756073, -0.71, 0]
    
    unrelaxed_vec_list = []
    for vector in original_pos_vec:
        if angle_between(vector, v_unrelax_1) < 30:
            unrelaxed_vec = v_unrelax_1
        elif angle_between(vector, v_unrelax_2) < 30:
            unrelaxed_vec = v_unrelax_2
        elif angle_between(vector, v_unrelax_3) < 30:
            unrelaxed_vec = v_unrelax_3
        elif angle_between(vector, v_unrelax_4) < 30:
            unrelaxed_vec = v_unrelax_4
        elif angle_between(vector, v_unrelax_5) < 30:
            unrelaxed_vec = v_unrelax_5
        elif angle_between(vector, v_unrelax_6) < 30:
            unrelaxed_vec = v_unrelax_6
        unrelaxed_vec_list.append(unrelaxed_vec)
        #print("The unrelaxed vector list: {}", unrelaxed_vec_list)
    return unrelaxed_vec_list

def get_tensor_matrix(atom_sequence, pairs_o):
    atom_sequence = atom_sequence - 1
    atom_neighbor_list = [] #defining the neighbor atoms of the atom of interest
    coord_of_this_atom = coord_list_o[atom_sequence]

    for pair in pairs_o:
        if pair[0] == atom_sequence:
            atom_neighbor_list.append(pair[1]) #store the sequence of neighborhood atom
            #print('found a pair')
        if pair[1] == atom_sequence:
            atom_neighbor_list.append(pair[0]) #store the sequence of neighborhood atom
            #print('found a pair')
    
    #print(f'The neighbor_atom_list is :{atom_neighbor_list}')
    
    original_pos_vec = [] #defining the list of position vector of original status
    final_pos_vec = [] #defining the list of position vector of final status
    
    for atom_neighbor in atom_neighbor_list:
        rij = coord_list_o[atom_sequence]-coord_list_o[atom_neighbor]
        rij_prime = coord_list_f[atom_sequence]-coord_list_f[atom_neighbor]
        original_pos_vec.append(rij)
        final_pos_vec.append(rij_prime)
    #here we recover the original relative position vector the the pristine regime.
    original_pos_vec = recover_to_unrelaxed(original_pos_vec)
    
    #print("The revocered relative postion vector is: {}",original_pos_vec)
    
    initial_vectors = np.array([[original_pos_vec[0][0], original_pos_vec[0][1]],
                                [original_pos_vec[1][0], original_pos_vec[1][1]]])

    transformed_vectors = np.array([[final_pos_vec[0][0], final_pos_vec[0][1]],
                                    [final_pos_vec[1][0], final_pos_vec[1][1]]])
    
    original_pos_vec[0] = np.array(original_pos_vec[0])-np.array(original_pos_vec[2])
    original_pos_vec[1] = np.array(original_pos_vec[1])-np.array(original_pos_vec[2])
    final_pos_vec[0] = np.array(final_pos_vec[0])-np.array(final_pos_vec[2])
    final_pos_vec[1] = np.array(final_pos_vec[1])-np.array(final_pos_vec[2])
    #print(original_pos_vec[0],final_pos_vec[0])
    #print(original_pos_vec[1],final_pos_vec[1])
    #constructing the equation set

    '''
    (x1, y1, 0 , 0 ) Sxx = x1'
    (0,  0 , x1, y1) Sxy = y1'
    (x2, y2, 0 , 0 ) Syx = x2'
    (0,  0 , x2, y2) Syy = y2'
    A_matrix dot S = b_matrix
    '''


    A_matrix = np.array([[original_pos_vec[0][0], original_pos_vec[0][1], 0 , 0],
                        [0, 0, original_pos_vec[0][0], original_pos_vec[0][1]],
                        [original_pos_vec[1][0], original_pos_vec[1][1],0,0],
                        [0, 0, original_pos_vec[1][0], original_pos_vec[1][1]]])

    b_matrix = np.array([final_pos_vec[0][0],final_pos_vec[0][1],final_pos_vec[1][0],final_pos_vec[1][1]])

    S = np.linalg.solve(A_matrix, b_matrix)


    S_matrix = S[:4].reshape(2, 2)

    Sxx = S_matrix[0][0]
    Sxy = S_matrix[0][1]
    Syx = S_matrix[1][0]
    Syy = S_matrix[1][1]
    
    eigenvalues, eigenvectors = np.linalg.eig(S_matrix)
    lambda1, lambda2 = eigenvalues
    v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]
    '''
    plt.figure()

    plt.plot([0, v1[0]], [0, v1[1]], label=f'eigenvector 1 (λ = {lambda1})', color='red')
    plt.plot([0, v2[0]], [0, v2[1]], label=f'eigenvector 2 (λ = {lambda2})', color='blue')

    plt.legend()

    plt.axis('equal')
    plt.savefig('eigenvectors.png')
    '''
    return coord_of_this_atom, v1,v2,lambda1,lambda2, Sxx, Sxy, Syx, Syy, original_pos_vec[0],original_pos_vec[1],original_pos_vec[2]

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

#defining the function for finding the nearest atom
def find_nearest_atom(atom_coords, point):
    nearest_atom = min(atom_coords, key=lambda x: distance(x, point))
    return nearest_atom

#defining the path and find the points along the way
def find_path_points(point1, point2, num_points):
    points_along_line = [point1 + i / (num_points - 1) * (point2 - point1) for i in range(num_points)]
    nearest_atoms = [find_nearest_atom(coord_list_o, point) for point in points_along_line]
    #print("points_along_line:", points_along_line)
    #print("find nearest atom:",nearest_atoms)
    index_list = []
    coord_array = np.array(coord_list_o)
    for path_atom_coord in nearest_atoms:
        index = np.where(np.all(coord_array == path_atom_coord, axis=1))[0][0]
        index_list.append(index+1)
    return index_list

class tensor_matrix():
    def __init__(self):
        self.coord_final_list = []
        self.coord_x_list = []
        self.coord_y_list = []
        
        self.v1_list = []
        self.v2_list = []
        self.lambda1_re_list = []
        self.lambda2_re_list = []
        self.lambda1_img_list = []
        self.lambda2_img_list = []
        
    def strain_matrix(self, sample_freq):
        pairs_o = init_atom() #initializing the atoms
        point1_sequence = 137064
        point2_sequence = 275508

    #note that point index is -1 of point sequence
        point1_sequence = point1_sequence - 1
        point2_sequence = point2_sequence - 1

        point1 = coord_list_o[point1_sequence]
        point2 = coord_list_o[point2_sequence]
        print('point1',point1)
        print('point2',point2)
        path_point_num = sample_freq
        path_points = find_path_points(point1, point2, path_point_num)
        print(path_points)

        #now calculate the second kind domain wall tensor evolution
        Sxx_list = []
        Sxy_list = []
        Syx_list = []
        Syy_list = []
        original_pos_vec_1_list = []
        original_pos_vec_2_list = []
        original_pos_vec_3_list = []

        bar = progressbar.ProgressBar(max_value=path_point_num)

        for i in range(path_point_num):
            coord_1, point_1_v1, point_1_v2,point_1_lambda1,point_1_lambda2, Sxx_1, Sxy_1, Syx_1, Syy_1,original_pos_vec_1,original_pos_vec_2,original_pos_vec_3 = get_tensor_matrix(path_points[i], pairs_o)
            self.coord_final_list.append(coord_1)
            self.coord_x_list.append(coord_1[0])
            self.coord_y_list.append(coord_1[1])
            self.v1_list.append(point_1_v1)
            self.v2_list.append(point_1_v2)
            self.lambda1_re_list.append(point_1_lambda1.real)
            self.lambda2_re_list.append(point_1_lambda2.real)
            self.lambda1_img_list.append(point_1_lambda1.imag)
            self.lambda2_img_list.append(point_1_lambda2.imag)
            Sxx_list.append(Sxx_1)
            Sxy_list.append(Sxy_1)
            Syx_list.append(Syx_1)
            Syy_list.append(Syy_1)
            original_pos_vec_1_list.append(original_pos_vec_1)
            original_pos_vec_2_list.append(original_pos_vec_2)
            original_pos_vec_3_list.append(original_pos_vec_3)
            bar.update(i + 1)
    
        print("Done with computing tensor matrix!")

        with open('filename.txt', 'w') as file:
            file.write('coord_x, coord_y, coord_z, v1, v2, lambda1_re, lambda2_re,lambda1_img, lambda2_img, Sxx, Sxy, Syx, Syy, original_pos_vec_1, original_pos_vec_2, original_pos_vec_3\n')
            for i in range(path_point_num):
                file.write(f'{self.coord_final_list[i][0]}, {self.coord_final_list[i][1]}, {self.coord_final_list[i][2]}, {self.v1_list[i]}, {self.v2_list[i]}, {self.lambda1_re_list[i]}, {self.lambda2_re_list[i]}, {self.lambda1_img_list[i]}, {self.lambda2_img_list[i]}, {Sxx_list[i]}, {Sxy_list[i]}, {Syx_list[i]}, {Syy_list[i]},{original_pos_vec_1_list[i]},{original_pos_vec_2_list[i]},{original_pos_vec_3_list[i]}\n')
            file.write(f'{path_points}\n')
            file.write('coord_x, coord_y, coord_z, lambda1_re, lambda2_re, lambda1_img, lambda2_img\n')
            for i in range(path_point_num):
                file.write(f'{self.coord_final_list[i][0]}, {self.coord_final_list[i][1]}, {self.coord_final_list[i][2]}, {self.lambda1_re_list[i]}, {self.lambda2_re_list[i]},{self.lambda1_img_list[i]}, {self.lambda2_img_list[i]}\n')

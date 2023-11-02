import matplotlib

import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import PathPatch

#strain_type is the type of lambda the input corresponding to
class plot_tools():
    def plot_strain(self, strain_type, strain_list, lam_list):
        plt.figure()
        plt.xlabel("coordinate")
        plt.ylabel("strain")
        plt.scatter(strain_list,lam_list, c = 'blue')
        plt.savefig(f"{strain_type}")
        plt.close()
        
    def draw_compression_arrow(self, ax, x_start, y_start, **kwargs):
        
        arrow1 = FancyArrowPatch((x_start, y_start), (0, 0),arrowstyle='-|>', mutation_scale=15, **kwargs)
        ax.add_patch(arrow1)
    
        arrow2 = FancyArrowPatch((-x_start, -y_start), (0, 0),arrowstyle='-|>', mutation_scale=15, **kwargs)
        ax.add_patch(arrow2)
    
    def draw_expansion_arrow(self, ax, x_start, y_start, **kwargs):
    
        arrow1 = FancyArrowPatch((0, 0), (x_start, y_start), arrowstyle='-|>', mutation_scale=15, **kwargs)
        ax.add_patch(arrow1)
    
        arrow2 = FancyArrowPatch((0, 0), (-x_start, -y_start),arrowstyle='-|>', mutation_scale=15, **kwargs)
        ax.add_patch(arrow2)
    
    def visualize_transform(self, coord_name, vec1, vec2, lam1, lam2):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        direction1 = np.arctan2(vec1[1], vec1[0])
        direction2 = np.arctan2(vec2[1], vec2[0])
    
        x1 = np.cos(direction1)*np.abs(lam1)
        x2 = np.cos(direction2)*np.abs(lam2)
    
        y1 = np.sin(direction1)*np.abs(lam1)
        y2 = np.sin(direction2)*np.abs(lam2)
    
    # 根据特征值决定箭头的颜色和方向
        if lam1 > 0:
            self.draw_expansion_arrow(ax, x1, y1, color="red")
        else:
            self.draw_compression_arrow(ax,x1, y1, color="blue")

        if lam2 > 0:
            self.draw_expansion_arrow(ax,x2, y2, color="red")
        else:
            self.draw_compression_arrow(ax,x2, y2, color="blue")

        plt.title("Transformation Visualization")
    
        plt.savefig(f'{coord_name}_tenor_matrix')
        plt.close()
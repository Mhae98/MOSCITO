# from _animated_plotter import AnimatedPlotter
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pyemma
from pyemma.coordinates import load as pyemma_load
from pyemma.coordinates.data import MDFeaturizer
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\heinr\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe'


def save_molecule_animation(save_path: str, traj_file: str, top_file: str = None, featurizer: MDFeaturizer = None):
    """
    Save animation of the trajectory of a molecule as a .gif file
    :param save_path: Path to the save location of the animation
    :param traj_file: Path to the trajectory file
    :param top_file: Path to the topology file
    :param featurizer: Featurizer if only selected parts of the molecule should be animated
    """
    if featurizer:
        data = pyemma_load(trajfiles=traj_file, features=featurizer)
    else:
        data = pyemma_load(trajfiles=traj_file, top=top_file)
    coordinates = data.reshape(data.shape[0], data.shape[1] // 3, 3)
    anim = get_animator(coordinates)
    writer = animation.FFMpegFileWriter(fps=30)
    anim.save(save_path, writer=writer)


def show_molecule_by_xyz(coordinates):
    anim = get_animator(coordinates)
    plt.show()


def animate(frame, data, graph, ax, start=0):
    graph.set_data(data[frame, :, 0], data[frame, :, 1])
    graph.set_3d_properties(data[frame, :, 2])
    ax.set_title(f'Timestep: {frame + start}', fontsize=24)


def get_animator(data, start=0) -> FuncAnimation:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Timestep: 0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    min_data = np.floor(np.min(data))
    max_data = np.ceil(np.max(data))
    ax.axes.set_xlim3d(left=min_data, right=max_data)
    ax.axes.set_ylim3d(bottom=min_data, top=max_data)
    ax.axes.set_zlim3d(bottom=min_data, top=max_data)
    graph, = ax.plot(data[0, :, 0], data[0, :, 1], data[0, :, 2], linestyle="", marker=".")
    return FuncAnimation(fig, func=animate, fargs=[data, graph, ax, start], frames=len(data), interval=1, save_count=len(data), cache_frame_data=False)


def get_indices(data, min_len):
    current_cluster_id = data[0]
    start_index = 0
    end_index = 0
    result = []
    for index, cluster in enumerate(data):
        if cluster != current_cluster_id:
            end_index = index
            if end_index - start_index >= min_len:
                result.append((start_index, end_index, cluster))
            current_cluster_id = cluster
            start_index = index
    end_index = len(data)
    if end_index - start_index >= min_len:
        result.append((start_index, end_index, data[len(data)]))
    return result


if __name__ == '__main__':
    feature = pyemma.coordinates.featurizer('../data/2f4k.gro')
    feature.add_selection(feature.select_Ca())
    save_molecule_animation('test.gif', '../data/tr8_folded.xtc', featurizer=feature)

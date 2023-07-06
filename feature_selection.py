from typing import Optional, Literal
import mdtraj as md
from mdtraj import Trajectory
import pyemma
import numpy as np
from pyemma.coordinates.data.feature_reader import FeatureReader
from pyemma.coordinates.data.featurization.featurizer import MDFeaturizer

# TODO: Remove warning suppression
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Class to simply get different features of a molecular dynamics simulation
    """

    def __init__(self, traj_file: str, top_file: str = None, reference=None):
        self.traj_file = traj_file
        self.top_file = top_file
        self.reference = reference
        self._icosphere = np.array([[0., 0.61803399, 1.],
                                    [0., -0.61803399, 1.],
                                    [0.61803399, 1., 0.],
                                    [-0.61803399, 1., 0.],
                                    [1., 0., 0.61803399],
                                    [-1., 0., 0.61803399],
                                    [-0., -0.61803399, -1.],
                                    [-0., 0.61803399, -1.],
                                    [-0.61803399, -1., -0.],
                                    [0.61803399, -1., -0.],
                                    [-1., -0., -0.61803399],
                                    [1., -0., -0.61803399]])

    def _load_data_md_traj(self, only_ca: bool = False) -> Trajectory:
        """
        Loads, centers and superposes trajectory using mdtraj
        :param only_ca: Set True to only get the coordinates of the c-alpha atoms
        :return: mdtraj Trajectory object
        """
        traj: Trajectory = md.load(self.traj_file, top=self.top_file)
        if only_ca:
            traj = traj.restrict_atoms(traj.topology.select('name == CA'))

        traj.center_coordinates()
        traj.xyz = (traj.xyz - np.mean(traj.xyz, axis=0)[np.newaxis, :, :]) / np.std(traj.xyz, axis=0)

        if self.reference is not None:
            traj.superpose(self.reference).center_coordinates(mass_weighted=True)
        else:
            traj.superpose(traj).center_coordinates(mass_weighted=True)
        return traj

    def get_xyz_all(self) -> np.ndarray:
        """
        For each frame get the xyz-coordinates of every atom
        :return: xyz-coordinates as a numpy array with shape (time-steps, n_atoms)
        """
        traj = self._load_data_md_traj(only_ca=False)
        return traj.xyz.reshape(traj.n_frames, -1)

    def get_xyz_c_alpha(self) -> np.ndarray:
        """
        For each frame get the xyz-coordinates of the c-alpha atoms
        :return: xyz-coordinates as a numpy array with shape (time-steps, n_atoms)
        """
        traj = self._load_data_md_traj(only_ca=True)
        return traj.xyz.reshape(traj.n_frames, -1)

    def get_backbone_torsions(self, cossin: bool = True) -> np.ndarray:
        """
        For each frame get all the backbone phi/psi angles
        :param cossin: If True, each angle will be returned as a pair of (sin(x), cos(x))
        :return: phi/psi angles as a numpy array
        """
        featurizer: MDFeaturizer = pyemma.coordinates.source(self.traj_file, top=self.top_file).featurizer
        featurizer.add_backbone_torsions(cossin=cossin)
        return pyemma.coordinates.load(self.traj_file, features=featurizer)

    def get_res_mindist(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get the minimal distance between residues
        :param threshold: distances below this threshold (in nm) will
            result in a feature 1.0, distances above will result in 0.0. If left to None, the numerical value will be
            returned
        :return: Minimal distances as a numpy array with shape (time-steps, ...)

        .. Note:: mdtraj default will select all pairs of residues separated by two or more residues
            (i.e. the i to i+1 and i to i+2 pairs will be excluded)
        """
        featurizer: MDFeaturizer = pyemma.coordinates.source(self.traj_file, top=self.top_file).featurizer
        featurizer.add_residue_mindist(scheme='closest-heavy', threshold=threshold)
        return pyemma.coordinates.load(self.traj_file, features=featurizer)

    def get_res_mindist_transformed(self, scheme: Literal['d1', 'd2', 'expd', 'log'] = 'd1', c=None) -> np.ndarray:
        """

        :param scheme:
        :param c:
        :return:
        """
        featurizer: MDFeaturizer = pyemma.coordinates.source(self.traj_file, top=self.top_file).featurizer
        featurizer.add_residue_mindist(scheme='closest-heavy', threshold=c)
        data = pyemma.coordinates.load(self.traj_file, features=featurizer)
        if c is None:
            if scheme == 'd1':
                return 1.0 / data
            elif scheme == 'd2':
                return 1.0 / np.sqrt(data)
            elif scheme == 'expd':
                return np.exp(-data)
            elif scheme == 'log':
                return np.log(data)
        else:
            res = np.zeros_like(data)
            # apply cut-off
            mask = data <= c
            res[mask] = 1.0
            return res

    def get_flex_torsions(self) -> np.ndarray:
        """
        Get the dihedral angles between the chi1 - chi5 dihedral
        :return: Numpy array containing sin and cos angles of the 5 dihedral angles
        """
        from mdtraj.geometry.dihedral import indices_chi1, indices_chi2, indices_chi3, indices_chi4, indices_chi5, \
            indices_omega
        featurizer: MDFeaturizer = pyemma.coordinates.source(self.traj_file, top=self.top_file).featurizer
        top = featurizer.topology
        indices = np.vstack((indices_chi1(top),
                             indices_chi2(top),
                             indices_chi3(top),
                             indices_chi4(top),
                             indices_chi5(top),
                             indices_omega(top)))
        assert indices.shape[1] == 4
        from mdtraj import compute_dihedrals

        def compute_side_chains(traj):
            res = compute_dihedrals(traj, indices)
            # cossin
            rad = np.dstack((np.cos(res), np.sin(res)))
            rad = rad.reshape(rad.shape[0], rad.shape[1] * rad.shape[2])
            return rad

        featurizer.add_custom_func(compute_side_chains, dim=len(indices) * 2)

        return pyemma.coordinates.load(self.traj_file, features=featurizer)

    def get_sasa(self, mode: Literal['atom', 'residue'] = 'residue') -> np.ndarray:
        """
        Get the solvent accessible surface area (SASA) of each atom or residue in each simulation frame
        :param mode: In mode == 'atom', the extracted areas are resolved per-atom. In mode == 'residue',
            this is consolidated down to the per-residue SASA by summing over the atoms in each residue
        :return: Solvent accessible surface area of each atom or residue as numpy array with shape
            (time-steps, n_atoms) or (time-steps, n_residues)
        """
        if mode not in ['atom', 'residue']:
            raise ValueError(f'Mode "{mode}" not supported. Only modes "atom" and "residue" are supported')

        def featurize(traj):
            res = md.shrake_rupley(traj, probe_radius=0.14, n_sphere_points=960, mode=mode, change_radii={'VS': 0})
            return res

        featurizer: MDFeaturizer = pyemma.coordinates.source(self.traj_file, top=self.top_file).featurizer
        if mode == 'atom':
            featurizer.add_custom_func(featurize, dim=featurizer.topology.n_atoms)
        elif mode == 'residue':
            featurizer.add_custom_func(featurize, dim=featurizer.topology.n_residues)
        return pyemma.coordinates.load(self.traj_file, features=featurizer)

    def get_shape_histogram_shell_model(self, n_shells: int = 10, only_ca: bool = False):
        """
        Creates a histogram for each frame of the trajectory using shells around the origin.
        Source paper: https://link.springer.com/chapter/10.1007/3-540-48482-5_14
        :param n_shells: Number of shells
        :param only_ca: If True, only calculates the histogram for c-alpha atoms, otherwise for all atoms
        :return: Numpy array of histograms with shape (timesteps, n_shells)
        """
        traj: Trajectory = md.load(self.traj_file, top=self.top_file)
        traj.superpose(traj).center_coordinates(mass_weighted=True)
        if only_ca:
            traj = traj.restrict_atoms(traj.topology.select('name == CA'))
        coordinates = traj.xyz
        # Calculate distances to center
        distances = np.linalg.norm(coordinates, axis=2)
        max_val = np.max(distances)
        shell_size = max_val / n_shells

        # For each frame calculate histogram
        histograms = []
        for frame in distances:
            hist = np.zeros(n_shells)
            for atom in frame:
                shell_id = min(int(atom // shell_size), n_shells - 1)
                hist[shell_id] += 1
            histograms.append(hist)
        return np.array(histograms)

    def _get_nearest_neighbors(self, coordinates, max_val) -> np.ndarray:
        """
        Assigns the atoms in a frame to the closest point in the list points
        :param coordinates: Coordinates of the data points
        :param max_val: Maximal value the icosphere should be scaled to
        :return: Numpy array containing the index of the closest point for each atom
        """
        # https://flothesof.github.io/k-means-numpy.html (Source for quick nearest neighbor calculation)
        scaled_icosphere = self._icosphere * max_val
        return np.argmin(np.sqrt(((coordinates - scaled_icosphere[:, np.newaxis]) ** 2).sum(axis=2)), axis=0)

    def get_shape_histogram_sector_model(self, only_ca: bool = False):
        """
        Creates a shape histogram based on a sector model from the 3d structure of the molecule
        Code based on the paper https://link.springer.com/chapter/10.1007/3-540-48482-5_14
        :param n_sectors: Number of sectors the molecule will be divided into
        :param only_ca: If True, only calculates the histogram for c-alpha atoms, otherwise for all atoms
        :return:
        """
        # Coordinates generated using https://pypi.org/project/icosphere/
        traj = self._load_data_md_traj(only_ca=only_ca)
        coordinates = traj.xyz
        max_coord = np.max(np.abs(coordinates))
        histograms = []
        for frame in coordinates:
            closest_ico_points = self._get_nearest_neighbors(frame, max_coord)
            indexes, counts = np.unique(closest_ico_points, return_counts=True)
            histograms.append(counts)
        # Normalize histograms between 0 and 1 for each frame
        histograms = np.array(histograms)
        for i in range(len(histograms)):
            current_frame = histograms[i]
            max_val = max(current_frame)
            normalized_frame = current_frame / max_val
            histograms[i] = normalized_frame

        return histograms

    def get_shape_histogram_combined(self, n_shells: int, only_ca: bool = False):
        """
        Calculates a histogram based on shells and sectors for each frame of the trajectory
        :param n_shells: Number of 3D shells
        :param only_ca: If True, only calculates the histogram for c-alpha atoms, otherwise for all atoms
        :return:
        """
        traj = self._load_data_md_traj(only_ca=only_ca)
        coordinates = traj.xyz
        max_coord = np.max(coordinates)

        distances = np.linalg.norm(coordinates, axis=2)
        max_distance = np.max(distances)
        shell_size = max_distance / n_shells
        n_sectors = len(self._icosphere)
        histograms = []
        for frame_index, frame in enumerate(coordinates):
            sector_ids = self._get_nearest_neighbors(frame, max_coord)
            shell_ids = [min(int(distance // shell_size), n_shells - 1) for distance in distances[frame_index]]
            combined_ids = np.array([sector_ids[i] + shell_ids[i] * n_sectors for i in range(len(sector_ids))])
            histogram_frame = np.zeros(n_shells * n_sectors)
            elements, counts = np.unique(combined_ids, return_counts=True)
            for index, element in enumerate(elements):
                count = counts[index]
                histogram_frame[element] = count
            histograms.append(histogram_frame)

        # Normalize histograms between 0 and 1 for each frame
        histograms = np.array(histograms)
        for i in range(len(histograms)):
            current_frame = histograms[i]
            max_val = max(current_frame)
            normalized_frame = current_frame / max_val
            histograms[i] = normalized_frame
        return histograms


if __name__ == '__main__':
    x = FeatureSelector('./data/tr8_folded.xtc', './data/2f4k.gro')
    hist = x.get_xyz_c_alpha()
    print(hist.shape)

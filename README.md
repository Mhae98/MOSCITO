# MOSCITO - Molecular Dynamics Subspace Clustering with Temporal Observance

Code for master thesis

## Requirements
For running the code the following libraries are required:
- [Pyemma 2.5.12](http://www.emma-project.org)
- [pypardiso 0.4.2](https://github.com/haasad/PyPardisoProject)

## Usage

### 1. Get the desired features
Depending on the format of the trajectory file a topology file has to be provided!
Available features:
- Coordinates of all atoms
- Coordinates of C-alpha atoms
- Backbone torsions
- Minimal distance between residues
- Solvent accessible surface area (SASA)
- Dihedral angles between the chi1 - chi5 dihedral

```python
from feature_selection import FeatureSelector

trajectory = 'path/to/trajectory_file'
topology = 'path/to/topology_file'
feature_selector = FeatureSelector(trajectory, topology)
feature = feature_selector.get_backbone_torsions()
```

### 2. Run TSC
```python
from moscito import MOSCITO

moscito = MOSCITO()
affinity = moscito.fit_predict(feature)
```

### 3. Run spectral clustering

```python
from sklearn.cluster import SpectralClustering

num_clusters = 10
sc = SpectralClustering(num_clusters, affinity='precomputed')
labels = sc.fit_predict(affinity)
```

### 4. Visualize clustering
```python
from visualization.visualize_clustering import show_clustering

show_clustering(labels)
```

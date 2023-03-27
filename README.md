# cc6
- GPU raycaster based on the six-direction box-spline (cc6)
- Implementation of the algorithm proposed in the article
  - Hyunjun Kim & Minho Kim, [``Volume Reconstruction with the Six-Direction Cubic Box-Spline''](https://doi.org/10.1016/j.gmod.2022.101168) Graphical Models, 125, (DOI: 10.1016/j.gmod.2022.101168) Jan. 2023
- Implemented in Python3 with OpenGL binding.
- How to run: >> python3 raycaster_cc6.py <volume_data_name>
  - volume_data_name: one of {"ML40", "ML50", "ML80"}.
- WARNING: install pyglm not glm.

# HR-Albedo: High-resolution albedo retrieval
# test s2gm
**Rui Song (rui.song@ucl.ac.uk)**

**University College London, Mullard Space Science Laboratory**

This github repo is the albedo retrieval module for the high-resolution 
albedo map (HR-AlbedoMap) project. The overall HR-AlbedoMap processing chain
also contains the Sensor invariant Atmospheric Correction (SIAC) module, and
the DeepLab v3+ cloud masking module.

Please refer to http://www.hr-albedo.org/ for more details about the HR-AlbedoMap
project. This project webpage contains a description of this project and downloadable 
hr-albedo products that are produced for different end users from this project. Relevant 
documents that are useful for understanding the algorithm (Algorithm Theoretical Basis Document and 
verification Report) and guidance on how to interpret the data (Product User Guide) are
also available from the project webpage.

# Install
All required packages for this project is list here [requirements.txt](./requirements.txt)

# Usage
Basic configurations see here [config.yaml](./config.yaml)

# how to run the hr-albedo retrieval
Once the configuration file is ready, using the following command to run the albedo 
retrieval directly:
#
**python main_albedo.py**

# Sample data
Sample data used in this example is available from: 
https://gws-access.jasmin.ac.uk/public/qa4ecv/hr_albedo/sample_data/



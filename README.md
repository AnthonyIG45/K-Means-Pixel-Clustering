This script implements a robust image segmentation approach using Repetitive K-Means clustering. 
By running the K-Means algorithm multiple times with randomized initial centroids, the system generates Probability Maps to identify skin regions against both simple and complex backgrounds.

Key Features
- Iterative Clustering: Runs K-Means 100 times (default) per image to account for the stochastic nature of centroid initialization.
- Label Alignment: Automatically sorts cluster labels by their Red-channel intensity to ensure consistent probability mapping across different iterations.
- Probability Maps: Calculates the likelihood ($P$) of a pixel belonging to a specific cluster, visualized via a "hot" colormap.Skin 
- Detection Mask: Generates a final segmentation where pixels with a probability $> 0.5$ are identified as the primary skin region.
- Multi-K Analysis: Tests various cluster counts ($k = 2, 3, 5$) to compare segmentation performance in different environments.

Prerequisites
You will need the following Python libraries installed:

       Bash
       pip install numpy opencv-python matplotlib

UsageInput Images:
- Ensure your input files are named Hand_Simple_Background.jpg and Hand_Complex_Background.jpg.
- 
Configuration:
- The script automatically resizes images to 750 x 750 for processing consistency.Execution:Bashpython segmentation_script.py

Outputs:
- Displays real-time probability maps using Matplotlib.Saves final results as output_simple_k[N].jpg and output_complex_k[N].jpg.

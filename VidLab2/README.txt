Laboratory Exercise Description:
This laboratory exercise focuses on advanced morphological image processing and object segmentation. The goal is to identify and extract specific objects - in this case, coins from a complex image using both grayscale and color information.

The main steps include:
1. Image preprocessing
	- Conversion to grayscale
	- Thresholding to create an initial binary mask
	- Morphological operations to clean the mask and fill holes. Two approaches are demonstrated:
	   Option 1 – using morphologyEx with a closing operation to fill small holes in the objects automatically.
	   Option 2 – manually applying dilation followed by erosion, which achieves a similar effect of closing gaps and smoothing object boundaries.
2. Color-based segmentation
	- Conversion to HSV color space
	- Extraction of the saturation channel to detect copper-colored coins
	- Thresholding and morphological filtering to create a marker for the object of interest
Color-based segmentation using HSV color space was decision that I independently made for this exercise because it takes advantage of additional information beyong brightmess, which grayscale intensity alone cannot provide. 
HSV separates hue (color), saturation (color strength), and value (brightness). By examining the saturation channel, you can detect objects with strong color presence (like copper) even if brightness varies. This allows a more precise selection of copper coins while ignoring silver coins or background areas, which may have similar grayscale intensity.
3. Morphological reconstruction
	-Iteratively dilating the marker while constraining it with the cleaned mask to extract the full object
	- Produces a clean segmentation of the coin within the mask
4. Mask application and visualization
	- Applying the reconstructed mask to the original image to isolate and visualize only the target object

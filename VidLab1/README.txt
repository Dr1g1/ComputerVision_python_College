Laboratory Exercise Description

This laboratory exercise focuses on image processing in the frequency domain, demonstrating how periodic noise can be removed using the 2D Fourier Transform.

The procedure includes the following steps:
1. Image loading and grayscale conversion
   The image is prepared for spectral analysis by converting it to grayscale.
2. Computing the 2D Fourier Transform
   The function `np.fft.fft2()` transforms the image from the spatial domain to the frequency domain, while `fftshift` centers the zero frequency component for easier interpretation.
3. Log-amplitude spectrum visualization
   A logarithmic scale is applied to the magnitude spectrum to make dominant frequency components more visible due to the wide dynamic range.
4. Noise detection and spectral modification
   Specific coordinates corresponding to periodic noise peaks are manually identified in the spectrum.
   Three filtering approaches are implemented:
	- amplitude attenuation
	- replacement with the average of direct neighbors
	- local Gaussian smoothing
5. Image reconstruction
   After modifying the spectrum, the inverse Fourier Transform is applied to reconstruct the filtered image back in the spatial domain.


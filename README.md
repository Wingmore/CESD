# Capacitance extraction from stability diagrams
CESD is a tool that extracts information from stability diagrams, namely gate to dot capacitances. This tool was created for the Andrew Dzurak research group in the University of New South Wales for use in their SET measurements and qubit experiments. 

Note this code was originally written in MATLAB and then rewritten in Python for its OpenCV package. The biggest advantage Python has over MATLAB is its abundance of online resources. Since Python is opensource and free, there is a massive online community with solutions to almost any and all topics. In terms of speed and flexibility, the code in Python felt both faster and easier to understand as data structures were not limited to arrays anymore. Personally, I felt a GUI in MATLAB was slower when compiling and processing, but this may be due to MATLAB's Appdesigner software.

## INSTALLATION
Requirements:
- Python 3.7+
- a few dozen packages

Extract all the files to a folder 'CESD' 

## USAGE
Unfortunately I have not been able to completely supercede the need for MATLAB since the original measurement data was acquired using MATLAB. Therefore, the first step is to run the Matlab2Python.m script. 

Throw everything in a folder called data. 

## CODE
Obviously not all the code will be explained here. This section will only explain the core processes in the `Cap_Extract.py` file. There are roughly 3 parts to this process
 

 1. Image Processing
 2. Data Processing and Manipulation
 3. Final Calculations

### 1. Image Processing

This section revolves around detecting where the lines in the D_I dataset is using a method called Hough Transform. Since the Hough transform is applied on images, the first step was to transform the data (stability diagram) to an image. The way this was done was to save load the the data from `Matlab2Python.m` using  

		from scipy import io
        loaded = io.loadmat(filename)
        D_I = loaded['D_I']
        FG_ST = loaded['FG_ST']
        mat = D_I['m'][0][0]
        mat2 = FG_ST['m'][0][0]

 where `filename` is the name of the output file, and `mat` contains the matrix data. This data is then saved as a `[PH].png` file using
					
		import matplotlib as plt
		plt.image.imsave("/test.jpg", mat3, origin='lower')
and then reread with Python's OpenCV package - the OpenCV contains the HoughTransform function. However, before applying the HoughTransform, some pre-processing had to be done, specifically

 1. Resizing -  HoughTransform does not work if the image/edges are too small
 2. Grayscale conversion and thresholding - to try filter out unecessary information
 3. Canny Edge detection - Creates a black and white edge image
 4. Morphological closing - "closes' gaps that may be caused by the Canny filter

        # Get image, rescale, convert to gray and blur
        img = cv2.imread(tmp_dir + "test.jpg")
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("test_resized.jpg", img)
       
        # Convert to grayscale,threshold and create edges
        kernel = np.ones((5,5),np.uint8)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh1,50,150,apertureSize = 3)
        fill = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
![N|Solid](https://photos.google.com/share/AF1QipOi8FdfMbrKt-NlOnnJPG5ILlaUYpbbDX740Khez3u8v6cN_lDEIG8Z-Q5flunMnQ/photo/AF1QipMVliqa6C5H01fQ3FT3ME8pqHDhV2PdoPVLgfXi?key=dnRQc0Raa3ExdGF6QThMcktCOFBLdVF6MFMtWUJ3)


![screenshot](https://github.com/Wingmore/CESD/blob/master/CESD_Documentation/pre-processing.png)


Finally the lines can be generated using OpenCV's [Probabilistic Hough Transform](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html). 

		lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 58,minLineLength = 50,maxLineGap = 100)
From the documentation, the input parameters are: 
>cv.HoughLinesP(dst, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
> -   _dst_: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
> -   _lines_: A vector that will store the parameters  (xstart,ystart,xend,yend)  of the detected lines
>-   _rho_  : The resolution of the parameter  r  in pixels. We use  **1**  pixel.
>-   _theta_: The resolution of the parameter  θ  in radians. We use  **1 degree**  (CV_PI/180)
>-   _threshold_: The minimum number of intersections to "*detect*" a line
>-   _minLinLength_: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
>-   _maxLineGap_: The maximum gap between two points to be considered in the same line.

Of particular importance are the last three parameters: `threshold`, `minLinLength` and `maxLineGap`. Since the 


## TODO

- Make a standalone compiler?
- Make it work in/independent of MATLAB
- do more testing
	- with double dot too
- add more error messages/edge cases
- add more gui functions
	- such as recountability
	- 

## MATLAB GUI (LEGACY/REDUNDANT)
The version created in MATLAB may still be used as it largely performs the same functions as the Python version. However, it does require multiple additional packages such as the Signal Processing and Curve Fitting Toolboxes which are both take up ~500mb of storage space - it took a while to download. Additionally, it is reliant on MATLAB's Appdesigner which feels super slow - especially when first starting up.

Anyhow, I have included the files for the interest of any. To use, launch MATLAB, type `appdesigner` in the console. After the Appdesigner app opens, find the `main.mlapp` file and open it, and then press the big green `RUN` button in the ribbon or press `f5`. 


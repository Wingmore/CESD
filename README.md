# Capacitance extraction from stability diagrams (CESD)
CESD is a tool that extracts information from stability diagrams, namely gate to dot capacitances. This tool was created for the Andrew Dzurak research group in the University of New South Wales for use in their SET measurements and qubit experiments. 

Note this code was originally written in MATLAB and then rewritten in Python for its OpenCV package. The biggest advantage Python has over MATLAB is its abundance of online resources. Since Python is opensource and free, there is a massive online community with solutions to almost any and all topics. In terms of speed and flexibility, the code in Python felt both faster and easier to understand as data structures were not limited to arrays anymore. Personally, I felt a GUI in MATLAB was slower when compiling and processing, but this may be due to MATLAB's Appdesigner software.

## INSTALLATION
Installation for code + editor:
1. Install Anaconda3.

2. Open anaconda prompt, create virtual environment via
	
	    conda create --name CESD_env python=3.8 spyder=4.1.1

3. Activate that environment

    	conda activate CESD_env

4. Clone from github/Extract all the files to a folder and navigate to your `CESD` install directory within conda

	    cd *path to CESD folder*

5. Install requirements

	    pip install -r requirements.txt

You should be able to edit and run everything using the Spyder IDE and/or the console and your own text editor.
## USAGE
Unfortunately I have not been able to completely supercede the need for MATLAB since the original measurement data was acquired using MATLAB. Therefore, the first step is to run the `Matlab2Python.m` script. 

There are two ways to do this:
1. Pressing f5 or run in MATLAB will convert all the files from a specified folder 'filepath' into another .mat file that can be read in Python. You will need to change the `filePath` line to the folder containing your measurements
2.  Alternatively if you only want to convert a single file, use the standalone script by typing `my_convert(filename, outPath)` in the console. (See `my_convert.m` for more information)

The resulting files should be in a directory called `/data/` which is the file you should select in the CESD Application in the next step.

To run the app, go to your virtual environment in the Anaconda console (step 3 in installation) and then navigate to the installation folder and then type

        python main.py

In the main window, click `browse` and select a file created from the `Matlab2Python.m` script, and then press start.  Clicking `Start` should start the processes automatically, and then a results window should pop up upon success.

![main_window](https://github.com/Wingmore/CESD/blob/master/CESD_Documentation/home.png)
![results_window](https://github.com/Wingmore/CESD/blob/master/CESD_Documentation/results.png)

It is also possible to use the `Cap_Extract` class without the main window by through the following Anaconda console commands (after navigating to the installation directory)
        
        python
        >>>> from ToRCode import Cap_Extract
        >>>> a = Cap_Extract("name of file")
        >>>> a.main_detection()
        >>>> a.show_results()

        
        

To edit anything, activate your virtual environment and run the Spyder IDE by typing `spyder` in the Anaconda console. Or just use your own text editor/ide. 

If the line-detection algorithm doesn't yeild accurate detection of lines there are some parameters that you can change in ToRCode.py. Find the following (~line 162)

        # Parameters
        hough_thresh = 60   #default 60. increase for less lines, decrease for more
        gray_thresh = 177   #default 177. decrease for fainter lines
        min_lines = 10  #default 10. Minimum number of lines you expect for a diagram

There are a few more that may be of interest but may not have as much of an impact on the line-detection. These are: Resized width and height, `shift_amt`, `minLinLength` and `maxLineGap`. These will be explained later.

There may also be an option to control these parameters in App at a later point in time.


## CODE
Obviously not all the code will be explained here. This section will only explain the core processes in the `main_detection` method found in the `Cap_Extract.py` file. There are roughly 3 parts to this process
 

 1. Image Processing
 2. Data Processing and Manipulation
 3. Final Calculations

### Theory
The theory is, whenever a "jump" occurs in the in a 1D row of the matrix `D_I` (`mat1`) current measurement, the `FG_ST` gate has to compensate and adjust the gate voltages for this jump. And so for a single row of the `FG_ST` matrix, there will be some "offset" which is at the same location as the jump in the `D_I` current. Now combining all the rows for the respective datasets, we get the following:

![D_I](https://github.com/Wingmore/CESD/blob/master/CESD_Documentation/D_I.jpg) ![FG_ST](https://github.com/Wingmore/CESD/blob/master/CESD_Documentation/FG_ST.jpg)

Evidently, it is much harder to see the offset in the `FG_ST` dataset than the `D_I` dataset, which is why we can use some image processing techniques to calculate the position of these lines from the `D_I` matrix and then use this position to find where each offset is for each row of `FG_ST` matrix. In the example above, there are 15 lines, so each row has 14-15 "jumps" which correspond to 14-15 offset locations.

After finding the locations of the offsets, we can segment the surrounding for each offset to get the following:
![offset](https://github.com/Wingmore/CESD/blob/master/CESD_Documentation/example_segment.JPG)
The final offset that we need is simply the vertical distance between the two slopes. i.e. the difference between y-intercepts.


### Part I - Image Processing

This section revolves around detecting where the lines in the D_I dataset is using a method called Hough Transform. Since the Hough transform is applied on images, the first step was to transform the data (stability diagram) to an image. The way this was done was to save load the the data from `Matlab2Python.m` using  

		from scipy import io
		
        loaded = io.loadmat(filename)
        D_I = loaded['D_I']
        FG_ST = loaded['FG_ST']
        mat = D_I['m'][0][0]
        mat2 = FG_ST['m'][0][0]

 where `filename` is the name of the output file, and `mat` contains the matrix data. This data is then saved as a `[PH].png` file using
					
		import matplotlib.image as mimg
		mimg.imsave("/test.jpg", mat3, origin='lower')
and then reread with Python's OpenCV package - the OpenCV contains the HoughTransform function. However, before applying the HoughTransform, some pre-processing had to be done, specifically

 1. Resizing -  HoughTransform does not work if the image/edges are too small
 2. Grayscale conversion and thresholding - to try filter out unecessary information such as noise
 3. Canny Edge detection - Creates a black and white edge image
 4. Morphological closing - "closes' gaps that may be caused by the Canny filter

        import cv2
        
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

The output from the following processes (Top Left to Bottom Right): Grayscale conversion, Threshold, Canny Edge Detection, Morphological Closing.
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

Of particular importance are the last three parameters: `threshold`, `minLinLength` and `maxLineGap`. Since the image size is now generally fixed, `minLinLength` and `maxLineGap` are chosen after some experimentation. `minLinLength` should be kept relatively long i. The `threshold` is the likeliest parameter you will need to change if the output is not desirable - increase `threshold` if you are detecting too many unwanted lines and decrease if you are not detecting the lines you want. 

The issue after running `HoughLinesP` is usually there are multiple variations of the same line that are detected, that is, some resulting lines overlap with one another. One way to overcome this is to lower the resolution of `rho` and `theta` arguments in `HoughLinesP` above. However a (better) method using code sourced from Stackoverflow [here](https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
) we can merge these lines in the line containing ` a.process_lines(lines, edges)`. This calls the uses the class `HoughBundler` in `HoughMerge.py` - full credits to Oleg Dats.

Finally, the last factor we can control is the shift between 'rows' of the Matrix. Usually alternating rows are not perfectly 'aligned' with one another due to hysteresis in the experimental measurements - the data is generated by sweeping the gate voltages up and down, which introduces hysteresis in the measurement device. To account for this, the `shiftMatrix` function is used (which was adapted from Henry Yang's shiftMatrix code in MATLAB). The full program then tries varying shift amounts (`shift_amt`) to determine which value will result in the cleanest output by looping from 0 to 10 and then trying from -10 to -1.

The full code for this part is then:

        import matplotlib.image as mimg
        import cv2
        from scipy import io
        from HoughMerge import HoughBundler

        loaded = io.loadmat(filename)
        D_I = loaded['D_I']
        FG_ST = loaded['FG_ST']
        mat = D_I['m'][0][0]
        mat2 = FG_ST['m'][0][0]
        
        width = 600
        height = 500
        
        shift_amt = 0
        # fig = plt.figure()
        best = 0
        best_l = []
        best_e = []
        a = HoughBundler()
        
        # Parameters
        hough_thresh = 60   #default 60. increase for less lines, decrease for more
        gray_thresh = 177   #default 177. decrease for fainter lines
        min_lines = 10  #default 10. Minimum number of lines you expect for a diagram
        while (1):
            mat3 = shiftMatrix(mat, shift_amt)

            mimg.imsave(tmp_dir +"/test.jpg", mat3, origin='lower')

            # Get image, rescale, convert to gray and blur
            img = cv2.imread(tmp_dir + "test.jpg")
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite("test_resized.jpg", img)
           
            # Convert to grayscale,threshold and create edges
            kernel = np.ones((5,5),np.uint8)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(gray,gray_thresh,255,cv2.THRESH_BINARY)
            edges = cv2.Canny(thresh1,50,150,apertureSize = 3)
            fill = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
            #Hough Transform - Actually generates the lines          
            lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 58,minLineLength = 50,maxLineGap = 100)
            
            # Save best line
            if lines is not None:
                if (len(lines) > len(best_l)):
                    best = shift_amt
                    best_l = lines
                    best_e = edges
                line_rem = a.process_lines(lines, edges)

            shift_amt += 1
            if shift_amt > 10:
                shift_amt = -10
            elif shift_amt == -1:
                line_rem = a.process_lines(best_l, best_e)
                break

### Part II - Data Processing and Manipulation

After detecting the lines using the above image processing techniques, the next step was to convert the data into some meaningful information and then use the line locations to extract the offset in the feedback control voltage matrix `FG_ST`. Now the output of  `HoughTransformP` and line merging method `process_lines` in `HoughBundler` is a list of points - [(x1,y1), (x2,y2)] -   containing the start and end of the lines that were detected in the resized image. Therefore, we had to resize these points again to match the original matrix dimensions (this is stored in `new_lines`) and then fill all the  points in between for later. The filled in points (`new_line_pts`), that is, a list of all the points corresponding to each row in the original matrix is generated using the bresenham algorithm. This process is repeated for each detected line.

Picture showing the resulting output of the bresenham drawing algorithm:
![bresenham](https://upload.wikimedia.org/wikipedia/commons/a/ab/Bresenham.svg)     

Code:

        # Create line joining two points using bresenham algorithm, also plot
        new_line_pts = []       # list of arrays that contain all the points for RESIZED TO ORIGINAL
        new_lines = []          # list of x1y1, x2y2 points that contain the start and end of lines RESIZED TO ORIGINAL
        y_scale = y_orig/height
        x_scale = x_orig/width
        S = [x_scale, y_scale]
        for idx, line in enumerate(line_rem):
            a = [int(round(x)) for x in line[1]*S]
            b = [int(round(x)) for x in line[0]*S]
            new_line_pts.append([a,b])
            new_lines.append(list(bresenham(*new_line_pts[-1][0], *new_line_pts[-1][1])))   

However, occasionally there are multiple points detected for a single row (shown in the bresenham pic), and thus we have to remove these points else the latter algorithm will fail (or there will be duplicate results). The following code simply checks the y-value for each point and then removes all the subsequent points with the same y-value resulting in only one unique point for each row of the matrix for that line. 

        #Remove duplicates so there is only one point in each row
        for idx, line in enumerate(new_lines):
            i = line[1][1]
            to_Remove =  []
            j = 0
            for pt in line:
                if pt[1] == i and j > 0:
                    to_Remove.append(pt)
                elif pt[1] != i:
                    i = pt[1]
                    j = 1
                else:
                    j += 1
            # line.remove(to_Remove)
            new_lines[idx] = [i for i in line if i not in to_Remove]
        
Now to calculate the capacitance later on, the  offset in the feedback control voltage matrix `FG_ST` (stored in `mat2`) is required. We do this by extracting "blocks" that surround the offset location for each row for a single line (which is the same as the offsets image in the Theory section above). For each block we then find the locations of the two longest lines and then the difference between the two y-intercepts are calculated from the line equations. This is repeated for each line.

        #%%  Generate the offsets between lines
        # Extract padded blocks - a single "block" is a row of mat2 given the location of the line generated from mat 1
        # In other words, this function extracts the elements surrounding a line in matrix2        
        blocks = []
        block_Hsize = 40    # block half size
        for idx, line in enumerate(new_lines):
            one_block = []
            for y in line:
                x = np.arange(max(y[0]-block_Hsize,0),min(y[0]+block_Hsize, x_orig))
                one_block.append(mat2[y[1]-1, x])
            blocks.append(one_block)
            
        derivative = np.diff(blocks[6]) 
        mode_m = (mode(mode(derivative, axis=1)[0])[0])
        print(mode_m)  
        slope_locations = []    #x-values where the gradient matches the mode
        offset = []     #list of the vertical offset before and after the "jumps" for each line
        sections = []   #array containing the x1y1, x2y2 points
        
        for block in blocks:
            slope_locations_tmp = []
            offset_tmp = []
            sections_tmp = []
            for row in block:
                derivative = np.diff(row)
                x = find_longest(derivative, mode_m)
                slope_locations_tmp.append(x)
                section1 = (x[0][0], row[x[0][0]], x[0][1], row[x[0][1]])
                section2 = (x[1][0], row[x[1][0]], x[1][1], row[x[1][1]])
                sections_tmp.append((section1, section2))
                offset_tmp.append(abs(getYInt(*section1)-getYInt(*section2)))
            slope_locations.append(slope_locations_tmp)
            offset.append(offset_tmp)
            sections.append(sections_tmp)
        



### Part III - Final Calculations
Code is rather self explanatory:
     
        x_intercepts = []
        y_intercepts = []
        for line in new_line_pts:
            x_intercepts.append(getXInt(*line[0], *line[1]))
            y_intercepts.append(getYInt(*line[0], *line[1]))
         
        #Delta Q = 1 electron charge
        DeltaQ = 1.60217662e-19;
        
        #chargine energy of Dot: assume 5 meV
        Edot = 5e-3;
        Cdot = DeltaQ/Edot; # e/eV
        
        #charging energy of SET: assume 20meV
        Eset = 20e-3;
        Cset = DeltaQ/Eset;
        
        #Capacitance between TG and SET
        Lset = 20   #SET Lever Arm
        Ldot = 20   #DOT Lever Arm (NOT USED)
        Ctgs = Lset*1e-2*Cset
        
        #capacitance between dot and SET
        DeltaV = mean_offset * 0.02; #the average of all the offset
        print("ΔVtgs = ", DeltaV, "V")
        Cds = DeltaV/Edot*Ctgs;
        
        #capacitance g1 and d
        DeltaV1d = np.mean(np.gradient(np.squeeze(y_intercepts)))
        Cg1d = DeltaQ/DeltaV1d*5;
        print("ΔVg1 = ", DeltaV1d, "* 0.2 V = ", DeltaV1d/5, "V")
        
        #capacitance g2 and dot
        DeltaV2d = np.mean(np.gradient(np.squeeze(x_intercepts)))
        Cg2d = DeltaQ/DeltaV2d*5;
        print("ΔVg2 = ", DeltaV2d, "* 0.2 V = ", DeltaV2d/5, "V")
        
        #Capactitance g2 and set
        Cg2s = 1e-19;
        
        #capacitance g1 and set
        m = 0.0010; #gradient of the TG map
        Cg1s = m*Ctgs;


## TODO

- Make a standalone compiler
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


import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
from scipy import io
from scipy.stats import mode
from IPython import get_ipython
import cv2
from HoughMerge import HoughBundler
from bresenham import  bresenham
from prettytable import PrettyTable
from dataclasses import dataclass

#constants
tmp_dir = "./tmp/"

@dataclass
class info:
    xmin: float
    xmax: None
    ymin: None
    ymax: None

@dataclass
class out:
    DeltaV1d: None
    DeltaV2d: None
    DeltaVtgs: None
    Energy: None
    DOT: None 
    SET: None
    G1: None
    G2: None
    TG: None

def refresh(mat, shift_amt, width, height):
    m = shiftMatrix(mat, shift_amt)
    mimg.imsave(tmp_dir +"tmp_refresh.jpg", m,origin='lower')
    img = cv2.imread(tmp_dir +"tmp_refresh.jpg")
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def shiftMatrix(m, shift_amt):
    '''SHIFTMATRIX 
    shifts the two matrices
     '''
    d = m.shape[1]
    shift = shift_amt*(-1)** np.arange(0,m.shape[0])
    mout = np.zeros(m.shape)
    if shift_amt  < 2:
        shift = shift_amt*np.ones(m.shape[0], dtype='int')

    for i in range(1,m.shape[0]):
        s = shift[i]
        if s > 0:
            mout[i,np.arange(0, d-s)] = m[i,np.arange(s,d)]
        else:
            mout[i,np.arange(-s, d)] = m[i,np.arange(0,d+s)]
    return mout



def find_longest(A, num):
    '''
     Parameters
    ----------
    A : Numpy Array
    num : element to find

    Returns
    -------
    index. Array of 2 longest consecutive lines/elements in array A
    function keeps looping until it has found all of the sections matching num

    '''
    index = []
    index_del = np.arange(0,len(A))
    while (1):
        count = 0
        prev = 0
        indexend = 0
        for i in range(0,len(A)):
            if A[i] - num < 0.000001:
                count += 1
                
                #check last element
                if i == (len(A)-1):
                    if count > prev:
                        prev = count
                        indexend = i+1    
                    count = 0
            else:            
              if count > prev:
                prev = count
                indexend = i
              count = 0
        index.append([index_del[indexend-prev], index_del[indexend-1]])
        A = np.delete(A,np.arange(indexend-prev,indexend))
        index_del = np.delete(index_del,np.arange(indexend-prev,indexend))
        if indexend == 0:
            break

    return (index[0], index[1])
    
def getGradient(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)
    
def getYInt(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b=y1-m*x1
    return b

def getXInt(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = getYInt(x1, y1, x2, y2)
    return -b/m

def show_im():
    img = cv2.imread(tmp_dir + "test.jpg")
    # fig, ax = plt.subplots()
    ax = plt.figure(figsize=(15,10))
    ax.imshow((img), interpolation='nearest', aspect='auto', origin='lower')

    return 0

#%% Capacitance Extraction from Stability Diagrams
class Cap_Extract():
    def __init__(self, in_file):
        self.filename = in_file
        
    def main_detection(self, filename=None):
        # Display options for Spyder
        # get_ipython().run_line_magic('matplotlib', 'qt')    #to plot in seqparate window in Spyder
        # get_ipython().run_line_magic('matplotlib', 'inline') #to plot in console inline
        
        filename = self.filename
        #get the data
        loaded = io.loadmat(filename)
        D_I = loaded['D_I']
        FG_ST = loaded['FG_ST']
        mat = D_I['m'][0][0]
        mat2 = FG_ST['m'][0][0]
        
        
        D_I_info = info(D_I['xmin'][0][0],D_I['xmax'][0][0],D_I['ymin'][0][0],D_I['ymax'][0][0])
        FG_ST_info = info(FG_ST['xmin'][0][0],FG_ST['xmax'][0][0],FG_ST['ymin'][0][0],FG_ST['ymax'][0][0])
        
        # Image sizes
        x_orig = mat.shape[1]
        y_orig = mat.shape[0]
        scale_percent = 220 # percent of original size
        width = 600 #int(img.shape[1] * scale_percent / 100)
        height = 500 #int(img.shape[0] * scale_percent / 100)
        
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
        
        # plt.imshow((mat), interpolation='nearest', aspect='auto', origin='lower')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # %% Part I - Image Processing / Line Detection
        while (1):
            mat3 = shiftMatrix(mat, shift_amt)
            # plt.imshow(mat, interpolation='nearest', aspect='auto')
            # plt.imshow(mat, interpolation='nearest', aspect='auto')
            
            mimg.imsave(tmp_dir +"/test.jpg", mat3, origin='lower')
            # mimg.imsave(tmp_dir +"/test2.jpg", mat, origin='lower')
            
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
        
            
            img_concate_Hori = np.concatenate((gray,thresh1),axis=1)
            img_concate_Vert = np.concatenate((edges, fill),axis=1)
            img_concate = np.concatenate((img_concate_Hori, img_concate_Vert), axis=0)
            # cv2.imshow('concatenated_H',img_concate)
        
            #Hough Transform - Actually generates the lines          
            lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 58,minLineLength = 50,maxLineGap = 100)
            
            # Save best line
            if lines is not None:
                if (len(lines) > len(best_l)):
                    best = shift_amt
                    best_l = lines
                    best_e = edges
                line_rem = a.process_lines(lines, edges)
                # if len(line_rem) > min_lines:
                    # break
                
            shift_amt += 1
            print("trying",shift_amt, "element shift")
            if shift_amt > 10:
                shift_amt = -10
            elif shift_amt == -1:
                print("returning: ", best)
                line_rem = a.process_lines(best_l, best_e)
                break
            
            
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Save image w/o lines
        mimg.imsave(tmp_dir +"/new_resized.jpg", img, origin='lower')
    
        
        # Create line joining two points using bresenham algorithm, also plot
        img = refresh(mat, best, width, height)
        new_line_pts = []       # list of arrays that contain all the points for RESIZED TO ORIGINAL
        new_lines = []          # list of x1y1, x2y2 points that contain the start and end of lines RESIZED TO ORIGINAL
        y_scale = y_orig/height
        x_scale = x_orig/width
        S = [x_scale, y_scale]
        for idx, line in enumerate(line_rem):
            # print ("{} - idk {} - {}".format(idx, line[0], line[1]))
            cv2.line(img,tuple(line[0]),tuple(line[1]),(0,255,255),3)
            a = [int(round(x)) for x in line[1]*S]
            b = [int(round(x)) for x in line[0]*S]
            new_line_pts.append([a,b])
            new_lines.append(list(bresenham(*new_line_pts[-1][0], *new_line_pts[-1][1])))   
        
        # cv2.imshow("edges", img)
        # plt.imshow(img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()   
        
        mimg.imsave(tmp_dir +"/lines.jpg", img, origin='lower')
    
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
         #%% Part II - Maths and my Algos - Generate the offsets between lines
           
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
                # Do i have to rescale the X??? not true current
                section1 = (x[0][0], row[x[0][0]], x[0][1], row[x[0][1]])
                section2 = (x[1][0], row[x[1][0]], x[1][1], row[x[1][1]])
                sections_tmp.append((section1, section2))
                offset_tmp.append(abs(getYInt(*section1)-getYInt(*section2)))
            slope_locations.append(slope_locations_tmp)
            offset.append(offset_tmp)
            sections.append(sections_tmp)
        
            
            
        tmp_offset = []
        for row in offset:
            tmp_offset.append(np.mean(row))
            
        mean_offset = (np.mean(tmp_offset))
        # derivative = np.diff(blocks[0][0])
        # x = find_longest(derivative, mode_m)
        
        
        # print(slope_locations)
        # plt.plot(blocks[0][0])
        # plt.plot(offset)
        #%% Reshape and remap points from  image resolution (0-x_orig)by(0-y_orig) pin pixels to measurement resolution (xmin-xmax)by(ymin-ymax) in amps
        orig_yrange = (D_I_info.ymax-D_I_info.ymin)
        orig_xrange = (D_I_info.xmax-D_I_info.xmin)
        orig_gradient = orig_yrange/orig_xrange
        new_gradient = []
        for row in new_line_pts:
            x1 = D_I_info.xmin + row[1][0]*orig_xrange/x_orig
            x2 = D_I_info.xmin + row[0][0]*orig_xrange/x_orig
            y1 = D_I_info.ymax - row[1][1]*orig_yrange/y_orig
            y2 = D_I_info.ymax - row[0][1]*orig_yrange/y_orig
            dx = x1 - x2
            dy = y1 - y2
            new_gradient.append(dy/dx)
            row = [[x1,y1],[x2,y2]]
            
        # print(new_gradient)
        
        
        #%% Part III - Final Calculations
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
        
        #capacitance to infinity - Dont need!
        #                 Cd = Cdot - Cds - Cg1d - Cg1s;
        #                 Cs = Cset - Cds - Cg1s - Cg2s - Ctgs;
        
        #Table Column data
        scale = 1e+18
        DOT = np.round(scale*np.asarray([Cdot, -Cds, -Cg1d, -Cg2d,0]), 4)
        SET =  np.round(scale*np.asarray([-Cds, Cset, -Cg1s, -Cg2s, -Ctgs]), 4)
        G1 =  np.round(scale*np.asarray([-Cg1d, -Cg1s, 0, 0, 0]), 4)
        G2 =  np.round(scale*np.asarray([-Cg2d, -Cg2s, 0, 0, 0]), 4)
        TG =  np.round(scale*np.asarray([0, -Ctgs, 0, 0, 0]), 4)
        
        t = PrettyTable([' ', 'Dot', 'SET', 'G1', 'G2', 'TopGate'])
        t.float_format = 4.4
        t.add_row(['Dot', *DOT])
        t.add_row(['SET', *SET])
        t.add_row(['G1', *G1])
        t.add_row(['G2', *G2])
        t.add_row(['TopGate', *TG])
        
        print("Capacitance Matrix (aF):")
        print(t)
        
        # Store some class properties
        results = out(DeltaV1d, DeltaV2d, DeltaV, (Edot*1e3, Eset*1e3, Ldot, Lset), DOT,SET,G1,G2,TG)
        
        self.mat0 = mat
        self.mat1 = mat
        self.meas_info = D_I_info
        self.lines = new_line_pts
        self.results = results
        
        
    def show_results(self):
        print("plotting...")
        xmin = self.meas_info.xmin
        xmax = self.meas_info.xmax
        ymin = self.meas_info.ymin
        ymax = self.meas_info.ymax
        

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # ax.imshow((img), aspect='auto')
        plt.imshow(self.mat0, interpolation='nearest', aspect='auto')

        ax.set_xlabel('G_G1 (0.2V)')
        ax.set_ylabel('G_G2 (0.2V)')
        ax.set_title('S_I')
        plt.rcParams.update({'font.size': 30})
        
        xlen = ax.get_xticks(0)
        xinc = (xmax-xmin)/(len(xlen)-3)
        xtick = np.round(np.arange(xmin-xinc, xmax+xinc,xinc), decimals=2)
        ax.set_xticklabels(xtick)
        ylen = ax.get_yticks(0)
        yinc = (ymax-ymin)/(len(ylen)-2)
        ytick = np.round(np.arange(ymin-yinc, ymax+yinc, yinc), decimals=2)
        ax.set_yticklabels(ytick)
        
        ax.invert_yaxis()


# a = Cap_Extract('Python_doubleDot.mat')
# a.main_detection()
# a.filename = 'asdf'
# # %%

# a.show_results()

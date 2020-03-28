# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:52:41 2020

@author: Sai
"""
import tkinter as tk
from tkinter import Tk, Button, Label, Frame, Entry, StringVar, filedialog, LabelFrame
from ToRCode import Cap_Extract
import os


    
def error(errcode):
    # popup success message
    popup = Tk()
    popup.title('Error')
    switcher = {
        1: "Error: No .mat files found",
        2: "Error: No input directory",
        4: "Error: File write error",
        5: "Error: DCP directory is invalid"
    }

    label = Label(popup, text=switcher.get(errcode, "what"))
    label.pack(pady=20)

    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()
    
def new_window(_class, in_Cap_Extract):
    new = tk.Toplevel(root)
    _class(new,in_Cap_Extract)
    
class App(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.grid_rowconfigure((0,1,2), weight=1)
        self.grid_columnconfigure((1,3), weight=0)
        Label0 = Label(self, text = "Welcome to ")
        Label0.grid(row=0, column=1, columnspan = 3)
        
        f_label = Label(self, text = "Matlab file:")
        f_label.grid(row=1, column=1, sticky="ew", padx=10)
        f_entry = Entry(self, textvariable=ovar)
        f_entry.grid(row=1, column=2, sticky="ew")
        f_button = Button(self, text = "Browse", command=input_newfile)
        f_button.grid(row=1, column=3, sticky="ew", padx=10, pady=10)
    
        s_button = Button(self, text="Start", command=lambda: CESD(in_file=f_entry.get()))
        s_button.grid(row=2, column=2, sticky="ew", pady=10, ipadx=20)
    
        self.grid_columnconfigure((0, 4), weight=1)
        self.grid_columnconfigure(2, weight=2)
        
        self.root = root
        self.root.title('CESD')

        # self.root["bg"] = "coral"
        # w_button = Button(self, text = "New window",
            # command= lambda: new_window(Results_window))
        # w_button.grid(row=2, column=3, sticky="ew", padx=10, pady=10)

# new window focus sourced from https://pythonprogramming.altervista.org/create-a-new-window-in-tkinter/
class Results_window:
    def __init__(self, root, in_Cap_Extract):
        # print(app.new.state())
        self.root = root
        # self.root.geometry("300x300+200+200")
        self.root.title('CESD Results')
        
        result_win = self.root
        result_win.grid_rowconfigure((0,3), weight=0)
        result_win.grid_columnconfigure((0, 2), weight=1)
        
        misc_frame = LabelFrame(root, width=100, height=50, padx=10, pady=10)
        misc_frame.grid(row=1, column=2, padx=10, pady=10)

        out_frame = LabelFrame(root, width=100, height=50, padx=10, pady=10)
        out_frame.grid(row=1, column=1, padx=10)

        table_frame = LabelFrame(root, width=100, height=50, padx=10, pady=10, text="Capacitance Matrix")
        table_frame.grid(row=2, column=1, padx=10, pady=10, columnspan=2)
        
        label_names = ['ΔVtgs', 'ΔVg1', 'ΔVg2','DOT Charging Energy (meV)', 'SET Charging Energy (meV)', 'DOT Lever Arm (%)', 'SET Lever Arm (%)']
        o_variables = []
        
        # Results from image processing and stuff  
        # source: https://stackoverflow.com/questions/18956945
        out = in_Cap_Extract.results
        o_variables.append(StringVar(self.root,value=out.DeltaVtgs))
        o_variables.append(StringVar(self.root,value=out.DeltaV1d))
        o_variables.append(StringVar(self.root,value=out.DeltaV2d))

        o_labels = []
        o_entrys = []
        for ii in range(3):
            o_labels.append(Label(out_frame, text=label_names[ii]))
            o_labels[-1].grid(padx=0, pady=0, row=ii, column=1, sticky='e')
            o_entrys.append(Entry(out_frame, textvariable=o_variables[ii], width=10))
            o_entrys[-1].grid(padx=0, pady=0, row=ii, column=2, sticky='w')        

        
        m_variables = []
        for i in range(4):
            m_variables.append(StringVar(self.root, value=out.Energy[i]))
            
        self.m_labels = []
        self.m_entrys = []
        for ii in range(4):
            self.m_labels.append(Label(misc_frame, text=label_names[ii++3]))
            self.m_labels[-1].grid(padx=0, pady=0, row=ii, column=1, sticky='e')
            self.m_entrys.append(Entry(misc_frame, textvariable=m_variables[ii], width=10))
            self.m_entrys[-1].grid(padx=0, pady=0, row=ii, column=2, sticky='w')
            print(ii, m_variables[ii])
        
        
        # Show Capacitance Matrix
        self.variables = []
        for i in range(25):
            self.variables.append([])
            
            
        #Fill out table entries
        for i in range(25):
            if (i+5)//5 == 1:
                self.variables[i] = StringVar(self.root, value=out.DOT[i%5])
            if (i+5)//5 == 2:
                self.variables[i] = StringVar(self.root, value=out.SET[i%5])
            if (i+5)//5 == 3:
                self.variables[i] = StringVar(self.root, value=out.G1[i%5])         
            if (i+5)//5 == 4:
                self.variables[i] = StringVar(self.root, value=out.G2[i%5])
            if (i+5)//5 == 5:
                self.variables[i] = StringVar(self.root, value=out.TG[i%5])
                
        self.labels = []
        self.entrys = []
        self.labels.append(Label(table_frame , text='DOT'))
        self.labels.append(Label(table_frame , text='SET'))
        self.labels.append(Label(table_frame , text='G1'))
        self.labels.append(Label(table_frame , text='G2'))
        self.labels.append(Label(table_frame , text='TopGate'))
        
        for i in range(5):
            self.labels[i].grid(padx=0, pady=0, row=0, column=i, stick="ew")
            
        for ii in range(25):
            self.entrys.append(Entry(table_frame, textvariable =self.variables[ii], width=10))
            self.entrys[-1].grid(padx=0, pady=0, row=(ii+5)//5+1, column=ii%5)



        # f_entry.grid(row=1, column=2, sticky="ew")
        # self.root["bg"] = "navy"
        
def input_newfile():
    # otext = filedialog.askdirectory(title='Choose file for input .mat')
    otext = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("mat files","*.mat"),("all files","*.*")))
    ovar.set(otext) 
    
def CESD(in_file=None):
    if in_file == None:
        error(1)
    # in_file = 'Python_DoubleDot.mat'
    a = Cap_Extract(in_file)
    a.main_detection()
    a.show_results()
    new_window(Results_window, a)

    
if __name__ == "__main__":
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    
    root = tk.Tk()
    ovar = StringVar(root)
    app = App(root).grid(sticky="nsew")
    root.mainloop()
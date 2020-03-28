%% Converts Matlab measuremnts to format readable in Python
% USAGE:
%   - Pressing f5 or run in matlab will my_convert all the files from a
%   specified folder 'filepath' into another .mat file. You will need to
%   change the filePath line to the folder containing your measurements
%   - Alternatively if you only want to convert a single file, use the
%   my_convert scripte by typing my_convert(filename, outPath) in the 
%   console. (See my_convert.m for more information)

close all, clear all, clc
addpath(genpath('..\'))

% Replace filePath with folder containing measurement data
filePath = 'I:\ToR office files\Taste of Ressearch 2018 - Edwin'; 

outPath = './data';
myFiles = dir(fullfile(filePath, '*.mat'));

extension = '.mat';

for k = 3:length(myFiles)
     baseFileName = myFiles(k).name;
     fileExtension = baseFileName(end-3:end);
     fullFileName = fullfile(filePath, baseFileName);
     fprintf(1, 'Now reading %s\n', fullFileName);
     if strcmp(fileExtension, extension)
         try
            my_convert(baseFileName, outPath, filePath)
         catch
            warning('error reading %s\n', fullFileName)
         end
     end
end

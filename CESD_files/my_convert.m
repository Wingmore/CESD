%Converts a single file to a format readable for Python
%USAGE:
%   my_convert(filename, outPath, filePath)
%   filename: name of .mat file. Can be just the name of the .mat file w/o
%       the folder path OR w/ the folder path. You dont need to specify a
%       filePath for the latter option
%   outPath: folder path to store output files. Should be in a
%       '../CESD/data/' folder. MAKE SURE FOLDER EXISTS FIRST'
%   filePath: (optional) Specify filepath if not already included in
%       filename
%
%EXAMPLES:
%   my_convert('EDSR_9297_2018.11.17.17.10.43_G1vsG2.mat', './data','C:\Users\Qudot')
%
%   my_convert('C:\Users\Qudot\EDSR_9297_2018.11.17.17.10.43_G1vsG2.mat', './data')   

function my_convert(filename, outPath, filePath)
    if ~exist('filePath', 'var')
        S = load(filename);
    else
        S = load([filePath '\' filename]);
    end
    obj.Data = S.meas.data{1,1}.rows;
%     table = table2struct(obj.Data, 'ToScalar', true);

    obj.DataSet = 'D_I';
    obj.Matrix = getMatrix(obj);
    D_I = obj.Matrix;
    obj.DataSet = 'FG_ST';
    obj.Matrix = getMatrix(obj);
    FG_ST = obj.Matrix;
    save([outPath '\CESD_' filename], 'D_I', 'FG_ST')
%     save('./Python/py_table.mat', 'table')
end


function A = getMatrix(obj)
    x = obj.Data.G_G1;
    y = obj.Data.G_G2;
    switch obj.DataSet
        case 'S_I'
            z = obj.Data.S_I;
        case 'D_I'
            z = obj.Data.D_I;
        case 'FG_ST'
            z = obj.Data.FG_ST;
        case 'control_feedback_error'
            z = obj.Data.control_feedback_error;
    end

    A = AutoMatrix(x, y, z);
    % Adapted from Henry's tool Automatrix:
    % Reshift D_I a bit so doesnt create zigzag effect, also remove the first row since its imcomplete data
    D_I = A;
    D_I = D_I.m(2:1:end,:)-mean(D_I.m(2:1:end,:),2);
    D_I1 = D_I(1:2:end,:);
    D_I2 = D_I(2:2:end,:);

    xc1 = xcorr2(D_I1, D_I2);

    [~,ix1]= max(xc1(size(D_I1,1)+0,:));
    [~,ix2]= max(xc1(size(D_I1,1)+0,:));
    shift = round((ix2-ix1)/2); %3 for other one
    D_I = shiftMatrix(D_I, 1*shift*(-1).^(1:size(D_I,1)));

    A .m= D_I;
end
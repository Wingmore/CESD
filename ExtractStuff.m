% -3 5 -9 -5
% * = things you can change
classdef ExtractStuff
   properties
      File
      ShowPlots = true
      DataSet = 'D_I'   %column in data. S_I, D_I etc
      Data
      Matrix
      MatrixPeaks %output 
      EqualWeightedPeaks = false % *
      LineEquations 
      NumLines = 20 %number of lines to test with *
      SweepWidth = 10 % *
      SweepSpacing = 10 % *
      
      Offset 
      OffsetLineThreshold = 10 %how much something should follow given % *
      OffsetPeakAdjust = 10 %modifies start of the sweeping alogorith  % *
      DeltaV
      repeatedLineThreshold = 10; %for removal of similar lines in generateLines  % *
      MatrixShift = 0
      ReallyBigData = 0
   end
   properties (Dependent)
      
   end 
   
%===================================================================
   methods
       function obj = set.File(obj,filename)
%             addpath('tools');
%             S = load(filename);
%             obj.Data = S.meas.data{1,1}.rows;
%             obj.File = filename;
% %             obj = runAll(obj);
%             obj.Matrix = generateMatrix(obj);

       end
%        function matrix = get.Matrix(obj)
%            matrix = obj.Matrix;
%        end
        function obj = set.MatrixPeaks(obj,matrix)  %idk why, have to put 'a' in otherwise it crashes
            obj.MatrixPeaks = matrix;
            if obj.ShowPlots
                plotPeaks(obj,20) %row 20
            end
        end
        function obj = set.LineEquations(obj, lines)
            obj.LineEquations = lines;
            if obj.ShowPlots
                plotLines(obj)
            end
        end
       function obj = set.Matrix(obj, matrix)
           fprintf('%s\n','ReWriting Matrix: ');
           obj.Matrix = matrix;
           if (obj.ShowPlots)
               plotMatrix(obj);
           end
       end
       function obj = set.Offset(obj, offset)
           obj.Offset = offset;
           obj.DeltaV = offset.difference;
           if (obj.ShowPlots)
               plotOffsets(obj);
           end
       end       
       function obj = set.DataSet(obj, name)
           obj.DataSet = name;
           obj.Matrix = generateMatrix(obj);
       end

       function obj = ExtractStuff(filename, shift)
           obj.File = filename;
           obj.MatrixShift = shift;
           
            addpath('tools');
            S = load(filename);
            obj.Data = S.meas.data{1,1}.rows;
            obj.File = filename;
%             obj = runAll(obj);
            obj.Matrix = generateMatrix(obj);
       end

       function matrix = generateMatrix(obj)
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
            
            shiftM = obj.MatrixShift;

            A = AutoMatrix(x, y, z);
            % Reshift D_I a bit so doesnt create zigzag effect, also remove the first row since its imcomplete data
            D_I = A;
            D_I = D_I.m(2:1:end,:)-mean(D_I.m(2:1:end,:),2);
            D_I1 = D_I(1:2:end,:);
            D_I2 = D_I(2:2:end,:);

            xc1 = xcorr2(D_I1, D_I2);
            
            [~,ix1]= max(xc1(size(D_I1,1)+0,:));
            [~,ix2]= max(xc1(size(D_I1,1)+0,:));
            shift = round((ix2-ix1)/2 + shiftM); %3 for other one
            D_I = shiftMatrix(D_I, 1*shift*(-1).^(1:size(D_I,1)));
            
            matrix = D_I;
       end

       function plotMatrix(obj, varargin)
            x = obj.Data.G_G1;
            y = obj.Data.G_G2;
            figure, 
            imagesc([min(x) max(x)],[min(y) max(y)],obj.Matrix)  %or use imshow?, 
            title(obj.DataSet,'Interpreter', 'none')
            xlabel(obj.Data.Properties.VariableNames(1),'Interpreter', 'none')
            ylabel(obj.Data.Properties.VariableNames(2),'Interpreter', 'none')
            set(gca,'YDir','normal')
       end
       
       function peak = generateMatrixPeaks(obj)
            Z = obj.Matrix;
            
            %remove values less than 0...
            if (obj.ReallyBigData)
%                 Z(find(Z > 0)) = 0;
            end
            
            peak = zeros(size(Z,1),size(Z,2));
            for i = 1:size(Z,1)
                [pk, lc] = findpeaks(-Z(i,:),'MinPeakProminence',1e-12,'Annotate', 'extents');
        %         lc = [lc, zeros(1,n - size(pk,2))];
                peak(i,lc) = pk;
            end
            
            % make new matrix w/ so give all points an equal weighting
            newPeak = abs(peak);
            if obj.EqualWeightedPeaks
                newPeak(peak ~= 0) = 1;       %gives all the points the same value
            end
            
%             obj.MatrixPeaks = newPeak;  %update property
            peak = newPeak;                %return adjusted poi9nts
       end
       
       function plotPeaks(obj, row, Z)
           
           if nargin < 2
                row = 1;
           end 
           if nargin < 3
                Z = obj.Matrix;
    
           end
           figure,plot(Z')
           if (obj.ReallyBigData)
%                Z(find(Z > 0)) = 0;
           end
%            findpeaks(-Z(row,:),'MinPeakProminence',4e-12,'Annotate', 'extents')
%            imagesc(obj.MatrixPeaks);
           
           %plot initial thing
            f = figure;
            ax = axes('Parent',f,'position',[0.13 0.39  0.77 0.54]);
            figureH = plot(Z(row,:))        %figure handler
            ylim([min(min(Z)),max(max(Z))]);
            TextH = title(sprintf('peaks along of %s', obj.DataSet),'Interpreter', 'none');

            %slider setup
            SliderH  = uicontrol('Parent',f,'Style','slider','Position',[81,54,419,23],...
                          'value',size(Z,1), 'min',1, 'max',size(Z,1)); %actual slider
            bgcolor = f.Color;
            bl1 = uicontrol('Parent',f,'Style','text','Position',[50,54,23,23],...
                            'String','1','BackgroundColor',bgcolor);%text to left of slider
            bl2 = uicontrol('Parent',f,'Style','text','Position',[500,54,23,23],...
                            'String',size(Z,1),'BackgroundColor',bgcolor);%text to the right
            bl3 = uicontrol('Parent',f,'Style','text','Position',[240,25,100,23],...
                            'String','thing','BackgroundColor',bgcolor);%text to display value of lsider

            addlistener(SliderH, 'Value', 'PostSet', @(source, eventdata) callbackfn(source, eventdata,Z,figureH, bl3));
            movegui(f, 'center')

            function callbackfn(source, eventdata,Z, figureH, bl3)
                num          = round(get(eventdata.AffectedObject, 'Value'));
                bl3.String = num2str(num);
                set(figureH, 'ydata', Z(num,:));
                drawnow;
            end
       end
       
       function p = generateLines(obj)
            %% BruteForce part1
            %start from top left, trace from bot left to bot right
            %and then sweep/repeat going from top left to top right
            % may have to section the graph and sweep later
            Z = obj.Matrix;
            sizeX = size(Z, 2); %x axis size
            sizeY = size(Z, 1); %y axis size
            endX = [0 5];
            startX = [0 0];
            startY = [0 0]; %fixed
            endY = [size(Z,1) size(Z,1)]; %fixed

            width = obj.SweepWidth;
            space = obj.SweepSpacing;  %should be the same - change for more iterations (data sets)

            sumPeak = zeros(1, 3);
            for j = 1:space:sizeX
                startX = [j j+width];      
                for i = max(1, j-sizeX/2):space:j  %CHANGE START THING BIT
                    Z = obj.MatrixPeaks;
                    x = [startX endX];         %[topL topR botR botL]
                    y = [startY endY];
                    endX = [i+width i];     
                    bw = poly2mask(x,y,sizeY,sizeX);
                    Z(~bw) = 0;         %apply mask
            %         imshow(bw)
            %         imagesc([min(x) max(x)],[min(y) max(y)],Z);
                    sum(Z(:));
                    sumPeak = [sumPeak; [i j sum(Z(:))]] ; %column1 is the i (endX), column2 is the j (startX)
            %         pause(0.0001) %animation~
                end
            end

            %% BruteForce part2
            %start from top right corner and then continue down along right edge (sweep
            % startX is fixed
            % endY also fixed

            startX = [sizeX sizeX];
            endY = [size(Z,1) size(Z,1)];
            sumPeak2 = zeros(1, 3);
            
            %idk what i did here lol... something about finding biggest x
            %value on the bottom to start the sweep
            n = obj.NumLines;     %number of max stuff
            [B, I] = maxk(sumPeak(:,3),n);  %get biggest n result indexs
            lineA = sumPeak(I,:);   %throw them in a vector based of above index
            startXPart2Sweep = max(lineA(:,1))-50;
            
            
            if (obj.ReallyBigData == 0)
                space = 2;      %IDK
                width = 2;
            end
            
            for j = 1:space:sizeY
                startY = [j j+width];      
                for i = startXPart2Sweep:space:sizeX   %maybe change starting bit
                    Z = obj.MatrixPeaks; 
                    x = [startX endX];  %[topL topR botR botL]
                    y = [startY endY];
                    endX = [i+width i];     
                    bw = poly2mask(x,y,sizeY,sizeX);
                    Z(~bw) = 0;         %apply mask
            %         imshow(bw)
            %         imagesc([min(x) max(x)],[min(y) max(y)],Z);
                    sum(Z(:));
                    sumPeak2 = [sumPeak2; [i j sum(Z(:))]]; %column1 is the i (endX), column2 is the j (startX)
            %         pause(0.0001) %animation~
                end
            end
            % plot(sumPeak2)

            %% get lines with the biggest sum
            n = obj.NumLines;     %number of max stuff
            [B, I] = maxk(sumPeak(:,3),n);  %get biggest n result indexs
            lineA = sumPeak(I,:);   %throw them in a vector based of above index
            x = [lineA(:,2)'; lineA(:,1)'];     %create vectors to draw [1stPoint; 2ndPoint]
            y = [zeros(size(lineA,1),1)'; sizeY*ones(size(lineA,1),1)'];
            z = B';
            %% get 2nd part
            n = obj.NumLines*2;
            [B, I] = maxk(sumPeak2(:,3),n);
            lineB = sumPeak2(I,:);
            p.x = [x [lineB(:,1)'; sizeX*ones(size(lineB,1),1)']];
            p.y = [y [sizeY*ones(size(lineB,1),1)'; lineB(:,2)']];
            p.z = [z B'];

            %%
            %get rid of repeated lines -> check the sum, easiest way i think
            checkS = (x(1,:)+x(2,:)); %maybe dont use round?
            last = size(checkS,2);
            toRemove = [];
            repeatedLineThreshold = obj.repeatedLineThreshold;
            for i = 1:last
                temp = checkS(i);
                toRemove = [toRemove find(checkS([i+1:last]) > temp-repeatedLineThreshold & checkS([i+1:last]) < temp+repeatedLineThreshold)+i ];  %5 is the threshold
            end
            %remove similar columns (based of a sum)
            index = true(1, size(x, 2));
            index(toRemove) = false;
            
            x = x(:,index);
            y = y(:,index);
            z = z(:,index);

            %% sort stuff
            [temp, order] = sort(x(1,:));
            p.x = x(:,order);
            p.y = y(:,order);
            p.z = z(:,order);
            %% get rid of intersecting lines NOT SURE IF IT DOES ANYTHING THO
            last = size(x,2);
            toRemove = [];
            for i = 1:last
                  line1 = [x(:,i), y(:,i)];
                  for j = min(i+1,last): min(i+10, last)
                      line2 = [x(:,j), y(:,j)];
                      isIntersect = isIntersecting(line1, line2);
                      if isIntersect
                         toRemove = [toRemove j];
                      end
                  end
            end
            index = true(1, size(x, 2));
            index(toRemove) = false;
            
            p.x = x(:,index);
            p.y = y(:,index);
            p.z = z(:,index);
       end
       
       function plotLines(obj)
            figure, imagesc(obj.MatrixPeaks);
            hold on, line(obj.LineEquations.x,obj.LineEquations.y)
            legend(strcat('sum =',num2str(obj.LineEquations.z')))
            title('Extracted Peaks')
            % figure, imagesc(AutoM);
            % hold on, line(x1,y1)
            %% get equation of lines - to check not needed

%             p = polyfit([p.x(1,1), p.x(2,1)], [y1(1,1), y1(2,1)], 1);
%             plot(p);
%             % xlim([x1(2,1) x1(1,1) ]);
%             % ylim([y1(1,1), y1(2,1)]);
%             polyval(p, 20)
%             % x=0:600;
%             % x = polyval(x, polyval(p,x))
%             % figure, plot(x)
%             p = polyfit([y1(1,1), y1(2,1)], [x1(1,1), x1(2,1)], 1);
%             plot(p);
       end
       
       function offset = generateOffsets(obj)
           x1 = obj.LineEquations.x;
           y1 = obj.LineEquations.y;
           Z = obj.Matrix;
            offset = [];
            offset.lineXPosition = [];
            p1 = [0 0]; %variables for line equations
            p2 = [0 0];
            %sweeping through all the lines we drew
            for m = 1:size(x1,2)
                
                p = polyfit([y1(1,m), y1(2,m)], [x1(1,m), x1(2,m)], 1);
                %get range of Y values.. so 0:45 for first group, sort our for 2nd
                y = min(y1(1,m),y1(2,m))+1:max(y1(1,m),y1(2,m));

                x = polyval(p, y); %these are the x values going down one line (or up)
                
                %sweeping DOWN
                for n = 1:size(x,2)
                    % get most common gradient of one horizonal sweet
                    grad = gradient(Z(n,:));
                    a = mode(gradient(Z(n,:)),2);
                    % index = find(gradient(Z(n,:)) == a);
                    % figure,plot(index);

                    %find a point left of the line AND witht the same gradient
                    %basically check points are consecutive
                    %basically sweeping left along x axis, starting at line
                    %pts is first consectutive line, pts2 is 2nd line
                    pts = [];
                    pts2 = [];
%                     find(index == round(x(n)));
                    sizeLine = obj.OffsetLineThreshold;
                    count = 0;

                    %ASSUME THAT LINE STARTS TO THE RIGHT OF ACTUAL PEAK
                    %do a check for a uphill slope with a margin of sizeLine
                    %should result in i being the start of a peak going
                    %downhill
                    for i = max(round(-x(n)-obj.OffsetPeakAdjust), -size(Z,2)):-1 
                        if (grad(-i) >= 0)
                            count = count + 1;
                        elseif (count < sizeLine && count > 0)
                            %size of 'hill' not big enough, reset
                            count = 0; 
                        end
                        %end after n pts
                        if (count == sizeLine)
                            break
                        end
                    end

                    %move cursor back
            %         i = i + sizeLine;
                    %now sweep right of the uphill peak
                    for j = -i:size(Z,2)
                        if (grad(j) == a)
                            pts = [pts j];
                        elseif (size(pts, 2) < sizeLine & size(pts, 2) > 0)
                            pts = [];
                        end
                        %end after n pts
                        if (size(pts,2) == sizeLine)
                            break
                        end
                    end

                    sizeLine = 5;
                    % get points for 2nd line segment
                    %sweep left of the uphill peak
                    for j = i:-1
                        if (grad(-j) == a)
                            pts2 = [pts2 -j];
                        elseif (size(pts2, 2) < sizeLine & size(pts2, 2) > 0)
                            pts2 = [];
                        end
                        %end after 10 pts
                        if (size(pts2,2) == sizeLine)
                            break
                        end
                    end

                       %% some error checking?
                    if (isempty(pts))
                        %means that the peak was far right -> just set p1 as a dummy
            %             p1 = [0 1];
                    else
                        p1 = polyfit([pts(1), pts(size(pts,2))], [Z(n,pts(1)), Z(n,pts(size(pts,2)))], 1);
                    end

                    if (isempty(pts2)) 
                        %peak too low? too far left
            %             p2 = [0 1]; % no difference if i comment/uncomment this out
                    else
                        p2 = polyfit([pts2(1), pts2(size(pts2,2))], [Z(n,pts2(1)), Z(n,pts2(size(pts2,2)))], 1);
                    end

                    offset.difference(n,m) = p1(2) - p2(2);
                    offset.p1(n,m) = p1(2);
                    offset.p2(n,m) = p2(2);
                    offset.peakPosition(n,m) = i;
                end

                %get x position of line from bruteforce2. Just want to compare
                %with actual peak found
                if size(x,2) <= size(Z,1)    %prob change, dont use Z
                    xfix = [x zeros(1,size(Z,1) - size(x,2))];
                    offset.lineXPosition = [offset.lineXPosition xfix'];
                else
                            error([mfilename ':WrongInputSize'], ...
                        'Inputs should be all row vectors or all column vectors.') ;
                end
            end
       end
       
       function plotOffsets(obj)
            offset = obj.Offset;

            figure, 
            plot(offset.difference);
            title('difference between left of line and right of line thing');
            ylabel('difference');
            xlabel('iteration going down the slope (n)');

            %%
            offset2 = offset.difference;
            offset2(find(offset.difference == 0)) = NaN;
            m_x = mean(offset2, 1, 'omitnan');
            t = 0:size(obj.Matrix, 1);
            hold on, plot(t,ones(length(t),1)*m_x)

            %doesnt do anything
            % a = filloutliers(offset2, 'previous', 'mean');
            % figure, plot(a);
            % ylim([-0.4 -0.2])
            % mean(a, 1, 'omitnan')
            a = filloutliers(offset.difference, 'previous', 'mean');
            a(find(offset.difference == 0)) = NaN;
            m_x = mean(a, 1, 'omitnan');
            t = 0:size(obj.Matrix, 1);

            figure, subplot 221, plot(a(:,1));
            hold on, plot(t,ones(length(t),1)*m_x(1))
            subplot 222, plot(a(:,2));
            hold on, plot(t,ones(length(t),1)*m_x(2))
            subplot 223, plot(a(:,3));
            hold on, plot(t,ones(length(t),1)*m_x(3))
            subplot 224, plot(a(:,4));
            hold on, plot(t,ones(length(t),1)*m_x(4))
            title('previous plot separated')

%             figure, subplot 221, plot(offset.difference(:,5));
%             subplot 222, plot(offset.difference(:,6));
%             subplot 223, plot(offset.difference(:,7));
%             subplot 224, plot(offset.difference(:,8));
%             title('previous plot separated')

            mean(offset2, 1, 'omitnan')
            %%
            figure,
%             imagesc(obj.MatrixPeaks); 
            hold on,
            plot(offset.lineXPosition(:,1:4)) %from bruteforce
            plot(-offset.peakPosition(:,1:4)) %extracted
            title('accuracy of brute force, extracted vs actual thingy\n (location of found peaks)');
            % c = corrcoef(a(:,1))
            % figure, imagesc(c)
            view([90 90])
            
       end
       function obj = runAll(obj)
           %obj.File = 'meas.mat';
           obj.DataSet = 'D_I';
           tic
%            obj.Matrix = generateMatrix(obj);
           obj.MatrixPeaks = generateMatrixPeaks(obj);
           toc, tic
           obj.LineEquations =  generateLines(obj); %generate lines of D_I
           toc
           obj.DataSet = 'FG_ST';
           obj.Offset = generateOffsets(obj);
       end
           
       
           
       
   end
end

%---------------------------------------------------------------------
function result = isIntersecting(line1, line2)
    slope = @(line) (line(2,2) - line(1,2))/(line(2,1) - line(1,1));
    intercept = @(line,m) line(1,2) - m*line(1,1);
    lineEq = @(m,b, myline) m * myline(:,1) + b;
    enderr = @(ends,line) ends - line(:,2);

    m1 = slope(line1);
    m2 = slope(line2);
    b1 = intercept(line1,m1);
    b2 = intercept(line2,m2);

    yEst2 = lineEq(m1, b1, line2);
%     plot(line2(:,1),yEst2,'om')

    errs1 = enderr(yEst2,line2);

    yEst1 = lineEq(m2, b2, line1);
%     plot(line1(:,1),yEst1,'or')
    errs2 = enderr(yEst1,line1);
    % check for actual intersection -> see if condition holds for both
    result =  sum(sign(errs1))==0 && sum(sign(errs2))==0 ;
end
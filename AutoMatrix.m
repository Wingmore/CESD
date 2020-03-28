function h = AutoMatrix(x, y, data)
%AUTOMATRIX Summary of this function goes here
%   Detailed explanation goes here
%   Credit to Henry Yang @ UNSW

dx = diff(x);
dy = diff(y);
xrez = min(abs(dx((abs(dx) > 1E-10))));
yrez = min(abs(dy((abs(dy) > 1E-10))));

xmin = round(min(x)/xrez)*xrez;
ymin = round(min(y)/yrez)*yrez;

xmax = round(max(x)/xrez)*xrez;
ymax = round(max(y)/yrez)*yrez;

xlen = round((xmax - xmin)/xrez) +1;
ylen = round((ymax - ymin)/yrez) +1;

% m = nan(ylen, xlen);
m = zeros(ylen, xlen);
zlen = length(data);
for i = 1:zlen
    if isempty(yrez)
        m(1, round((x(i)-xmin)/xrez)+1) = data(i);
    else
        m(round((y(i)-ymin)/yrez)+1, round((x(i)-xmin)/xrez)+1) = data(i);
    end
end


h.m = m;
h.xrez = xrez;
h.xmin = xmin;
h.xmax = xmax;
h.yrez = yrez;
h.ymin = ymin;
h.ymax = ymax;

end


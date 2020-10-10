open A/mmse_2_a_10.fig %open your fig file, data is the name I gave to my file
D=get(gca,'Children'); %get the handle of the line object
XData=get(D,'XData'); %get the x data
YData=get(D,'YData'); %get the y data

optimal = YData(3)

%Data=[XData' YData']; %join the x and y data on one array nx2
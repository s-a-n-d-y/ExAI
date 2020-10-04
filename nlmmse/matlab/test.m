I = imread('image.png');
corners = detectSURFFeatures(rgb2gray(I));
% corners = detectHarrisFeatures(rgb2gray(I));
J = insertMarker(I,corners,'circle');
imshow(J)
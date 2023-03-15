%% load movie
avipath = "Y:\Jasmine\checkerboard_6feb2023\cam2.avi";

v = VideoReader(avipath);
nframes = v.NumFrames;
allFrames = read(v);

%% extract 1 frame for each position
top = allFrames(:,:,1,1);
right = allFrames(:,:,1,401);
bottom = allFrames(:,:,1,531);
left = allFrames(:,:,1,831);

%% save each frame for future ref
imwrite(top, 'Y:\Jasmine\checkerboard_6feb2023\top.png');
imwrite(right, 'Y:\Jasmine\checkerboard_6feb2023\right.png');
imwrite(bottom, 'Y:\Jasmine\checkerboard_6feb2023\bottom.png');
imwrite(left, 'Y:\Jasmine\checkerboard_6feb2023\left.png');

%% sum them

fourframes = double(cat(3,top,left,right,bottom));
y = [346 711;5 289; 702 1017; 345 675];
x = [278 45; 727 382; 724 412; 985 777];

mergy = NaN(size(top,1),size(top,2),4);
for i = 1:4
    mergy(x(i,2):x(i,1),y(i,1):y(i,2),i) = fourframes(x(i,2):x(i,1),y(i,1):y(i,2),i);
    %mergy(:,:,i) = mergy(:,:,i)./nanmax(nanmax(mergy(:,:,i)));
end

allcheckers = mat2gray(nansum(mergy,3));
figure(2); clf; imshow(allcheckers)

imwrite(allcheckers, 'Y:\Jasmine\checkerboard_6feb2023\allcheckers.png');

%% test matlab fisheye correction
I = imread('Y:\Jasmine\checkerboard_6feb2023\allcheckers.png');
[imagePoints,boardSize] = detectCheckerboardPoints(I, 'HighDistortion', true);
squareSize = 20; % millimeters
worldPoints = generateCheckerboardPoints(boardSize,squareSize);
imageSize = [size(allcheckers,1) size(allcheckers,2)];
params = estimateFisheyeParameters(imagePoints,worldPoints,imageSize);

J2 = undistortFisheyeImage(allcheckers,params.Intrinsics,'OutputView','same', 'ScaleFactor', 0.2);
figure
imshow(J2)
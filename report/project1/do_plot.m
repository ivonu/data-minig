%% get rid of some stuff

clear all;
close all;

%% parameters
s = [0:0.001:1];

y1 = 1 - (1 - s.^16).^16;
y2 = 1 - (1 - s.^8).^32;

%% plots

plot(s,y1); 
hold on;
plot(s,y2);
hold on;


line([0.85,0.85],[1,0]);


clear all;
close all;

raw = importfile1('../data/training/training.txt',1,1000);

f = figure;
bar(var(raw(:,2:end)));
title('feature variance');
set(f, 'Position', [100 100 700 500]);
saveas(f,'feature-variance','epsc2');


f = figure;
bar(mean(raw(:,2:end)));
title('feature mean');
set(f, 'Position', [100 100 700 500]);
saveas(f,'feature-mean','epsc2');



y = raw(:,1);

x0 = ones(size(x1))

x1 = raw(:,2);
x2 = raw(:,3);

scatter(x1,x2,5,y);
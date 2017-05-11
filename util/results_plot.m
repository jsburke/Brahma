function results_plot( file )

%read data
data = csvread(file, 0, 1);
x    = data(1, 1:end-1);
res  = data(2:end, 1:end-1);
[opts extra] = size(res);

%get method names
fid  = fopen(file);
i    = 0;
methods = {};

while ~feof(fid)
   i = i + 1;
   method =  strsplit(fgetl(fid),',');
   if i > 1
        methods{i-1} = method{1};
   end
end

% set up plot
figure;
colors  = ['k','r','m','b','g'];   % colors for plots

for i = 1:opts
    plot(x, res(i,:), 'Color', colors(i));
    hold on;
end
hold off;
legend(methods);
end


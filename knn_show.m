% generate anchor points as non-balance way
% reset rand seed
rng('default');
sz = 40;

x = 0.5 * rand(1,100);
y = 0.5 * rand(1,100);
X = x;
Y = y;
% scatter(x,y);
% hold on
x = 1 - 0.5 * rand(1,5);
y = 1 - 0.5 * rand(1,5);
X = [X,x];
Y = [Y,y];
% scatter(x,y);

% hold on
x = 0.5 * rand(1,10);
y = 1 - 0.5 * rand(1,10);
X = [X,x];
Y = [Y,y];
% scatter(x,y);

% hold on
x = 1 - 0.5 * rand(1,10);
y = 0.5 * rand(1,10);
X = [X,x];
Y = [Y,y];

scatter(X,Y,sz,'MarkerEdgeColor','b',...
              'MarkerFaceColor','b',...
              'LineWidth',1.5);
set(gca,'YTick',[]);
set(gca,'XTick',[]);
% set(gca, 'box', 'off')
% hold on;
% scatter(0.1,0.25,sz,'MarkerEdgeColor','r',...
%               'MarkerFaceColor','r',...
%               'LineWidth',1.5);

hold on;
scatter(0.6,0.7,sz,'MarkerEdgeColor','r',...
              'MarkerFaceColor','r',...
              'LineWidth',1.5);
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'MySavedFile','-dpdf')

%%
fig = load('fig/magic04.fig');
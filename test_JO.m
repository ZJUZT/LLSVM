% load data
training.train_X = train_X;
training.train_Y = train_Y;

validation.test_X = test_X;
validation.test_Y = test_Y;

% pack paras
% pars.task = 'binary-classification';
% pars.task = 'multi-classification';
pars.iter_num = 1;
pars.epoch = 10;

% initial model
[~, p] = size(train_X);
class_num = max(train_Y);

%% svm
rng('default');
pars.skip = 1e1;
pars.w0 = 0;
pars.W = 0.1*randn(1,p);

pars.learning_rate = 1e2;
pars.t0 = 1e5;

disp('Training SVM...')
[model_svm, metric_svm] = svm(training, validation, pars);

%% llsvm
rng('default');
pars.skip = 1e1;

pars.learning_rate = 1e4 ;
pars.t0 = 1e5;

pars.beta = 1;
pars.anchors_num = 50; 
pars.nearest_neighbor = 5;

pars.w0 = zeros(1, pars.anchors_num);
pars.W = 0.1*randn(p, pars.anchors_num);

disp('Training LLSVM...')
[model_llsvm, metric_llsvm] = llsvm(training, validation, pars);

%% llc-sapl
rng('default');
pars.skip = 1e1;

pars.learning_rate = 1e5;
pars.learning_rate_anchor = 1e3;
pars.t0 = 1e5;

pars.beta = 1;
pars.anchors_num = 50; 
pars.nearest_neighbor = 5;

pars.w0 = zeros(1, pars.anchors_num);
pars.W = 0.1*randn(p, pars.anchors_num);

disp('Training LLC-SAPL...')
[model_llc_sapl, metric_llc_sapl] = llc_sapl(training, validation, pars);

%% llc-jo
rng('default');
pars.skip = 1e1;

pars.learning_rate = 1e5;
pars.learning_rate_anchor = 1e4;
pars.t0 = 1e5;

pars.LC = 5;
pars.anchors_num = 40; 
% pars.nearest_neighbor = 10;

pars.w0 = zeros(1, pars.anchors_num);
pars.W = 0.1*randn(p, pars.anchors_num);

disp('Training LLC-JO...')
[model_llc_jo, metric_llc_jo] = llc_jo(training, validation, pars);

%% llc-djo
rng('default');
pars.skip = 1e1;

pars.learning_rate = 1e4;
pars.learning_rate_anchor = 1e3;
pars.t0 = 1e5;

pars.epsilon = 1e-1;
pars.skip1 = 1e1;
pars.LC = 40;
pars.anchors_num = 100;
% pars.nearest_neighbor = 10;

pars.w0 = zeros(1, pars.anchors_num);
pars.W = 0.1*randn(p, pars.anchors_num);

disp('Training LLC-DJO...')
[model_llc_djo, metric_llc_djo] = llc_djo(training, validation, pars);

%% plot
% SVM
plot(metric_svm.loss_test(1,:),'g--o','DisplayName','SVM');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('hinge loss');
grid on;
hold on;

%% LLSVM
plot(metric_llsvm.loss_test(1,:),'b--o','DisplayName','LLSVM');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('hinge loss');
grid on;
hold on;

%% LLC_SAPL
plot(metric_llc_sapl.loss_test(1,:),'m--o','DisplayName','LLC-SAPL');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('hinge loss');
grid on;
hold on;

%% LLC_JO
plot(metric_llc_jo.loss_test(1,:),'c--o','DisplayName','LLC-JO');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('hinge loss');
grid on;
hold on;

%% LLC_DJO
plot(metric_llc_djo.loss_test(1,:),'r--o','DisplayName','LLC-DJO');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('hinge loss');
grid on;
hold on;

%% plot nn_avg
plot(metric_llc_jo.nn_avg_test(1,:),'r--o','DisplayName','LLC-DJO');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('average\_nn');
grid on;
hold on;

%%
metric_llc_jo.loss_train = 0.4122;
metric_llc_jo.loss_test = [0.5211, 0.4585, 0.4446, 0.4417, 0.4326, 0.4216, 0.4199,0.4190,0.4192,0.4190];
metric_llc_jo.loss_accuracy = 0.8175;
%% plot anchor_num (JO vs DJO)
an_jo = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
loss_jo = [0.1318,0.1197,0.1125,0.1072,0.1053,0.1040,0.1034,0.1030,0.1028,0.1026 ];

loss_djo = [0.1327,0.1134,0.1045,0.0997,0.0980,0.0970,0.0968,0.0965,0.0963 0.0965];
plot(an_jo, loss_jo,'c--o','DisplayName','LLC-JO');
hold on
plot(an_jo, loss_djo,'r--o','DisplayName','LLC-DJO');
legend('-DynamicLegend');
xlabel('anchor points number');

%% plot lip noise

%%
x = [0.5,1,2,5,10,20,30,40];
y = [0.3892,0.3633,0.3532,0.3365,0.3277,0.3411,0.3761,0.3905];
z = [57.8016, 38.3879, 24.5396, 12.3528, 7.2692, 3.2388, 1.8170, 1.2334];
a = [0,50];
b = [0.3923,0.3923];
plot(a,b,'k--','DisplayName', 'LLC-SAPL baseline');
legend('-DynamicLegend');
hold on
plot(x,y,'r--o','DisplayName','LLC-DJO');
legend('-DynamicLegend');
hold on;
grid on;
ylabel('hinge loss');
yyaxis right
plot(x,z,'b--o','DisplayName','average\_nn');
legend('-DynamicLegend');
xlabel('\mu');
ylabel('average nn number');
hold on;
%% stats
clear stats;
%-----------mean
%svm
stats.svm_mean_train_loss = mean(metric_svm.loss_train(:,end));
stats.svm_mean_test_loss = mean(metric_svm.loss_test(:,end));
stats.svm_mean_accuracy = mean(metric_svm.loss_accuracy(:,end));

stats.svm_std_train_loss = std(metric_svm.loss_train(:,end));
stats.svm_std_test_loss = std(metric_svm.loss_test(:,end));
stats.svm_std_accuracy = std(metric_svm.loss_accuracy(:,end));

%llsvm
stats.llsvm_mean_train_loss = mean(metric_llsvm.loss_train(:,end));
stats.llsvm_mean_test_loss = mean(metric_llsvm.loss_test(:,end));
stats.llsvm_mean_accuracy = mean(metric_llsvm.loss_accuracy(:,end));

stats.llsvm_std_train_loss = std(metric_llsvm.loss_train(:,end));
stats.llsvm_std_test_loss = std(metric_llsvm.loss_test(:,end));
stats.llsvm_std_accuracy = std(metric_llsvm.loss_accuracy(:,end));

%llc-sapl
stats.llc_sapl_mean_train_loss = mean(metric_llc_sapl.loss_train(:,end));
stats.llc_sapl_mean_test_loss = mean(metric_llc_sapl.loss_test(:,end));
stats.llc_sapl_mean_accuracy = mean(metric_llc_sapl.loss_accuracy(:,end));

stats.llc_sapl_std_train_loss = std(metric_llc_sapl.loss_train(:,end));
stats.llc_saplstd_test_loss = std(metric_llc_sapl.loss_test(:,end));
stats.llc_sapl_std_accuracy = std(metric_llc_sapl.loss_accuracy(:,end));

%llc-jo
stats.llc_jo_mean_train_loss = mean(metric_llc_jo.loss_train(:,end));
stats.llc_jo_mean_test_loss = mean(metric_llc_jo.loss_test(:,end));
stats.llc_jo_mean_accuracy = mean(metric_llc_jo.loss_accuracy(:,end));

stats.llc_jo_std_train_loss = std(metric_llc_jo.loss_train(:,end));
stats.llc_jo_std_test_loss = std(metric_llc_jo.loss_test(:,end));
stats.llc_jo_std_accuracy = std(metric_llc_jo.loss_accuracy(:,end));

%llc-djo
stats.llc_djo_mean_train_loss = mean(metric_llc_djo.loss_train(:,end));
stats.llc_djo_mean_test_loss = mean(metric_llc_djo.loss_test(:,end));
stats.llc_djo_mean_accuracy = mean(metric_llc_djo.loss_accuracy(:,end));

stats.llc_djo_std_train_loss = std(metric_llc_djo.loss_train(:,end));
stats.llc_djo_std_test_loss = std(metric_llc_djo.loss_test(:,end));
stats.llc_djo_std_accuracy = std(metric_llc_djo.loss_accuracy(:,end));
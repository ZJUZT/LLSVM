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
pars.minibatch = 10;

% initial model
[~, p] = size(train_X);
class_num = max(train_Y);

%% svm
rng('default');
pars.skip = 1e2;
pars.w0 = 0;
pars.W = 0.1*randn(1,p);

pars.learning_rate = 1e2;
pars.t0 = 1e5;

disp('Training SVM...')
[model_svm, metric_svm] = svm(training, validation, pars);

%% llsvm
rng('default');
pars.skip = 1e2;

pars.learning_rate = 1e3;
pars.t0 = 1e5;

pars.beta = 1;
pars.anchors_num = 50; 
pars.nearest_neighbor = 10;

pars.w0 = zeros(1, pars.anchors_num);
pars.W = 0.1*randn(p, pars.anchors_num);

disp('Training LLSVM...')
[model_llsvm, metric_llsvm] = llsvm(training, validation, pars);

%% llc-sapl
rng('default');
pars.skip = 1e2;

pars.learning_rate = 1e3;
pars.learning_rate_anchor = 1e1;
pars.t0 = 1e5;

pars.beta = 1;
pars.anchors_num = 50; 
pars.nearest_neighbor = 10;

pars.w0 = zeros(1, pars.anchors_num);
pars.W = 0.1*randn(p, pars.anchors_num);

disp('Training LLC-SAPL...')
[model_llc_sapl, metric_llc_sapl] = llc_sapl(training, validation, pars);

%% llc-jo
rng('default');
pars.skip = 1e2;

pars.learning_rate = 1e3;
pars.learning_rate_anchor = 1e1;
pars.t0 = 1e5;

pars.LC = 10;
pars.anchors_num = 50; 
% pars.nearest_neighbor = 10;

pars.w0 = zeros(1, pars.anchors_num);
pars.W = 0.1*randn(p, pars.anchors_num);

disp('Training LLC-JO...')
[model_llc_jo, metric_llc_jo] = llc_jo(training, validation, pars);

%% llc-djo
rng('default');
pars.skip = 1e1;

pars.learning_rate = 1e3;
pars.learning_rate_anchor = 1e1;
pars.t0 = 1e5;

pars.epsilon = 1e-1;
pars.skip1 = 1e2;
pars.LC = 10;
pars.anchors_num = 50; 
% pars.nearest_neighbor = 10;

pars.w0 = zeros(1, pars.anchors_num);
pars.W = 0.1*randn(p, pars.anchors_num);

disp('Training LLC-DJO...')
[model_llc_djo, metric_llc_djo] = llc_djo(training, validation, pars);
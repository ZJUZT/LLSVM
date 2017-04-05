% IJCNN dataset
ijcnn_train = 'data/ijcnn/train_data';
ijcnn_test = 'data/ijcnn/test_data';

% banana dataset
banana_train = 'data/banana/train_data';
banana_test = 'data/banana/test_data';

% magic04 dataset
magic04_train = 'data/magic04/train_data';
magic04_test = 'data/magic04/test_data';

training_data = ijcnn_train;
test_data = ijcnn_test;

% training_data = banana_train;
% test_data = banana_test;

% training_data = magic04_train;
% test_data = magic04_test;


load(training_data);
load(test_data);
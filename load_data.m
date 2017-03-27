% IJCNN dataset
ijcnn_train = 'data/ijcnn/train_data';
ijcnn_test = 'data/ijcnn/test_data';

% banana dataset
banana_train = 'data/banana/train_data';
banana_test = 'data/banana/test_data';

% training_data = ijcnn_train;
% test_data = ijcnn_test;

training_data = banana_train;
test_data = banana_test;


load(training_data);
load(test_data);
% IJCNN dataset
ijcnn_train = 'data/ijcnn/train_data';
ijcnn_test = 'data/ijcnn/test_data';

% banana dataset
banana_train = 'data/banana/train_data';
banana_test = 'data/banana/test_data';

% magic04 dataset
magic04_train = 'data/magic04/train_data';
magic04_test = 'data/magic04/test_data';

% USPS dataset
usps_train = 'data/usps/train_data';
usps_test = 'data/usps/test_data';

% LETTER dataset
letter_train = 'data/letter/train_data';
letter_test = 'data/letter/test_data';

% RCV1 dataset
% rcv1_train = 'data/rcv1/train_data';
% rcv1_test = 'data/rcv1/test_data';

% w8a dataset
% w8a_train = 'data/w8a/train_data';
% w8a_test = 'data/w8a/test_data';

% training_data = ijcnn_train;
% test_data = ijcnn_test;

% training_data = banana_train;
% test_data = banana_test;

% training_data = magic04_train;
% test_data = magic04_test;

% training_data = rcv1_train;
% test_data = rcv1_test;

% training_data = w8a_train;
% test_data = w8a_test;

% training_data = usps_train;
% test_data = usps_test;

training_data = letter_train;
test_data = letter_test;

load(training_data);
load(test_data);
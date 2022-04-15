clear; % Clear all the variables from the workspace.
close all; % Close all figure windows.
rng(1); % For reproducible results

% Read all the data from csv files
% X_train = readtable('X_train.csv');
% y_train = readtable('y_train.csv');
% X_test = readtable('X_test.csv');
% y_test = readtable('y_test.csv');

%Read the entire data
X = readtable('XRF.csv');
y = readtable('yRF.csv');

%Define train/test split ratios
holdouts = [0.5, 0.3, 0.1];


%Partition the dataset
cv = cvpartition(size(X,1),'HoldOut', holdouts(3)); % Using only the 10% holdout because it's the best performer (because of more training data).
% To find the best train/test split ratio, we ran a for loop to test which one yielded the best results.
                                                    
idx = cv.test;

X_train = X(~idx,:);
y_train = y(~idx,:);
X_test = X(idx, :);
y_test = y(idx,:);

colNames = categorical(X_train.Properties.VariableNames); % Get variable names



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Optimizing Hyperparameters%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Hyperparameters
%The parameters that we will experiment with are: 
% 1. Number of trees 
% 2. Minimum Leaf size
% 3. Number of predictors to sample at each node

%Specify ranges of hyperparameters to tune.
num_predictors_range = [1, size(X_test,2)-1]; %the number of predictors to sample at each node
tree_num_range = [10, 500];
min_leaf_size_range = [1, 20];
%max_num_splits = [1, 2, 3, 4, 5]; %Could tune also.

tree_num = optimizableVariable('tree_num', tree_num_range, 'Type','integer');
min_leaf_size = optimizableVariable('min_leaf_size', min_leaf_size_range, 'Type','integer');
num_predictors = optimizableVariable('num_predictors', num_predictors_range, 'Type','integer');
hyperparametersRF = [tree_num; min_leaf_size; num_predictors];

startRF=datetime('now'); % Time for starting training

results = bayesopt(@(params)oobErrRF(params, X_train, y_train), hyperparametersRF, ...
    'AcquisitionFunctionName','expected-improvement-plus', 'Verbose', 1, 'PlotFcn','all');

bestHyperparameters = results.XAtMinObjective; % Best parameters at the minimum of objective function.
bestOOBErr = results.MinObjective;

%Run model
Mdl = TreeBagger(bestHyperparameters.tree_num, X_train, y_train,'OOBPrediction','On', 'Method','classification', 'OOBPredictorImportance','on', 'MinLeafSize', bestHyperparameters.min_leaf_size, 'NumPredictorstoSample', bestHyperparameters.num_predictors)

endRF = datetime('now'); % Time after training
training_time = endRF - startRF;
display(training_time)

yHat = oobPredict(Mdl);
oobErrorBaggedEnsemble = oobError(Mdl);
figure('Name','Outofbag error'); % Open a new window(figure)
plot(oobErrorBaggedEnsemble, '-g');
xlabel('Number of trees')
ylabel('OOB error')

% Plot feature importance
figure('Name', 'Feature Importances')
color = [0.13 0.54 0.13]; %color
imp = Mdl.OOBPermutedPredictorDeltaError;
b = bar(colNames,imp,'FaceColor',color);
b.FaceColor = 'flat';
ylabel('RF Predictor importance estimates');
xlabel('Predictors');
grid on

% Make prediction
RFpredictedY_test = predict(Mdl, X_test);
RFpredictedY_test = str2double(RFpredictedY_test);

%Plot confusion matrix
figure();
y_test = table2array(y_test); % Change the type of y_train to array
confmatrix = confusionmat(y_test, RFpredictedY_test); %Estimate the confusion matrix
cm = confusionchart(confmatrix,  {'malignant', 'benign'});

% Confusion matrix specifications for the plotting 
cm.Title = 'Confusion matrix RF';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%ROC Curve
[FPR, TPR, T, AUC] = perfcurve(y_test, RFpredictedY_test, 1)
%plot(FPR, TPR)
plot(FPR, TPR, 'DisplayName', num2str(holdouts(3)))
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Random Forests')
legend(num2str(holdouts(3)))

%Precision Recall Metrics
%Precision = TP / TP + FP
%Recall = TP / TP + FN

% F1 score 
% Accuracy

%PR Estimation
prec = precision(confmatrix);
rec = recall(confmatrix);
F1 = (2.*prec.*rec)./(prec+rec)
accuracy = sum(diag(confmatrix))/sum(confmatrix, 'all')


% Error
misPredictions = sum(RFpredictedY_test ~= y_test);
allObservations = size(y_test);
error = misPredictions ./ allObservations


function y = precision(M)
  y = diag(M) ./ sum(M,2);
end

function y = recall(M)
  y = diag(M) ./ sum(M,1)';
end


%% Function for the objective
function oobErr = oobErrRF(params, X_train, y_train)
randomForest = TreeBagger(params.tree_num, X_train, y_train,'Method','classification', ...
    'OOBPrediction','on','MinLeafSize', params.min_leaf_size, 'NumPredictorstoSample', params.num_predictors)
oobErr = mean(oobError(randomForest));
end
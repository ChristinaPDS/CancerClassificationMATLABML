clear; % Clear all the variables from the workspace.
close all; % Close all figure windows.
rng(1); % For reproducible results

% Read all the data from csv files
% X_train = readtable('X_train.csv');
% y_train = readtable('y_train.csv');
% X_test = readtable('X_test.csv');
% y_test = readtable('y_test.csv');

%Read the entire data
X = readtable('X.csv');
y = readtable('y.csv');

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

%Count label occurencies for the prior 
y_table = table2array(y);
counts = histc(y_table(:), unique(y_table))
size_y = size(y) % Length of the data
prior = [counts(1,:)/size_y(1), counts(2,:)/size_y(1)]; %the prior class probability distribution is the relative frequency distribution of the classes in the data set

classNames= ['Malignant', 'Benign'];
classes = [0,1];

%Start measuring training time
startNB = datetime('now');

%%%% RUN NB MODELS %%%%%%%
% Mdl = fitcnb(X_train,y_train) % Simple NB
% Mdl = fitcnb(X_train, y_train, 'ClassNames', classes,'Prior', prior);
% Mdl = fitcnb(X_train,y_train, 'ClassNames',classes,'OptimizeHyperparameters','auto', 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%      'expected-improvement-plus'));
% Mdl = fitcnb(X_train,y_train, 'ClassNames',classes,'OptimizeHyperparameters','all', 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','Optimizer', 'bayesopt', 'ShowPlots', true));

Mdl = fitcnb(X,y,'ClassNames',classes,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus', 'ShowPlots', true))

endNB = datetime('now');
trainingTime = endNB - startNB;
display(trainingTime)

%Test model on the test set
yhat = Mdl.predict(X_test);

y_test = table2array(y_test); % Change the type of y_train to array
confmatrix = confusionmat(y_test, yhat);

cm = confusionchart(confmatrix,  {'malignant', 'benign'});

% Confusion matrix specifications for the plotting 
cm.Title = 'Confusion matrix Naive Bayes';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%ROC curve
[FPR, TPR, T, AUC] = perfcurve(y_test, yhat, 1)
figure();
plot(FPR, TPR, 'DisplayName', num2str(holdouts(3)))
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes')
legend(num2str(holdouts(3)))

%PR Estimation
prec = precision(confmatrix);
rec = recall(confmatrix);
F1 = (2.*prec.*rec)./(prec+rec)
accuracy = sum(diag(confmatrix))/sum(confmatrix, 'all')

% Error
misPredictions = sum(yhat ~= y_test);
allObservations = size(y_test);
error = misPredictions ./ allObservations


function y = precision(M)
  y = diag(M) ./ sum(M,2);
end

function y = recall(M)
  y = diag(M) ./ sum(M,1)';
end
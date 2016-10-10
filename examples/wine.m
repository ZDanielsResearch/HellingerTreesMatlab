%Clear existing data
clear all;
close all;
clc;

%Load cities dataset
load('wine_dataset.mat');

%Just for computing splits
labels = wineTargets(1,:)';
features = wineInputs';

%Split Dataset
split = crossvalind('LeaveMOut',length(labels),floor(0.2 .* length(labels)));

%Just for computing statistics
trainingLabels = labels(split == 1);
trainingFeatures = features(split == 1,:);
testLabels = labels(split == 0);
testFeatures = features(split == 0,:);

%Dataset statistics
disp('Dataset: Wine')
disp(['Number of Training Instances: ' num2str(size(trainingFeatures,1))]);
disp(['Number of Test Instances: ' num2str(size(testFeatures,1))]);
disp(['Number of Features (Measurements): ' num2str(size(trainingFeatures,2))]);
disp(' ');

%Convert to binary classification problem: 1 vs other 2
for i = 1:1:size(wineTargets,1)
    %Fix labels to be binary
    labels = wineTargets(i,:)';
    features = wineInputs';
    
    trainingLabels = labels(split == 1);
    trainingFeatures = features(split == 1,:);
    testLabels = labels(split == 0);
    testFeatures = features(split == 0,:);
    
    %Run classifiers
    
    %HellingerTree
    disp('Hellinger Tree:')
    tic();
    model = fit_Hellinger_tree(trainingFeatures,trainingLabels,[],5);
    trainingTime = toc();
    tic();
    [predictions,scores] = predict_Hellinger_tree(model,testFeatures);
    testTime = toc();
    correct = (predictions == testLabels);
    correct = sum(correct) / length(correct);
    disp(['Percent of instances correctly classified: ' num2str(correct)]);
    
    [precision,recall,f1] = get_statistics(testLabels,predictions);
    [~,~,~,AUC] = perfcurve(testLabels,scores,1);
    disp(['Precision: ' num2str(precision)]);
    disp(['Recall: ' num2str(recall)]);
    disp(['F-Measure: ' num2str(f1)]);
    disp(['AUROC: ' num2str(AUC)]);
    
    disp(['Training time: ' num2str(trainingTime) ' seconds']);
    disp(['Test time: ' num2str(testTime) ' seconds']);
    disp(['Total time: ' num2str(trainingTime + testTime) ' seconds']);
    disp(' ');
    
    %Hellinger Forest
    disp('Hellinger Forest:')
    tic();
    model = fit_Hellinger_forest(trainingFeatures,trainingLabels,10,[],[],5);
    trainingTime = toc();
    tic();
    [predictions,scores] = predict_Hellinger_forest(model,testFeatures);
    testTime = toc();
    correct = (predictions == testLabels);
    correct = sum(correct) / length(correct);
    disp(['Percent of instances correctly classified: ' num2str(correct)]);
    
    [precision,recall,f1] = get_statistics(testLabels,predictions);
    [~,~,~,AUC] = perfcurve(testLabels,scores,1);
    disp(['Precision: ' num2str(precision)]);
    disp(['Recall: ' num2str(recall)]);
    disp(['F-Measure: ' num2str(f1)]);
    disp(['AUROC: ' num2str(AUC)]);
    
    disp(['Training time: ' num2str(trainingTime) ' seconds']);
    disp(['Test time: ' num2str(testTime) ' seconds']);
    disp(['Total time: ' num2str(trainingTime + testTime) ' seconds']);
    disp(' ');
    
    %Linear SVM for Comparison
    disp('SVM:')
    tic();
    model = fitcsvm(trainingFeatures,trainingLabels);
    trainingTime = toc();
    tic();
    [predictions,scores] = predict(model,testFeatures);
    testTime = toc();
    correct = (predictions == testLabels);
    correct = sum(correct) / length(correct);
    disp(['Percent of instances correctly classified: ' num2str(correct)]);
    [precision,recall,f1] = get_statistics(testLabels,predictions);
    [~,~,~,AUC] = perfcurve(testLabels,scores(:,2),1);
    disp(['Precision: ' num2str(precision)]);
    disp(['Recall: ' num2str(recall)]);
    disp(['F-Measure: ' num2str(f1)]);
    disp(['AUROC: ' num2str(AUC)]);
    
    disp(['Training time: ' num2str(trainingTime) ' seconds']);
    disp(['Test time: ' num2str(testTime) ' seconds']);
    disp(['Total time: ' num2str(trainingTime + testTime) ' seconds']);
    disp(' ');
end
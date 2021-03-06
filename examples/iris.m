%Clear existing data
clear all;
close all;
clc;

%Load iris dataset
load('fisheriris.mat');

%Convert species strings to numeric labels
classes = {'setosa','versicolor','virginica'};
labels = zeros(size(species,1),1);
for i = 1:1:length(classes)
    indices = find(~cellfun('isempty',strfind(species,classes{i})));
    labels(indices) = i;
end

%Split Dataset
split = crossvalind('LeaveMOut',length(labels),floor(0.2 .* length(labels)));

%Convert to binary classification problem: 1 vs other 2 
for i = 1:1:length(classes)
    binaryLabels = labels == i;
    trainingLabels = binaryLabels(split == 1);
    trainingFeatures = meas(split == 1,:);
    testLabels = binaryLabels(split == 0);
    testFeatures = meas(split == 0,:);
    
    %Dataset statistics
    disp('Dataset: Iris (Flower)')
    disp(['Number of Training Instances: ' num2str(size(trainingFeatures,1))]);
    disp(['Number of Test Instances: ' num2str(size(testFeatures,1))]);
    disp(['Number of Features (Measurements): ' num2str(size(trainingFeatures,2))]);
    disp(' ');
    
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
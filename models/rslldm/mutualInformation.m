function [indMutInf] = mutualInformation(data, label, topN)
    data = spones(data);
    indMutInf = [];
    numdocs = size(data, 1);
    numfeatures = size(data, 2);
    ulabels = unique(label);
    numlabels = numel(ulabels);
    labelFeat = zeros(numlabels, numfeatures);
    labelAndFeatCooccr = zeros(numlabels, numfeatures);
    weights = zeros(numlabels, numfeatures);
    weighs = zeros(numlabels, numfeatures);
    for i=1:numlabels
        labelOcc = label == ulabels(i);
        for j=1:numfeatures
            featureOcc = data(:,j);
            n = numdocs;
            n11 = sum((featureOcc + labelOcc) == 2);
            n01 = sum((featureOcc - labelOcc) == -1);
            n00 = sum((featureOcc + labelOcc) == 0);
            n10 = sum((featureOcc - labelOcc) == 1);
            n1_ = sum(featureOcc == 1);
            n0_ = sum(featureOcc == 0);
            n_1 = sum(labelOcc == 1);
            n_0 = sum(labelOcc == 0);
            weights(i,j) = (n11/n)*log2((n*n11)/(n1_*n_1)) + (n01/n)*log2((n*n01)/(n0_*n_1)) + (n10/n)*log2((n*n10)/(n1_*n_0)) + (n00/n)*log2((n*n00)/(n0_*n_0));
            labelAndFeatCooccr(i,j) = n11;
        end
        weights(isnan(weights)) = -1;
        [ts, te] = sort(weights(i,:), 'descend');
        labelFeat(i,:) = te;
        weighs(i,:) = ts;
        indMutInf = [indMutInf, te(1:topN)];
        indMutInf = unique(indMutInf);
    end
end
function [weightEachClass, indMutInfEachClass, weights, indMutInf, indxImpf, indxImptf] = mutualInfo(data, label, topN, topM, topO)
    odata = data;
    data = spones(data);
    indMutInf = [];
	indMutInfEachClass = [];
    numdocs = size(data, 1);
    numfeatures = size(data, 2);
    ulabels = unique(label);
    numlabels = numel(ulabels);
    weights = zeros(1, numfeatures);
    weightEachClass = zeros(numlabels, numfeatures);
	
	N = numdocs;
	for i=1:numfeatures
		wgt = 0;
		featureOcc = data(:,i);
		for j=1:numlabels
			lb = ulabels(j);
			F1L1 = sum(((featureOcc == 1) + (label == lb)) == 2);
			F1 = sum(featureOcc == 1);
			L1 = sum(label == lb);
			temp1 = (F1L1/N)*log2((N*F1L1)/(F1*L1));
			weightEachClass(j,i) = temp1;
			F1L1 = sum(((featureOcc == 0) + (label == lb)) == 2);
			F1 = sum(featureOcc == 0);
			L1 = sum(label == lb);
			temp2 = (F1L1/N)*log2((N*F1L1)/(F1*L1));
			wgt = wgt + temp1 + temp2;
		end
		weights(1, i) = wgt;
	end
	weights(isnan(weights)) = -1;
	weightEachClass(isnan(weightEachClass)) = -1;
	[ts, te] = sort(weights, 'descend');
	indMutInf = te(1:topN);
	
	[ts, te] = sort(weightEachClass(1,:), 'descend');
	indMutInfEachClass = te(1:topN);
	for i=2:numlabels
		[ts, te] = sort(weightEachClass(i,:), 'descend');
		indMutInfEachClass = [indMutInfEachClass; te(1:topN)];
	end
	
	indxImptf = []; 
    	indxImpf = [];
    	dataO = sum(odata, 1);
    	[ts, te] = sort(dataO, 'descend');
    	indxImpf = te(topO+1:topM+topO);
end


function [ ] = lblDmAltMF_rpc( wFull, newDocBOW, maxOuter, savefile, impFeat)
%LBLDMALTMF Iteratively solve subproblems using minFunc
% takes initial parameter vector as argument
% assumes model parameters in global context
% handles saving results etc within this function

global modelParams;

% minfunc options when optimizing word representations (R,b)
options.Diagnostics = 'on';
options.Display = 'iter';
options.MaxIter = 10;
%options.MaxFunEvals = 100;
options.Corr = 20;
options.DerivativeCheck = 'off';

% load data into worker global state
% loading the first numDocs examples without randomly permuting
numLoaded=slaveLoadBOW((1:modelParams.NumDocs)', modelParams.BowFname, modelParams.DictSize, newDocBOW);
assert(numLoaded == modelParams.NumDocs);

% pass modelParams to worker global state
slaveSetNonopt(modelParams.toVector(), -1);

for t = 1 : maxOuter
	maxiterVal = t;
    curW = [wFull(modelParams.repConIndex()); ...
        wFull(modelParams.wordBiasIndex())];
    % optimize for doc thetas
    fprintf(1,'\noptimizing for doc thetas ...\n');
    % load word representations into worker non-opt data
    slaveSetNonopt( -1, curW );
    % run doc optimizer
    wFull(modelParams.thetaMatIndex()) = slaveOptDt(wFull(modelParams.thetaMatIndex()));
     
    % optimize for repConMat
    fprintf(1,'\noptimizing for word reps ...\n');  
    % pass slaves their doc thetas
    slaveSetNonopt( -1, wFull(modelParams.thetaMatIndex()));
    tic;
    [coVec, coF, coFlag, coInfo] = minFunc(@slaveLblDmRcErObj, curW, options);
    toc;
    % copy out 'optimal' params
    wFull(modelParams.repConIndex()) = coVec(modelParams.repConIndex());        
    wFull(modelParams.wordBiasIndex()) = coVec(modelParams.wordBiasIndex());
    
	wordRep = reshape(wFull(modelParams.repConIndex()),modelParams.DictSize,[]);
	docRep = reshape(wFull(modelParams.thetaMatIndex()),modelParams.NumDocs,[]);
	bias = reshape(wFull(modelParams.wordBiasIndex()),modelParams.DictSize,[]);
	
	save(savefile, 'wordRep', 'docRep', 'bias', 'maxiterVal', 'impFeat');
    fprintf(1,'LBLDM Iteration: %f\n', t);
end
end


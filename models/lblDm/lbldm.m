function [theta, phi, bias] = lblDm(phiPriorMean, thetaPriorMean, biasPriorMean, phiInit, thetaInit, biasInit, phiPrec, thetaPrec, biasPrec, options, data, maxiter, dataNumber, impFeat, wPrior, dPrior)
	savefile = strcat('modelsResult', num2str(dataNumber), '.mat');
	dataN = sum(data,2);

	numDocs = size(data, 1);
	numWords = size(data, 2);
	ndim = size(phiPriorMean, 2);
	run_lblDm(ndim, numWords, numDocs, data, maxiter, savefile, impFeat, wPrior, dPrior);
end

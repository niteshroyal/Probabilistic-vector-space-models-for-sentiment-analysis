function [ numLoaded ] = slaveLoadBOW( docInd, bowFname, vocabSize, newDocBOW)
	%SLAVELOADBOW loads BOW data insto slave state
	% docInd is the index into the full data matrix
	% loads data and stores it in global used by worker/slave

	% assumes word count matrix is stored in a var called data_docBOW
	% word count matrix is num documents by num vocab
	data_docBOW = newDocBOW;

	numDocs = size(data_docBOW,1);
	% count total number of words in doc avoiding 0s
	data_docLen = full(max(sum(data_docBOW,2), ones(numDocs,1)));
	% normalize BOW Distros
	data_docBOW = bsxfun(@rdivide, data_docBOW, data_docLen);
	
	% load data into global state
	global state;
	state.docBow = data_docBOW;
	state.docLen = data_docLen;
	numLoaded = size(state.docBow,1);
end
function [bias] = lldmBiasFunc(theta, phi, phiMean, thetaMean, biasMean, phiPrec, thetaPrec, biasPrec, bias0, options, data, dataN, numDoc, numWord, dim)
    phi = reshape(phi, [dim, numWord])';
    theta = reshape(theta, [dim, numDoc])';
    bias = minFunc(@crossEntropyFunc,bias0, options);
    %bias = fminunc(@crossEntropyFunc,bias0, options);
    function [fval, fgrad] = crossEntropyFunc(bias)
        p1 = 0.5*thetaPrec*sumsqr(theta - thetaMean);
        p2 = 0.5*phiPrec*sumsqr(phi - phiMean);
        p3 = 0.5*biasPrec*sumsqr(bias - biasMean);
        Y = bsxfun(@plus, theta*phi', bias');
        Y = exp(Y);
        Y = bsxfun(@rdivide, Y, sum(Y,2));
        logY = log(Y);
        t1 = logY.*data;
        p4 = sum(t1(:));
        fval = p1 + p2 + p3 - p4;
        
        fgrad = biasPrec*(bias - biasMean);
        for j=1:numWord
            q1 = 0;
            for d=1:numDoc
                q1 = q1 + (Y(d,j)*dataN(d) - data(d,j));
            end
            fgrad(j) = fgrad(j) + q1;
        end
    end
end
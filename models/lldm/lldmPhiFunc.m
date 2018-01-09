function [phi] = lldmPhiFunc(theta, bias, phiMean, thetaMean, biasMean, phiPrec, thetaPrec, biasPrec, phi0, options, data, dataN, numDoc, numWord, dim)
    theta = reshape(theta, [dim, numDoc])';
    phi = minFunc(@crossEntropyFunc,phi0, options);
    %phi = fminunc(@crossEntropyFunc,phi0, options);
    function [fval, fgrad] = crossEntropyFunc(phi)
        phi = reshape(phi, [dim, numWord])';
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
        
        fgrad = phiPrec*(phi - phiMean);
        for j=1:numWord
            q1 = zeros(1,dim);
            for d=1:numDoc
                q1 = q1 + (Y(d,j)*dataN(d) - data(d,j))*theta(d,:);
            end
            fgrad(j,:) = fgrad(j,:) + q1;
        end
        fgrad = reshape(fgrad', [dim*numWord, 1]);
    end
end
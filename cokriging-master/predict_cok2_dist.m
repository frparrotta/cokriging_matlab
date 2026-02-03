function [mu, sigma_gp, var_gp] = predict_cok2_dist(x0, dmodel)

% This function extends predict_cok2.m:
% it returns not only the mean prediction (mu),
% but also the predictive variance (var_gp) and std (sigma_gp) from the GP.

smean = dmodel.smean;
sstd  = dmodel.sstd;
ymean = dmodel.ymean;
ystd  = dmodel.ystd;

% --- Normalize inputs (same normalization used during training in cokriging2.m) ---
x = (x0 - repmat(smean,size(x0,1),1)) ./ repmat(sstd,size(x0,1),1);

% --- DACE predictor can return [y, mse] ---
% mse is the predictive variance in the normalized output space.
[yc_n, mse_c] = predictor(x, dmodel.dmc);   % low-fidelity GP
[yd_n, mse_d] = predictor(x, dmodel.dmd);   % discrepancy GP

% --- Combine mean (normalized) ---
mu_n = yd_n + yc_n * dmodel.p;

% --- Combine variance (normalized), assuming independence of the two GPs ---
var_n = (dmodel.p.^2) .* mse_c + mse_d;

% --- De-normalize to original output units ---
mu     = mu_n .* repmat(ystd,size(mu_n,1),1) + repmat(ymean,size(mu_n,1),1);
var_gp = var_n .* (repmat(ystd,size(var_n,1),1).^2);
sigma_gp = sqrt(var_gp);

end



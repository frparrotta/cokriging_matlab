clear; close all; clc;

% --- Path: serve per usare DACE (dacefit, predictor, ecc.) ---
addpath('dace');

%% --- Load LOW-FIDELITY dataset (simulations) ---
% File: func_D1.xlsx
% Expected columns (Excel headers):
%   - "Scale factor"  (unitless, e.g., 0.55)
%   - "Time"          (s)
%   - "D1"            (mm)
LF = readtable('func_D1.xlsx','VariableNamingRule','preserve');

% Build input matrix X_LF (N_LF x 2) and output vector y_LF (N_LF x 1)
X_LF = [LF.("Scale factor"), LF.("Time")];
y_LF = LF.("D1");

% Quick sanity check
fprintf('Loaded LF: %d samples | X_LF = %dx%d | y_LF = %dx%d\n', ...
    size(X_LF,1), size(X_LF,1), size(X_LF,2), size(y_LF,1), size(y_LF,2));

%% --- Load HIGH-FIDELITY dataset (experiments) ---
% File: simu_validation_calibration.xlsx
% We use the first 3 columns:
%   - "Scale Factor [%]"   (convert to fraction by /100)
%   - "Time [s]"
%   - "Exp Mean  D1"       (per-repetition D1 values; 6 reps per condition)
HF = readtable('simu_validation_calibration.xlsx','VariableNamingRule','preserve');

disp('HF columns:');
disp(HF.Properties.VariableNames);

% Extract inputs/outputs from HF table
ScaleHF = HF.("Scale Factor [%]")/100;   % IMPORTANT: % -> fraction to match LF
TimeHF  = HF.("Time [s]");
D1rep   = HF.("Exp Mean  D1");

% Build a table for grouping replicates by experimental condition (ScaleHF, TimeHF)
T = table(ScaleHF, TimeHF, D1rep);

% Aggregate replicates -> one HF point per condition (mean), plus SD across replicates
Gm = groupsummary(T, ["ScaleHF","TimeHF"], "mean", "D1rep");  % mean D1 per condition
Gs = groupsummary(T, ["ScaleHF","TimeHF"], "std",  "D1rep");  % std  D1 per condition

X_HF = [Gm.ScaleHF, Gm.TimeHF];     % (nCond x 2)
y_HF = Gm.mean_D1rep;               % (nCond x 1) -> HF target for cokriging

sd_cond  = Gs.std_D1rep;            % experimental SD per condition
var_cond = sd_cond.^2;

fprintf('Loaded HF: %d rows (replicates) -> %d unique conditions\n', height(HF), size(X_HF,1));
disp(table(X_HF(:,1), X_HF(:,2), y_HF, sd_cond, ...
    'VariableNames', {'ScaleFactor','Time','D1_mean','D1_sd'}));

% --- Pooled experimental variance (constant noise) ---
% Assumption: experimental variance is roughly similar across the 6 conditions.
% We estimate it as the average variance observed across conditions.
%
% NOTE: You currently multiply by 10 (sensitivity / conservative inflation).
% This will make sigma_Tot much larger and will reduce the ratio sigma_GP/sigma_Tot.
sigma2_pooled =  mean(var_cond, 'omitnan');
fprintf('sigma2_pooled (experimental) = %.6g  (units: D1^2)\n', sigma2_pooled);

%% --- Fit Co-Kriging model for D1 ---
% Bounds for theta (one per input dimension)
lb = 1e-6 * ones(1, size(X_LF,2));
ub = 1e2  * ones(1, size(X_LF,2));

% Train the co-kriging model
[dmodel_D1, dmc_D1, dmd_D1] = cokriging2(X_LF, y_LF, X_HF, y_HF, @regpoly0, @corrgauss, lb, ub);

% Quick check: prediction on the 6 HF conditions
[mu_hf, sigma_gp_hf, var_gp_hf] = predict_cok2_dist(X_HF, dmodel_D1);

% Total uncertainty = GP + pooled experimental variability
sigma_tot_hf = sqrt(var_gp_hf + sigma2_pooled);

rmse_hf = sqrt(mean((mu_hf - y_HF).^2));
fprintf('RMSE on HF-condition means (6 points) = %.6g\n', rmse_hf);

disp(table(X_HF(:,1), X_HF(:,2), y_HF, mu_hf, sigma_gp_hf, sigma_tot_hf, ...
    'VariableNames', {'ScaleFactor','Time','D1_HF_mean','D1_pred_mu','D1_pred_sigmaGP','D1_pred_sigmaTot'}));

%% --- Sanity check: uncertainty away from HF points (1D slice) ---
% We inspect a 1D slice: D1 vs ScaleFactor at a fixed Time.
% This plot is useful to visually see:
%   - mean prediction
%   - GP-only uncertainty
%   - total uncertainty (including experimental variability)
%   - HF anchors (blue)
%   - LF samples (gray) close to the selected time

time_slice = 10;  % choose 5 or 10 (HF times)
sf_min = min(X_LF(:,1));
sf_max = max(X_LF(:,1));
sf_grid = linspace(sf_min, sf_max, 200)';

Xq_1d = [sf_grid, time_slice*ones(size(sf_grid))];

% Predict mean + GP uncertainty
[mu_1d, sigma_gp_1d, var_gp_1d] = predict_cok2_dist(Xq_1d, dmodel_D1);

% Add pooled experimental variability
sigma_tot_1d = sqrt(var_gp_1d + sigma2_pooled);

% 95% bands
low_gp  = mu_1d - 1.96*sigma_gp_1d;
high_gp = mu_1d + 1.96*sigma_gp_1d;

low_tot  = mu_1d - 1.96*sigma_tot_1d;
high_tot = mu_1d + 1.96*sigma_tot_1d;

% HF points on this time slice
mask_hf = abs(X_HF(:,2) - time_slice) < 1e-12;
XHF_slice = X_HF(mask_hf,1);
yHF_slice = y_HF(mask_hf);

% LF samples close to this time slice (visual reference)
tol_time = 0.2;  % [s] tolerance for selecting LF points near time_slice
mask_lf = abs(X_LF(:,2) - time_slice) < tol_time;

figure; hold on; grid on;

% Total uncertainty band (includes experimental variability)
fill([sf_grid; flipud(sf_grid)], [low_tot; flipud(high_tot)], ...
     [0.8 0.8 0.8], 'EdgeColor','none');

% GP-only band (model uncertainty only)
fill([sf_grid; flipud(sf_grid)], [low_gp; flipud(high_gp)], ...
     [0.6 0.6 0.6], 'EdgeColor','none');

% Mean prediction
plot(sf_grid, mu_1d, 'k', 'LineWidth', 1.5);

% HF condition means
if any(mask_hf)
    scatter(XHF_slice, yHF_slice, 60, 'b', 'filled');
end

% LF samples near the time slice (semi-transparent)
if any(mask_lf)
    scatter(X_LF(mask_lf,1), y_LF(mask_lf), ...
        15, [0.5 0.5 0.5], 'filled', ...
        'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);
end

xlabel('Scale factor');
ylabel('D1 [mm]');
title(sprintf('Sanity check: D1 vs Scale factor at Time = %g s', time_slice));
legend('95% total (GP + exp pooled)', ...
       '95% GP-only', ...
       'Mean prediction', ...
       'HF means', ...
       'LF samples (near time)', ...
       'Location', 'best');

%% --- Prediction over the full input space (2D grid) + maps ---
% We compute predictions on a grid over the LF domain, then plot:
%   - MU   : mean prediction
%   - SGP  : GP-only uncertainty (model uncertainty)
%   - STOT : total uncertainty (model + experimental variability)
% plus we overlay:
%   - LF sample locations (black, very transparent)
%   - HF condition locations (white)

% Grid limits from LF domain
sf_min = min(X_LF(:,1));  sf_max = max(X_LF(:,1));
t_min  = min(X_LF(:,2));  t_max  = max(X_LF(:,2));

nSF = 80;
nT  = 80;

sf_grid = linspace(sf_min, sf_max, nSF);
t_grid  = linspace(t_min,  t_max,  nT);

[SF, TT] = meshgrid(sf_grid, t_grid);
Xq = [SF(:), TT(:)];

% Predict GP distribution on grid
[mu, sigma_gp, var_gp] = predict_cok2_dist(Xq, dmodel_D1);

% Total uncertainty
sigma_tot = sqrt(var_gp + sigma2_pooled);

% Reshape to grid
MU   = reshape(mu,        nT, nSF);
SGP  = reshape(sigma_gp,  nT, nSF);
STOT = reshape(sigma_tot, nT, nSF);

% LF overlay style (important to avoid clutter)
lf_marker_size = 6;
lf_color = 'k';
lf_alpha = 0.06;   % try 0.03 if still too dense

%% --- FIGURE: mean map (MU) ---
figure;
imagesc(sf_grid, t_grid, MU); axis xy; colorbar;
hold on;

% Overlay LF sample locations (all LF points) - very transparent
scatter(X_LF(:,1), X_LF(:,2), lf_marker_size, lf_color, ...
    'filled','MarkerFaceAlpha', lf_alpha, 'MarkerEdgeAlpha', lf_alpha);

% Overlay HF conditions
scatter(X_HF(:,1), X_HF(:,2), 60, 'w', 'filled');

xlabel('Scale factor'); ylabel('Time [s]');
title('CoKriging mean \mu(Scale factor, Time) for D1');

%% --- FIGURE: GP-only uncertainty map (SGP) ---
figure;
imagesc(sf_grid, t_grid, SGP); axis xy; colorbar;
hold on;

scatter(X_LF(:,1), X_LF(:,2), lf_marker_size, lf_color, ...
    'filled','MarkerFaceAlpha', lf_alpha, 'MarkerEdgeAlpha', lf_alpha);

scatter(X_HF(:,1), X_HF(:,2), 60, 'w', 'filled');

xlabel('Scale factor'); ylabel('Time [s]');
title('GP-only uncertainty \sigma_{GP}(Scale factor, Time) for D1');

%% --- FIGURE: total uncertainty map (STOT) ---
figure;
imagesc(sf_grid, t_grid, STOT); axis xy; colorbar;
hold on;

scatter(X_LF(:,1), X_LF(:,2), lf_marker_size, lf_color, ...
    'filled','MarkerFaceAlpha', lf_alpha, 'MarkerEdgeAlpha', lf_alpha);

scatter(X_HF(:,1), X_HF(:,2), 60, 'w', 'filled');

xlabel('Scale factor'); ylabel('Time [s]');
title('Total uncertainty \sigma_{Tot} (GP + pooled exp) for D1');

%% --- FIGURE: ratio map RATIO = sigma_GP / sigma_Tot ---
% This diagnostic tells where HF data would help:
%   - ratio ~ 0 : exp-dominated uncertainty (HF won't reduce total much)
%   - ratio ~ 1 : model-dominated uncertainty (HF will reduce total a lot)
RATIO = SGP ./ STOT;
RATIO(~isfinite(RATIO)) = 0;

figure;
imagesc(sf_grid, t_grid, RATIO); axis xy; colorbar;
caxis([0 1]);
hold on;

scatter(X_LF(:,1), X_LF(:,2), lf_marker_size, lf_color, ...
    'filled','MarkerFaceAlpha', lf_alpha, 'MarkerEdgeAlpha', lf_alpha);

scatter(X_HF(:,1), X_HF(:,2), 60, 'w', 'filled');

xlabel('Scale factor');
ylabel('Time [s]');
title('Where HF data helps: ratio \sigma_{GP} / \sigma_{Tot} (0 = exp-dominated, 1 = model-dominated)');

%% --- Quantify how much the model contributes to total uncertainty ---

% Ratio on std (intuitive)
r_sigma = SGP(:) ./ STOT(:);

% Ratio on variance (more rigorous)
r_var = (SGP(:).^2) ./ (STOT(:).^2);

% Summary on the whole grid
fprintf('\n=== Contribution of model uncertainty over the WHOLE grid ===\n');
fprintf('r_sigma = sigmaGP/sigmaTot: mean = %.2f, median = %.2f, max = %.2f\n', ...
    mean(r_sigma,'omitnan'), median(r_sigma,'omitnan'), max(r_sigma));
fprintf('r_var   = varGP/varTot:     mean = %.2f, median = %.2f, max = %.2f\n', ...
    mean(r_var,'omitnan'), median(r_var,'omitnan'), max(r_var));

% "Far from HF" region (exclude neighborhood around HF points)
dist_thr = 0.15;

% Normalize inputs using model normalization
Xq_norm  = (Xq - dmodel_D1.smean) ./ dmodel_D1.sstd;
XHF_norm = (X_HF - dmodel_D1.smean) ./ dmodel_D1.sstd;

% Distance of each grid point to nearest HF condition
D = pdist2(Xq_norm, XHF_norm);
dmin = min(D, [], 2);

mask_far = dmin > dist_thr;

fprintf('\n=== Contribution of model uncertainty FAR from HF points (dmin > %.2f) ===\n', dist_thr);
fprintf('Points considered: %d / %d (%.1f%% of grid)\n', sum(mask_far), numel(mask_far), 100*mean(mask_far));

fprintf('r_sigma: mean = %.2f, median = %.2f, max = %.2f\n', ...
    mean(r_sigma(mask_far),'omitnan'), median(r_sigma(mask_far),'omitnan'), max(r_sigma(mask_far)));
fprintf('r_var:   mean = %.2f, median = %.2f, max = %.2f\n', ...
    mean(r_var(mask_far),'omitnan'), median(r_var(mask_far),'omitnan'), max(r_var(mask_far)));

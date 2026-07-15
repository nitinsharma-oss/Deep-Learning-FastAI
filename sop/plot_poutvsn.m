%% ========================================================================
%  OUTAGE PROBABILITY vs ARRAY SIZE  --  P_out(N) and P_out(M)
%  at fixed average SNR  gamma_bar_E = 10 dB      (linear y-axis, exact only)
%  ------------------------------------------------------------------------
%  SNR convention :  gamma = gamma_bar_E * Z / E[Z]   (E[gamma] = gamma_bar)
%  Outage         :  gamma < gamma_th
%                    <=>  Z < z_th = E[Z]*gamma_th/gamma_bar
%  With E[Z] = k_g*theta_g the Meijer-G formula argument is
%                    x = z_th/theta_g = k_g*gamma_th/gamma_bar
%
%  Meijer-G outage formula:
%   P_out = 1/Gamma(k_g)*[ G^{1,1}_{1,2}(x|1;k_g,0)
%                        - G^{1,1}_{1,2}(L_B/theta_g|1;k_g,0) ]
%         = gammainc(x,k_g) - gammainc(L_B/theta_g,k_g)
%  (identity G^{1,1}_{1,2}(u|1;k,0) = gamma_lower(k,u); gammainc is the
%   regularized form, so the 1/Gamma(k_g) prefactor is already included).
%
%  THRESHOLD CHOICE.  On a LINEAR y-axis P_out is visible only near the
%  outage cliff.  With gamma_th = 1 and gamma_bar = 10 dB the system sits
%  ~10 dB inside the no-outage region and P_out ~ 10^-300 = 0 on any
%  linear axis.  We therefore set the threshold at a 1% margin below the
%  mean SNR, gamma_th = 0.99*gamma_bar, where P_out spans a visible range
%  and decays with the array size (pure diversity effect, k_g = M*N*482).
%  Adjust 'margin' below to move the operating point.
%  ========================================================================
clear; clc; close all;

%% ---------------- Truncated-Laplace orientation model ------------------
xa = 0; ya = 0; za = 3;   xu = 3; yu = 3; zu = 1.5;
Omega = pi/2;
d = sqrt((xa-xu)^2 + (ya-yu)^2 + (za-zu)^2);
a = -((xa-xu)/d)*cos(Omega) - ((ya-yu)/d)*sin(Omega);
b =  (za-zu)/d;

mu_theta = deg2rad(41.39);
sigma_L  = deg2rad(7.68);
b_theta  = sqrt(sigma_L^2 / 2);

mu_xi = a*sin(mu_theta) + b*cos(mu_theta);
b_xi  = b_theta * abs(a*cos(mu_theta) - b*sin(mu_theta));

xi_min = 0;
if a < 0, xi_max = b; else, xi_max = sqrt(a^2 + b^2); end

%% ------------------------- Moments of xi -------------------------------
alpha_int = xi_min - mu_xi;
beta_int  = xi_max - mu_xi;
Lambda = b_xi * (2 - exp(alpha_int/b_xi) - exp(-beta_int/b_xi));
E_u  = (b_xi / Lambda) * ( exp( alpha_int/b_xi)*(b_xi - alpha_int) ...
                         - exp(-beta_int /b_xi)*(beta_int + b_xi) );
E_u2 = (1 / Lambda) * ( 4*b_xi^3 ...
     - exp( alpha_int/b_xi)*(b_xi*alpha_int^2 - 2*b_xi^2*alpha_int + 2*b_xi^3) ...
     - exp(-beta_int /b_xi)*(b_xi*beta_int^2  + 2*b_xi^2*beta_int  + 2*b_xi^3) );
E_xi   = mu_xi + E_u;
Var_xi = E_u2 - E_u^2;

kg_per_elem = E_xi^2 / Var_xi;             % k_g contribution of ONE element
ak = 1.0000e-6;                            % per-element gain (for MC only)
fprintf('E[xi] = %.5f  Var[xi] = %.4e  ->  k_g per element = %.2f\n\n', ...
        E_xi, Var_xi, kg_per_elem);

%% ------------------------- Outage settings -----------------------------
gamma_bar_dB = 10;                         % average SNR (fixed)
gamma_bar    = 10^(gamma_bar_dB/10);
margin       = 0.99;                       % threshold margin below the mean
gamma_th     = margin * gamma_bar;         % outage threshold (linear)
L_B          = 0;                          % Meijer-G lower boundary
N_mc         = 1e5;                        % Monte Carlo trials per point

ratio = gamma_th / gamma_bar;              % = margin (used as z_th/E[Z])

%% ================= SWEEP 1:  P_out vs N  (M = 1) =======================
M1     = 1;
N_vec  = 1:128;
Pout_N = zeros(size(N_vec));
for ii = 1:numel(N_vec)
    kg = M1 * N_vec(ii) * kg_per_elem;
    x  = kg * ratio;                       % z_th/theta_g = k_g*g_th/g_bar
    Pout_N(ii) = gammainc(x, kg) - gammainc(L_B, kg);
end

% Monte Carlo at selected N (truncated-Laplace sampling)
N_mc_pts = [1 2 4 8 16 24 32 48 64 96 128];
Pmc_N = zeros(size(N_mc_pts));
for ii = 1:numel(N_mc_pts)
    N = N_mc_pts(ii);
    tot = N_mc * M1 * N;  xi_s = zeros(1, tot);  cnt = 0;
    while cnt < tot
        U  = rand(1, 2*(tot-cnt)) - 0.5;
        Ls = mu_xi - b_xi * sign(U) .* log(1 - 2*abs(U));
        ok = Ls(Ls >= xi_min & Ls <= xi_max);
        nd = min(length(ok), tot - cnt);
        xi_s(cnt+1 : cnt+nd) = ok(1:nd);  cnt = cnt + nd;
    end
    Z  = ak * sum(reshape(xi_s, M1*N, N_mc), 1);
    EZ = M1 * N * ak * E_xi;
    Pmc_N(ii) = mean(Z/EZ < ratio & Z >= L_B);
end

%% ================= SWEEP 2:  P_out vs M  (N = 16) ======================
N2     = 16;
M_vec  = 1:16;
Pout_M = zeros(size(M_vec));
for ii = 1:numel(M_vec)
    kg = M_vec(ii) * N2 * kg_per_elem;
    x  = kg * ratio;
    Pout_M(ii) = gammainc(x, kg) - gammainc(L_B, kg);
end

% Monte Carlo at every M
Pmc_M = zeros(size(M_vec));
for ii = 1:numel(M_vec)
    Mv = M_vec(ii);
    tot = N_mc * Mv * N2;  xi_s = zeros(1, tot);  cnt = 0;
    while cnt < tot
        U  = rand(1, 2*(tot-cnt)) - 0.5;
        Ls = mu_xi - b_xi * sign(U) .* log(1 - 2*abs(U));
        ok = Ls(Ls >= xi_min & Ls <= xi_max);
        nd = min(length(ok), tot - cnt);
        xi_s(cnt+1 : cnt+nd) = ok(1:nd);  cnt = cnt + nd;
    end
    Z  = ak * sum(reshape(xi_s, Mv*N2, N_mc), 1);
    EZ = Mv * N2 * ak * E_xi;
    Pmc_M(ii) = mean(Z/EZ < ratio & Z >= L_B);
end

%% ============================== PLOTS ==================================
figure('Color','w','Position',[80 80 1150 500]);

subplot(1,2,1); hold on; grid on;
plot(N_vec, Pout_N, 'b-', 'LineWidth', 2.2);
plot(N_mc_pts, Pmc_N, 'ko', 'MarkerFaceColor', [0.3 0.8 0.3], 'MarkerSize', 6);
xlabel('Number of OIRS elements  N   (M = 1)');
ylabel('P_{out}');
title(sprintf('(a) P_{out} vs N   (\\gamma\\_bar_E = %g dB, \\gamma_{th} = %.2f)', ...
      gamma_bar_dB, gamma_th));
legend('Exact (Meijer-G formula)', 'Monte Carlo', 'Location', 'northeast');
xlim([1 128]); ylim([0 max(Pout_N)*1.15]);

subplot(1,2,2); hold on; grid on;
plot(M_vec, Pout_M, 'b-o', 'LineWidth', 2.2, 'MarkerFaceColor', 'b', ...
     'MarkerSize', 4.5);
plot(M_vec, Pmc_M, 'ks', 'MarkerFaceColor', [0.3 0.8 0.3], 'MarkerSize', 6.5);
xlabel(sprintf('Number of transmitters  M   (N = %d)', N2));
ylabel('P_{out}');
title(sprintf('(b) P_{out} vs M   (\\gamma\\_bar_E = %g dB, \\gamma_{th} = %.2f)', ...
      gamma_bar_dB, gamma_th));
legend('Exact (Meijer-G formula)', 'Monte Carlo', 'Location', 'northeast');
xlim([1 16]); set(gca,'XTick',0:2:16); ylim([0 max(Pout_M)*1.2]);

%% ------------------------- Console table -------------------------------
fprintf('   N (M=1) |    k_g     |  P_out exact |  P_out MC\n');
fprintf('  --------------------------------------------------\n');
for ii = 1:numel(N_mc_pts)
    nn = N_mc_pts(ii);
    fprintf('   %6d | %10.1f | %12.5f | %9.5f\n', ...
        nn, M1*nn*kg_per_elem, Pout_N(nn), Pmc_N(ii));
end
fprintf('\n   M (N=16)|    k_g     |  P_out exact |  P_out MC\n');
fprintf('  --------------------------------------------------\n');
for ii = [1 2 4 8 16]
    fprintf('   %6d | %10.1f | %12.5f | %9.5f\n', ...
        M_vec(ii), M_vec(ii)*N2*kg_per_elem, Pout_M(ii), Pmc_M(ii));
end
fprintf(['\nNote: threshold set at %.0f%% of the mean SNR (gamma_th = %.2f)\n' ...
         'so that P_out is visible on a linear axis; with gamma_th = 1 the\n' ...
         'operating point is ~10 dB inside the no-outage region and\n' ...
         'P_out ~ 10^-300 (indistinguishable from zero without a log axis).\n'], ...
         100*margin, gamma_th);

%% ========================================================================
%  OUTAGE PROBABILITY vs ARRAY SIZE  --  P_out(N) and P_out(M)
%  at fixed average SNR  gamma_bar_E = 10 dB
%  ------------------------------------------------------------------------
%  SNR convention:  gamma = gamma_bar_E * Z / E[Z]     (so E[gamma] =
%  gamma_bar_E exactly, for every M,N).  This is the standard normalization
%  that isolates the DIVERSITY effect of the array size; without it, the
%  raw per-element gain ak = 1e-6 puts every configuration in total outage
%  at 10 dB (P_out = 1 flat line -- see note at the bottom).
%
%  Outage:  gamma < gamma_th  <=>  Z < z_th = E[Z]*gamma_th/gamma_bar
%  Since E[Z] = k_g*theta_g, the Meijer-G formula argument becomes
%
%      x = z_th/theta_g = k_g * gamma_th / gamma_bar
%
%  Meijer-G outage formula:
%   P_out = 1/Gamma(k_g) * [ G^{1,1}_{1,2}( x | 1 ; k_g,0 )
%                          - G^{1,1}_{1,2}( L_B/theta_g | 1 ; k_g,0 ) ]
%         =  gammainc(x, k_g) - gammainc(L_B/theta_g, k_g)
%  (identity G^{1,1}_{1,2}(u|1;k,0) = gamma_lower(k,u)).
%
%  Shape grows linearly with the array:  k_g = M*N*E[xi]^2/Var[xi],
%  so both sweeps below are pure diversity-order sweeps.  P_out values are
%  astronomically small (10^-hundreds), hence everything is computed in
%  the log10 domain (Monte Carlo cannot reach these levels -- noted).
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

kg_per_elem = E_xi^2 / Var_xi;              % k_g contribution of ONE element
fprintf('E[xi] = %.5f  Var[xi] = %.4e  ->  k_g per element = %.2f\n\n', ...
        E_xi, Var_xi, kg_per_elem);

%% ------------------------- Outage settings -----------------------------
gamma_bar_dB = 10;                          % average SNR (fixed)
gamma_bar    = 10^(gamma_bar_dB/10);
gamma_th     = 1.0;                         % outage threshold (linear)
L_B          = 0;                           % Meijer-G lower boundary

%% ================= SWEEP 1:  P_out vs N  (M = 1) =======================
M1     = 1;
N_vec  = 1:128;
l10P_N = zeros(size(N_vec));
l10A_N = zeros(size(N_vec));
for ii = 1:numel(N_vec)
    kg = M1 * N_vec(ii) * kg_per_elem;
    x  = kg * gamma_th / gamma_bar;         % z_th/theta_g = k_g*g_th/g_bar
    if x >= kg + 1
        l10P_N(ii) = log10(gammainc(x, kg));
    else
        S = 1; term = 1; nn = 0;
        while true
            nn = nn + 1;  term = term * x/(kg+nn);  S = S + term;
            if term < eps*S || nn > 1e6, break; end
        end
        l10P_N(ii) = (kg*log(x) - x - gammaln(kg+1) + log(S)) / log(10);
    end
    l10A_N(ii) = (kg*log(x) - gammaln(kg+1)) / log(10);   % asymptote
end

%% ================= SWEEP 2:  P_out vs M  (N = 16) ======================
N2     = 16;
M_vec  = 1:16;
l10P_M = zeros(size(M_vec));
l10A_M = zeros(size(M_vec));
for ii = 1:numel(M_vec)
    kg = M_vec(ii) * N2 * kg_per_elem;
    x  = kg * gamma_th / gamma_bar;         % z_th/theta_g = k_g*g_th/g_bar
    if x >= kg + 1
        l10P_M(ii) = log10(gammainc(x, kg));
    else
        S = 1; term = 1; nn = 0;
        while true
            nn = nn + 1;  term = term * x/(kg+nn);  S = S + term;
            if term < eps*S || nn > 1e6, break; end
        end
        l10P_M(ii) = (kg*log(x) - x - gammaln(kg+1) + log(S)) / log(10);
    end
    l10A_M(ii) = (kg*log(x) - gammaln(kg+1)) / log(10);
end

%% ============================== PLOTS ==================================
figure('Color','w','Position',[80 80 1150 520]);

subplot(1,2,1); hold on; grid on;
plot(N_vec, l10P_N, 'b-',  'LineWidth', 2.2);
plot(N_vec, l10A_N, 'r--', 'LineWidth', 1.6);
xlabel('Number of OIRS elements  N   (M = 1)');
ylabel('log_{10} P_{out}');
title(sprintf('(a) P_{out} vs N   (\\gamma\\_bar_E = %g dB, \\gamma_{th} = %g)', ...
      gamma_bar_dB, gamma_th));
legend('Exact (Meijer-G / \gamma_{low} form)', ...
       'Asymptote  x^{k_g}/\Gamma(k_g{+}1)', 'Location', 'southwest');
xlim([1 128]);

subplot(1,2,2); hold on; grid on;
plot(M_vec, l10P_M, 'b-o',  'LineWidth', 2.2, 'MarkerFaceColor','b', ...
     'MarkerSize', 4.5);
plot(M_vec, l10A_M, 'r--s', 'LineWidth', 1.6, 'MarkerSize', 5);
xlabel(sprintf('Number of transmitters  M   (N = %d)', N2));
ylabel('log_{10} P_{out}');
title(sprintf('(b) P_{out} vs M   (\\gamma\\_bar_E = %g dB, \\gamma_{th} = %g)', ...
      gamma_bar_dB, gamma_th));
legend('Exact (Meijer-G / \gamma_{low} form)', ...
       'Asymptote  x^{k_g}/\Gamma(k_g{+}1)', 'Location', 'southwest');
xlim([1 16]); set(gca,'XTick',0:2:16);

%% ------------------------- Console table -------------------------------
fprintf('   N (M=1) |    k_g     | log10 P_out | log10 P_asym\n');
fprintf('  ---------------------------------------------------\n');
for ii = [1 2 4 8 16 32 64 128]
    fprintf('   %6d | %10.1f | %11.1f | %12.1f\n', ...
        N_vec(ii), M1*N_vec(ii)*kg_per_elem, l10P_N(ii), l10A_N(ii));
end
fprintf('\n   M (N=16)|    k_g     | log10 P_out | log10 P_asym\n');
fprintf('  ---------------------------------------------------\n');
for ii = [1 2 4 8 16]
    fprintf('   %6d | %10.1f | %11.1f | %12.1f\n', ...
        M_vec(ii), M_vec(ii)*N2*kg_per_elem, l10P_M(ii), l10A_M(ii));
end

fprintf(['\nNotes:\n' ...
  '(1) SNR convention gamma = gamma_bar*Z/E[Z] (E[gamma] = gamma_bar).\n' ...
  '    With the UN-normalized convention gamma = gamma_bar*Z and\n' ...
  '    ak = 1e-6, every (M,N) here is in total outage at 10 dB\n' ...
  '    (P_out = 1 flat): meaningful gamma_bar is ~45-60 dB there.\n' ...
  '(2) P_out levels are ~10^-300 and below -> Monte Carlo cannot\n' ...
  '    verify them; log-domain analytics is the only route.\n' ...
  '(3) Slope: log10 P_out is nearly linear in the array size because\n' ...
  '    k_g = M*N*%.0f and log10 P ~ -k_g*[g_bar-dependent const].\n'], ...
  kg_per_elem);

% =========================================================================
% ASYMPTOTIC Outage Probability: Uniform vs. Truncated-Laplace Orientation
% OIRS-Aided VLC System with Zero-Outage Cliff
%
% Combines: (a) the base outage script (exact Gamma analysis + Monte Carlo)
%           (b) the moment-matching script (truncated-Laplace moments)
% and ADDS the high-SNR ASYMPTOTIC analysis:
%
%   Z ~ Gamma(kg, th), outage threshold z_th = sqrt(gamma_th / SNR).
%   Small-argument expansion of the lower incomplete gamma,
%       gammainc: gamma(k,x) ~ x^k / k   as x -> 0,
%   gives the asymptote
%       P_out^inf(SNR) = z_th^kg / (Gamma(kg+1) * th^kg)
%                      = (gamma_th/SNR)^(kg/2) / (Gamma(kg+1) th^kg)
%   =>  DIVERSITY ORDER  d = kg/2
%   =>  CODING GAIN      Gc = (th^2/gamma_th) * Gamma(kg+1)^(1/d)
%       so that P_out^inf = (Gc * SNR)^(-d).
%
%   Truncation-corrected asymptote (physical lower bound LB):
%       P_out^inf,tr = [z_th^kg - LB^kg]^+ / (Gamma(kg+1) th^kg)
%   which hits EXACTLY zero at the cliff  SNR_cliff = gamma_th / LB^2.
%
%   NUMERICAL NOTE: for the truncated-Laplace model kg ~ 1.6e3, so
%   Gamma(kg+1) overflows double precision. All asymptotes are therefore
%   computed in the LOG10 DOMAIN via gammaln.
% =========================================================================
clear; clc; close all;

%% 1. System Parameters
M  = 1;                 % Number of LEDs
N  = 4;                 % Number of OIRS elements
ak = 1e-4;              % Aggregated path loss and receiver gain
R  = 0.5;               % Target data rate (bps/Hz)
gamma_th = 2^(2*R) - 1; % Required electrical SNR threshold

SNR_dB     = 0:0.1:150;
SNR_linear = 10.^(SNR_dB/10);
z_th       = sqrt(gamma_th ./ SNR_linear);   % required equivalent gain

%% 2. Orientation Moments: Uniform Model (Baseline)
Psi        = 89.9*(pi/180);                  % FoV semi-angle (rad)
E_xi_uni   = sin(Psi)/Psi;
E_xi2_uni  = 0.5 + sin(2*Psi)/(4*Psi);
Var_xi_uni = E_xi2_uni - E_xi_uni^2;
xi_min_uni = cos(Psi);

%% 3. Orientation Moments: Truncated-Laplace Model (Proposed)
mu_xi      = 0.8789;
b_xi       = 0.03600;
xi_min_lap = cos(Psi);
xi_max_lap = 0.9574;

alpha  = xi_min_lap - mu_xi;
beta   = xi_max_lap - mu_xi;
Lambda = b_xi*(2 - exp(alpha/b_xi) - exp(-beta/b_xi));

E_u  = (b_xi/Lambda)*( exp(alpha/b_xi)*(b_xi - alpha) ...
                     - exp(-beta/b_xi)*(beta + b_xi) );
E_u2 = (1/Lambda)*( 4*b_xi^3 ...
     - exp(alpha/b_xi) *(b_xi*alpha^2 - 2*b_xi^2*alpha + 2*b_xi^3) ...
     - exp(-beta/b_xi)*(b_xi*beta^2  + 2*b_xi^2*beta  + 2*b_xi^3) );

E_xi_lap   = mu_xi + E_u;
Var_xi_lap = E_u2 - E_u^2;

%% 4. Gamma Approximation Parameters (moment matching)
% Uniform
E_Z_uni  = M*N*ak*E_xi_uni;   Var_Z_uni = M*N*ak^2*Var_xi_uni;
kg_uni   = E_Z_uni^2/Var_Z_uni;   th_uni = Var_Z_uni/E_Z_uni;
LB_uni   = M*N*ak*xi_min_uni;
% Truncated-Laplace
E_Z_lap  = M*N*ak*E_xi_lap;   Var_Z_lap = M*N*ak^2*Var_xi_lap;
kg_lap   = E_Z_lap^2/Var_Z_lap;   th_lap = Var_Z_lap/E_Z_lap;
LB_lap   = M*N*ak*xi_min_lap;

%% 5. Exact Analytical Outage + Monte Carlo (from base script)
num_realizations = 1e6;
Z_sim_uni = max(gamrnd(kg_uni, th_uni, 1, num_realizations), LB_uni);
Z_sim_lap = max(gamrnd(kg_lap, th_lap, 1, num_realizations), LB_lap);

P_out_uni_ana = gamcdf(z_th, kg_uni, th_uni) - gamcdf(LB_uni, kg_uni, th_uni);
P_out_lap_ana = gamcdf(z_th, kg_lap, th_lap) - gamcdf(LB_lap, kg_lap, th_lap);

P_out_uni_sim = zeros(size(SNR_dB));
P_out_lap_sim = zeros(size(SNR_dB));
for i = 1:length(SNR_dB)
    P_out_uni_sim(i) = mean(Z_sim_uni < z_th(i));
    P_out_lap_sim(i) = mean(Z_sim_lap < z_th(i));
end

%% 6. ASYMPTOTIC Outage (log10 domain, overflow-safe)  <-- NEW SECTION
% log10 P_out^inf     = [ kg*ln(z_th/th) - gammaln(kg+1) ] / ln(10)
% log10 P_out^inf,tr  = above + log10( 1 - (LB/z_th)^kg ),  = -inf past cliff
log10_asym = @(zt, kg, th) (kg.*log(zt./th) - gammaln(kg+1))/log(10);

trunc_corr = @(zt, kg, LB) log10(max(1 - min((LB./zt).^kg, 1), 0)); % -inf if zt<=LB

L10_uni_asym    = log10_asym(z_th, kg_uni, th_uni);
L10_uni_asym_tr = L10_uni_asym + trunc_corr(z_th, kg_uni, LB_uni);
L10_lap_asym    = log10_asym(z_th, kg_lap, th_lap);
L10_lap_asym_tr = L10_lap_asym + trunc_corr(z_th, kg_lap, LB_lap);

% Diversity order and coding gain, P_out^inf = (Gc*SNR)^(-d):
d_uni  = kg_uni/2;
d_lap  = kg_lap/2;
Gc_uni = (th_uni^2/gamma_th) * exp(gammaln(kg_uni+1)/d_uni);
Gc_lap = (th_lap^2/gamma_th) * exp(gammaln(kg_lap+1)/d_lap);

% Zero-outage cliff (same for both models: LB depends only on cos(Psi))
SNR_cliff_dB = 10*log10(gamma_th / LB_lap^2);

fprintf('--- Asymptotic Analysis ---\n');
fprintf('Uniform          : kg = %8.2f  ->  diversity d = %8.2f, Gc = %.3e (%.1f dB)\n', ...
        kg_uni, d_uni, Gc_uni, 10*log10(Gc_uni));
fprintf('Truncated-Laplace: kg = %8.2f  ->  diversity d = %8.2f, Gc = %.3e (%.1f dB)\n', ...
        kg_lap, d_lap, Gc_lap, 10*log10(Gc_lap));
fprintf('Zero-outage cliff: %.2f dB (both models, LB = %.3e)\n\n', SNR_cliff_dB, LB_lap);

%% 7. Plotting
floor_val = 1e-8;
clip = @(P) max(P, floor_val);

figure('Color','w','Position',[100 100 820 600]);
% exact
semilogy(SNR_dB, clip(P_out_uni_ana), 'b-',  'LineWidth', 2); hold on;
semilogy(SNR_dB, clip(P_out_lap_ana), 'r-',  'LineWidth', 2);
% Monte Carlo markers (sparse)
idx = 1:40:length(SNR_dB);
semilogy(SNR_dB(idx), clip(P_out_uni_sim(idx)), 'bs', 'MarkerSize', 7, 'LineWidth', 1.2);
semilogy(SNR_dB(idx), clip(P_out_lap_sim(idx)), 'ro', 'MarkerSize', 7, 'LineWidth', 1.2);
% asymptotes (plot only where they are below 1, i.e. log10 <= 0)
mU = L10_uni_asym_tr <= 0;
mL = L10_lap_asym_tr <= 0;
semilogy(SNR_dB(mU), clip(10.^L10_uni_asym_tr(mU)), 'b--', 'LineWidth', 2.2);
semilogy(SNR_dB(mL), clip(10.^L10_lap_asym_tr(mL)), 'r--', 'LineWidth', 2.2);
% cliff
xline(SNR_cliff_dB, 'k--', 'LineWidth', 2.2, 'Label', 'Zero-Outage Cliff', ...
      'LabelVerticalAlignment','bottom', 'LabelHorizontalAlignment','left', 'FontSize', 12);

grid on; ylim([floor_val 1]); xlim([min(SNR_dB) max(SNR_dB)]);
xlabel('Average Transmit SNR, $\bar{\gamma}$ (dB)', 'Interpreter','latex', 'FontSize', 14);
ylabel('Outage Probability, $P_{out}$',            'Interpreter','latex', 'FontSize', 14);
title(sprintf(['Exact, Simulated and Asymptotic Outage ($N=%d$, $M=%d$)\n' ...
    'diversity: $d_{uni}=%.1f$, $d_{lap}=%.1f$'], N, M, d_uni, d_lap), ...
    'Interpreter','latex', 'FontSize', 14);
legend({'Uniform (Exact)','Trunc.-Laplace (Exact)', ...
        'Uniform (Simulated)','Trunc.-Laplace (Simulated)', ...
        'Uniform (Asymptotic)','Trunc.-Laplace (Asymptotic)', ...
        'Theoretical Cliff'}, ...
        'Interpreter','latex','FontSize',11,'Location','southwest');
set(gca,'TickLabelInterpreter','latex','FontSize',12);

%% 8. Asymptote Tightness Check (printed table)
fprintf('--- Asymptote vs Exact (Uniform model) ---\n');
fprintf('%8s %14s %14s %10s\n','SNR(dB)','Exact','Asymptote','Ratio');
for snr = [80 90 100 110 120]
    j = find(abs(SNR_dB - snr) < 1e-9, 1);
    ex = P_out_uni_ana(j);
    as = 10^L10_uni_asym_tr(j);
    fprintf('%8d %14.3e %14.3e %10.3f\n', snr, ex, as, as/max(ex,realmin));
end
fprintf(['\n(Truncated-Laplace: kg ~ %.0f makes its waterfall a near-vertical\n' ...
         ' brick wall around %.0f dB; exact values there underflow double\n' ...
         ' precision, so its asymptote is meaningful only in log10 form.)\n'], ...
         kg_lap, 10*log10(gamma_th/E_Z_lap^2));

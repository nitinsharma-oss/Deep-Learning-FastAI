%% ========================================================================
%  OUTAGE PROBABILITY (EXACT + ASYMPTOTIC) FOR OIRS-AIDED CHANNEL
%  ------------------------------------------------------------------------
%  Formula implemented (Meijer-G form):
%
%      P_out = 1/Gamma(k_g) * [ G^{1,1}_{1,2}( z_th/theta_g | 1 ; k_g,0 )
%                             - G^{1,1}_{1,2}( L_B /theta_g | 1 ; k_g,0 ) ]
%
%  Using the standard identity  G^{1,1}_{1,2}(x | 1 ; k,0) = gamma_lower(k,x)
%  (lower incomplete gamma), the formula is EXACTLY
%
%      P_out = [ gammainc(z_th/theta_g,k_g) - gammainc(L_B/theta_g,k_g) ]
%            = F_Z(z_th) - F_Z(L_B),      Z ~ Gamma(k_g, theta_g)
%
%  High-SNR / small-argument asymptote  (gamma_lower(k,x) ~ x^k / k):
%
%      P_out^asym = [ (z_th/theta_g)^k_g - (L_B/theta_g)^k_g ] / Gamma(k_g+1)
%
%  NUMERICAL NOTE.  With the physical parameters below, moment matching
%  yields a VERY large shape parameter (k_g ~ 7.7e3), so (x)^(k_g) and
%  Gamma(k_g+1) overflow double precision.  Both the exact CDF and the
%  asymptote are therefore ALSO evaluated in the log10 domain (helper
%  functions at the end of the file), which is stable for any k_g and lets
%  the deep-tail diversity slope k_g be displayed correctly.
%
%  The Gamma parameters (k_g, theta_g) come from the SAME moment-matching
%  chain as the original channel-verification script.
%  ========================================================================
clear; clc; close all;

%% ------------------------- System Parameters ---------------------------
rho_OIRS = 0.9;
A_PD     = 1e-4;
dA_i     = 0.01;
m        = 1;
T_gain   = 1;

n_ref    = 1.5;
Psi_FOV  = deg2rad(80);
G_conc   = n_ref^2 / (sin(Psi_FOV)^2);

%% ------------------------- Coordinates ---------------------------------
T = [2.5 2.5 3  ];      % Transmitter (AP)
P = [2.5 0   1.5];      % OIRS array position
R = [2.5 2.5 0  ];      % Receiver (UE)

d_ai = norm(P - T);     % AP  -> IRS
d_ik = norm(R - P);     % IRS -> UE

%% ---------------- cos^m(Phi_ai) : AP radiation angle -------------------
PT = P - T;
OT = R - T;
cos_Phi_ai  = dot(PT, OT) / (norm(PT)*norm(OT));
cosm_Phi_ai = cos_Phi_ai^m;

%% ---------------- cos(xi_ai) : incidence on OIRS -----------------------
alpha_rot = deg2rad(0);
beta_rot  = deg2rad(0);

n0 = [0;1;0];
Rz = [cos(alpha_rot) -sin(alpha_rot) 0;
      sin(alpha_rot)  cos(alpha_rot) 0;
      0               0              1];
Rx = [1 0              0;
      0 cos(beta_rot)  sin(beta_rot);
      0 sin(beta_rot)  cos(beta_rot)];
Rcom = Rz * Rx;
nr   = Rcom * n0;

incident  = (T - P).';
cos_xi_ai = dot(nr, incident) / (norm(nr)*norm(incident));

%% ---------------- cos(Phi_ik) : reflected ray at UE --------------------
I     = (P - T).';  I = I / norm(I);
R_ref = I - 2*dot(I,nr)*nr;
n_rx  = [0;0;1];
cos_Phi_ik = dot(n_rx, R_ref) / (norm(n_rx)*norm(R_ref));

%% ------------------------- Channel Parameters --------------------------
M    = 1;
N    = 16;
N_mc = 1e5;

% Per-element deterministic gain (kept identical to original script)
% ak = rho_OIRS*((m+1)*A_PD)/(2*pi^2*d_ai^2*d_ik^2) * dA_i * ...
%      cosm_Phi_ai * cos_xi_ai * cos_Phi_ik * T_gain * G_conc;
ak = 1.0000e-6;

%% ------------------ Geometry for Truncated Laplace ---------------------
xa = 0; ya = 0; za = 3;
xu = 3; yu = 3; zu = 1.5;

Omega = pi/2;
d = sqrt((xa-xu)^2 + (ya-yu)^2 + (za-zu)^2);

a = -((xa-xu)/d)*cos(Omega) - ((ya-yu)/d)*sin(Omega);
b =  (za-zu)/d;

mu_theta_deg = 41.39;
sigma_L_deg  = 7.68;
mu_theta = deg2rad(mu_theta_deg);
sigma_L  = deg2rad(sigma_L_deg);

b_theta = sqrt(sigma_L^2 / 2);

% Linearization of xi = a*sin(theta) + b*cos(theta) around mu_theta
mu_xi = a*sin(mu_theta) + b*cos(mu_theta);
b_xi  = b_theta * abs(a*cos(mu_theta) - b*sin(mu_theta));

xi_min = 0;
if a < 0
    xi_max = b;
else
    xi_max = sqrt(a^2 + b^2);
end

%% ------------------------- Moments of xi -------------------------------
alpha_int = xi_min - mu_xi;
beta_int  = xi_max - mu_xi;

Lambda = b_xi * (2 - exp(alpha_int/b_xi) - exp(-beta_int/b_xi));

E_u  = (b_xi / Lambda) * ...
       ( exp( alpha_int/b_xi)*(b_xi - alpha_int) ...
       - exp(-beta_int /b_xi)*(beta_int + b_xi) );

E_u2 = (1 / Lambda) * ...
       ( 4*b_xi^3 ...
       - exp( alpha_int/b_xi)*(b_xi*alpha_int^2 - 2*b_xi^2*alpha_int + 2*b_xi^3) ...
       - exp(-beta_int /b_xi)*(b_xi*beta_int^2  + 2*b_xi^2*beta_int  + 2*b_xi^3) );

E_xi   = mu_xi + E_u;
Var_xi = E_u2 - E_u^2;

%% ---------------- Gamma moment matching:  Z ~ Gamma(kg, theta_g) -------
E_Z   = M * N * ak   * E_xi;
Var_Z = M * N * ak^2 * Var_xi;

kg      = (E_Z^2) / Var_Z;          % shape  k_{g_k}
theta_g = Var_Z / E_Z;              % scale  theta_{g_k}
sigma_Z = sqrt(Var_Z);

fprintf('--------------------------------------------------------\n');
fprintf('E[xi]      = %.5f     Var[xi] = %.4e\n', E_xi, Var_xi);
fprintf('E[Z]       = %.4e   std[Z]  = %.4e\n', E_Z, sigma_Z);
fprintf('Gamma k_g  = %.2f   (also the diversity order)\n', kg);
fprintf('Gamma th_g = %.4e\n', theta_g);
fprintf('--------------------------------------------------------\n\n');

%% ================== OUTAGE PROBABILITY COMPUTATION =====================
% Lower boundary L_B of the Meijer-G formula.  L_B = 0 recovers the
% classical outage P_out = F_Z(z_th).  Set it to a positive value (e.g. a
% minimum detectable aggregated gain) if the model requires it.
L_B = 0;

% --- Sweep 1: transition region (where outage is observable / MC-able) --
z_tr = linspace(max(E_Z - 8*sigma_Z, 0), E_Z + 5*sigma_Z, 500);
Pout_tr_exact = gammainc(z_tr./theta_g, kg) - gammainc(L_B/theta_g, kg);

% --- Sweep 2: wide log sweep for the deep tail / diversity slope --------
z_wd = logspace(log10(E_Z) - 2, log10(E_Z) + 0.15, 400);
x_wd = z_wd ./ theta_g;

% Exact CDF in log10 domain (stable even when P_out < 1e-308).
% For x < k+1 use the series  P(k,x) = x^k e^{-x}/Gamma(k+1) * S,
% S = sum_{n>=0} x^n/((k+1)...(k+n)); otherwise gammainc is safe.
log10P_exact = zeros(size(x_wd));
for jj = 1:numel(x_wd)
    x = x_wd(jj);
    if x >= kg + 1
        log10P_exact(jj) = log10(gammainc(x, kg));
    else
        S = 1; term = 1; nn = 0;
        while true
            nn = nn + 1;
            term = term * x / (kg + nn);
            S = S + term;
            if term < eps*S || nn > 1e6, break; end
        end
        log10P_exact(jj) = (kg*log(x) - x - gammaln(kg+1) + log(S)) / log(10);
    end
end
if L_B > 0     % subtract the L_B term in the log domain (series form)
    xB = L_B/theta_g;  S = 1; term = 1; nn = 0;
    while true
        nn = nn + 1;  term = term * xB / (kg + nn);  S = S + term;
        if term < eps*S || nn > 1e6, break; end
    end
    lB = (kg*log(xB) - xB - gammaln(kg+1) + log(S)) / log(10);
    log10P_exact = log10P_exact + log10(1 - 10.^(lB - log10P_exact));
end

% Asymptote in log10 domain:
% log10 Pasym = [ kg*ln(x) - gammaln(kg+1) ] / ln(10)   (for L_B = 0)
log10P_asym = (kg*log(x_wd) - gammaln(kg + 1)) / log(10);
if L_B > 0
    lBa = (kg*log(L_B/theta_g) - gammaln(kg + 1)) / log(10);
    log10P_asym = log10P_asym + log10(1 - 10.^(lBa - log10P_asym));
end

% --- Direct Meijer-G evaluation (cross-check; needs Symbolic Toolbox) ---
try
    idx_chk = round(linspace(numel(z_tr)*0.55, numel(z_tr), 6));
    Pout_mg = zeros(size(idx_chk));
    for ii = 1:numel(idx_chk)
        g1 = double(meijerG(1, [], kg, 0, z_tr(idx_chk(ii))/theta_g));
        g2 = double(meijerG(1, [], kg, 0, L_B/theta_g));
        Pout_mg(ii) = (g1 - g2) / gamma(kg);
    end
    err_mg = max(abs(Pout_mg - Pout_tr_exact(idx_chk)));
    fprintf('Meijer-G identity check: max |G-form - gammainc-form| = %.3e\n\n', err_mg);
    have_mg = true;
catch
    fprintf(['Symbolic Toolbox / meijerG unavailable -> skipping direct ' ...
             'Meijer-G evaluation.\n(The gammainc form is mathematically ' ...
             'identical: G^{1,1}_{1,2}(x|1;k,0) = gamma_lower(k,x).)\n\n']);
    have_mg = false;
end

%% ==================== MONTE CARLO VERIFICATION =========================
% Truncated-Laplace sampling of xi (same sampler as original script)
total_paths = N_mc * M * N;
xi_samples  = zeros(1, total_paths);
count = 0;
while count < total_paths
    batch_size = 2 * (total_paths - count);
    U = rand(1, batch_size) - 0.5;
    L_samples = mu_xi - b_xi * sign(U) .* log(1 - 2*abs(U));
    valid = L_samples(L_samples >= xi_min & L_samples <= xi_max);
    needed = min(length(valid), total_paths - count);
    xi_samples(count+1 : count+needed) = valid(1:needed);
    count = count + needed;
end
Z_mc = ak * sum(reshape(xi_samples, M*N, N_mc), 1);

z_mc_pts = linspace(max(E_Z - 5*sigma_Z, 0), E_Z + 4*sigma_Z, 16);
Pout_mc  = arrayfun(@(z) mean(Z_mc >= L_B & Z_mc <= z), z_mc_pts);

%% ============================== PLOTS ==================================
figure('Color','w','Position',[80 80 1150 520]);

% ---- (a) Transition region: exact vs Monte Carlo -----------------------
subplot(1,2,1); hold on; grid on;
semilogy_safe = @(x,y) plot(x, max(y, 1e-12), 'LineWidth', 2);
plot(z_tr/theta_g, max(Pout_tr_exact, 1e-12), 'b-', 'LineWidth', 2);
msk = Pout_mc > 0;
plot(z_mc_pts(msk)/theta_g, Pout_mc(msk), 'ko', ...
     'MarkerSize', 7, 'MarkerFaceColor', [0.3 0.8 0.3]);
if have_mg
    plot(z_tr(idx_chk)/theta_g, Pout_mg, 'ms', 'MarkerSize', 10);
    legend('Exact (Meijer-G = \gamma_{low} form)', 'Monte Carlo', ...
           'Meijer-G direct', 'Location', 'southeast');
else
    legend('Exact (Meijer-G = \gamma_{low} form)', 'Monte Carlo', ...
           'Location', 'southeast');
end
set(gca, 'YScale', 'log'); ylim([1e-6 1.5]);
xlabel('z_{th}/\theta_{g_k}');
ylabel('P_{out}');
title(sprintf('(a) Outage, transition region  (k_g = %.0f)', kg));

% ---- (b) Deep tail in log10 domain: exact vs asymptote -----------------
subplot(1,2,2); hold on; grid on;
plot(log10(x_wd), log10P_exact, 'b-',  'LineWidth', 2);
plot(log10(x_wd), log10P_asym,  'r--', 'LineWidth', 2);
legend('Exact  log_{10}P_{out}', ...
       sprintf('Asymptote, slope = k_g = %.0f', kg), 'Location', 'southeast');
xlabel('log_{10}( z_{th}/\theta_{g_k} )');
ylabel('log_{10} P_{out}');
title('(b) Deep tail: asymptote & diversity order');

% Numerical slope check of BOTH curves in the deep-tail region
lo = 5; hi = 60;
sl_ex = polyfit(log10(x_wd(lo:hi)), log10P_exact(lo:hi), 1);
sl_as = polyfit(log10(x_wd(lo:hi)), log10P_asym(lo:hi),  1);
fprintf('Deep-tail slope: exact = %.2f,  asymptote = %.2f,  theory k_g = %.2f\n', ...
        sl_ex(1), sl_as(1), kg);

%% -------- Table of sample values (log10 exact vs log10 asymptote) ------
fprintf('\n  z_th/theta_g  | log10 P_exact | log10 P_asym  |  gap (dB-dec)\n');
fprintf(' ---------------------------------------------------------------\n');
for zz = round(linspace(1, numel(x_wd), 10))
    fprintf('   %10.4g |  %12.2f |  %12.2f |  %10.3g\n', ...
        x_wd(zz), log10P_exact(zz), log10P_asym(zz), ...
        log10P_asym(zz) - log10P_exact(zz));
end
fprintf(['\nNote: with these physical parameters k_g is very large, so the\n' ...
         'outage curve is a sharp cliff near E[Z]; the power-law asymptote\n' ...
         'is tight only in the (extremely deep) tail, as panel (b) shows.\n']);

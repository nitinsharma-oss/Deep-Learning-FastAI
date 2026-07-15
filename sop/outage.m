%% ========================================================================
%  OUTAGE PROBABILITY vs AVERAGE SNR (gamma_bar_E)  --  Meijer-G formula
%  ------------------------------------------------------------------------
%  Instantaneous SNR :  gamma = gamma_bar_E * Z ,  Z ~ Gamma(k_g, theta_g)
%  Outage event      :  gamma < gamma_th
%                       <=>  Z < z_th(gamma_bar_E) = gamma_th / gamma_bar_E
%
%  Meijer-G outage formula (with the lower boundary L_B):
%
%   P_out = 1/Gamma(k_g) * [ G^{1,1}_{1,2}( z_th/theta_g | 1 ; k_g,0 )
%                          - G^{1,1}_{1,2}( L_B /theta_g | 1 ; k_g,0 ) ]
%
%  Identity  G^{1,1}_{1,2}(x|1;k,0) = gamma_lower(k,x)  gives exactly
%
%   P_out(gamma_bar) = gammainc( gamma_th/(gamma_bar*theta_g), k_g )
%                    - gammainc( L_B/theta_g,                  k_g )
%
%  Asymptote (high gamma_bar):  gamma_lower(k,x) ~ x^k / k  =>
%
%   P_out^asym = [ (gamma_th/(gamma_bar*theta_g))^k_g - (L_B/theta_g)^k_g ]
%                / Gamma(k_g+1)
%   ->  slope  -k_g  per decade of gamma_bar  (diversity order k_g)
%
%  NUMERICAL NOTE:  k_g from this geometry is huge (7.7e3 for N = 16), so
%  the waterfall is a near-vertical cliff (~0.1 dB wide) and (x)^(k_g)
%  overflows double precision.  The asymptote and deep-tail exact values
%  are therefore computed in the log10 domain.  A curve family over
%  N_list shows how diversity steepens the cliff.
%  ========================================================================
clear; clc; close all;

%% ------------------------- System Parameters ---------------------------
rho_OIRS = 0.9;   A_PD = 1e-4;   dA_i = 0.01;   m = 1;   T_gain = 1;
n_ref = 1.5;      Psi_FOV = deg2rad(80);
G_conc = n_ref^2 / (sin(Psi_FOV)^2);

%% ------------------------- Coordinates ---------------------------------
T = [2.5 2.5 3  ];      % Transmitter (AP)
P = [2.5 0   1.5];      % OIRS array position
R = [2.5 2.5 0  ];      % Receiver (UE)
d_ai = norm(P - T);     d_ik = norm(R - P);

PT = P - T;  OT = R - T;
cos_Phi_ai  = dot(PT, OT) / (norm(PT)*norm(OT));
cosm_Phi_ai = cos_Phi_ai^m;

alpha_rot = deg2rad(0);  beta_rot = deg2rad(0);
n0 = [0;1;0];
Rz = [cos(alpha_rot) -sin(alpha_rot) 0; sin(alpha_rot) cos(alpha_rot) 0; 0 0 1];
Rx = [1 0 0; 0 cos(beta_rot) sin(beta_rot); 0 sin(beta_rot) cos(beta_rot)];
nr = Rz * Rx * n0;
incident  = (T - P).';
cos_xi_ai = dot(nr, incident) / (norm(nr)*norm(incident));

I = (P - T).';  I = I / norm(I);
R_ref = I - 2*dot(I,nr)*nr;
n_rx  = [0;0;1];
cos_Phi_ik = dot(n_rx, R_ref) / (norm(n_rx)*norm(R_ref));

%% ------------------------- Channel Parameters --------------------------
M      = 1;
N_ref  = 16;                 % reference array size (as in original code)
N_list = [4 16 64];          % curve family: effect of diversity k_g ~ N
N_mc   = 1e5;

% Per-element deterministic gain (kept identical to original script)
% ak = rho_OIRS*((m+1)*A_PD)/(2*pi^2*d_ai^2*d_ik^2) * dA_i * ...
%      cosm_Phi_ai * cos_xi_ai * cos_Phi_ik * T_gain * G_conc;
ak = 1.0000e-6;

%% ------------------ Truncated-Laplace orientation model ----------------
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

% Gamma scale is INDEPENDENT of N;  shape grows linearly with N:
theta_g = ak * Var_xi / E_xi;               % theta_{g_k} = Var[Z]/E[Z]
kg_of_N = @(N) M .* N .* E_xi.^2 ./ Var_xi; % k_{g_k}     = E[Z]^2/Var[Z]

fprintf('E[xi] = %.5f   Var[xi] = %.4e   theta_g = %.4e\n', E_xi, Var_xi, theta_g);
for N = N_list
    fprintf('  N = %3d  ->  k_g = %10.2f,  E[Z] = %.4e\n', ...
            N, kg_of_N(N), M*N*ak*E_xi);
end
fprintf('\n');

%% ------------------- Outage definition & SNR sweep ---------------------
gamma_th = 1.0;      % SNR threshold (linear). e.g. 2^(2*Rs)-1 for target rate Rs
L_B      = 0;        % lower boundary of the Meijer-G formula (0 = classical outage)

% Sweep of average SNR gamma_bar_E in dB.  Each curve has its cliff at
% gamma_bar ~ gamma_th/E[Z_N]; make the window cover all N in N_list.
EZ_of_N       = @(N) M.*N.*ak.*E_xi;
gbar_cliff_dB = @(N) 10*log10(gamma_th ./ EZ_of_N(N));
gbar_lo  = gbar_cliff_dB(max(N_list)) - 1.5;
gbar_hi  = gbar_cliff_dB(min(N_list)) + 1.5;
gbar_dB  = linspace(gbar_lo, gbar_hi, 2400);
gbar_lin = 10.^(gbar_dB/10);

%% ------------------- Monte Carlo samples of xi (shared) ----------------
total_paths = N_mc * M * max(N_list);
xi_samples  = zeros(1, total_paths);
count = 0;
while count < total_paths
    batch = 2 * (total_paths - count);
    U = rand(1, batch) - 0.5;
    Ls = mu_xi - b_xi * sign(U) .* log(1 - 2*abs(U));
    ok = Ls(Ls >= xi_min & Ls <= xi_max);
    need = min(length(ok), total_paths - count);
    xi_samples(count+1 : count+need) = ok(1:need);
    count = count + need;
end

%% ============================ MAIN LOOP ================================
figure('Color','w','Position',[80 80 1150 520]);
cols = [0.16 0.47 0.84;  0.85 0.33 0.10;  0.20 0.62 0.30];

% ---- Panel (a): P_out vs gamma_bar_E [dB]  (observable range) ----------
subplot(1,2,1); hold on; grid on;
leg = {};
for in = 1:numel(N_list)
    N  = N_list(in);
    kg = kg_of_N(N);

    % exact Meijer-G ( = regularized lower incomplete gamma) outage
    z_th  = gamma_th ./ gbar_lin;                      % threshold on Z
    Pout  = gammainc(z_th./theta_g, kg) - gammainc(L_B/theta_g, kg);
    plot(gbar_dB, max(Pout,1e-12), '-', 'Color', cols(in,:), 'LineWidth', 2);
    leg{end+1} = sprintf('Exact, N = %d (k_g = %.0f)', N, kg); %#ok<SAGROW>

    % Monte Carlo (points clustered around this curve's own cliff)
    Zmc = ak * sum(reshape(xi_samples(1:N_mc*M*N), M*N, N_mc), 1);
    g_pts = gbar_cliff_dB(N) + linspace(-0.35, 0.15, 11);
    Pmc = arrayfun(@(gdB) mean( (10^(gdB/10)).*Zmc < gamma_th & ...
                                 Zmc >= L_B ), g_pts);
    msk = Pmc > 0;
    plot(g_pts(msk), Pmc(msk), 'o', 'Color', cols(in,:), ...
         'MarkerFaceColor', cols(in,:), 'MarkerSize', 5, ...
         'HandleVisibility','off');
end
set(gca,'YScale','log'); ylim([1e-6 1.5]); xlim([gbar_dB(1) gbar_dB(end)]);
xlabel('\gamma\_bar_E  (average SNR)  [dB]');
ylabel('P_{out}');
title(sprintf('(a) Outage vs average SNR   (\\gamma_{th} = %.2g, L_B = %g)', ...
      gamma_th, L_B));
legend(leg, 'Location', 'southwest');

% ---- Panel (b): deep tail in log10 domain + asymptote slope -k_g -------
subplot(1,2,2); hold on; grid on;
gbar_dB_w  = linspace(gbar_cliff_dB(max(N_list)) - 1, ...
                      gbar_cliff_dB(min(N_list)) + 20, 300);
gbar_lin_w = 10.^(gbar_dB_w/10);
leg2 = {};
for in = 1:numel(N_list)
    N  = N_list(in);
    kg = kg_of_N(N);
    xw = (gamma_th ./ gbar_lin_w) ./ theta_g;          % z_th/theta_g

    % exact log10 P_out : series for x < kg+1 (log-safe), gammainc otherwise
    l10P = zeros(size(xw));
    for jj = 1:numel(xw)
        x = xw(jj);
        if x >= kg + 1
            l10P(jj) = log10(gammainc(x, kg));
        else
            S = 1; term = 1; nn = 0;
            while true
                nn = nn + 1;  term = term * x/(kg+nn);  S = S + term;
                if term < eps*S || nn > 1e6, break; end
            end
            l10P(jj) = (kg*log(x) - x - gammaln(kg+1) + log(S)) / log(10);
        end
    end
    % asymptote:  log10 Pasym = [kg*ln(x) - gammaln(kg+1)]/ln(10)
    l10A = (kg*log(xw) - gammaln(kg+1)) / log(10);

    plot(gbar_dB_w, l10P, '-',  'Color', cols(in,:), 'LineWidth', 2);
    plot(gbar_dB_w, l10A, '--', 'Color', cols(in,:), 'LineWidth', 1.6, ...
         'HandleVisibility','off');
    leg2{end+1} = sprintf('N = %d: slope = -k_g/10 = %.0f /dB', ...
                          N, kg/10); %#ok<SAGROW>

    % numeric slope check (per decade of gamma_bar)
    p = polyfit(gbar_dB_w(150:280)/10, l10P(150:280), 1);
    fprintf('N = %3d : fitted tail slope = %.1f per decade (theory -k_g = %.1f)\n', ...
            N, p(1), -kg);
end
xlabel('\gamma\_bar_E  (average SNR)  [dB]');
ylabel('log_{10} P_{out}');
title('(b) Deep tail (log-domain): dashed = asymptote, slope -k_g');
legend(leg2, 'Location', 'southwest');

fprintf(['\nNote: the exact curve (solid) and the Meijer-G small-argument\n' ...
         'asymptote (dashed) merge at high SNR; the waterfall for N = 16 is\n' ...
         'only ~0.1 dB wide because k_g = %.0f acts as the diversity order.\n'], ...
         kg_of_N(N_ref));

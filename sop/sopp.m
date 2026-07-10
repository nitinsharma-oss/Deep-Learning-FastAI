function sop_analysis()
% =========================================================================
% SOP_ANALYSIS  Secrecy Outage Probability over the 3-region incomplete-
%               Beta piecewise fading model (Bob & Eve i.i.-parameterized).
%
%   PDF (per link L in {B,E}), support a_L <= g <= d_L, where
%       a = go*m1*m2,  b = go*m2*M1,  c = go*m1*M2,  d = go*M1*M2
%   Region 1 (a<=g<b):  f = K*[ B(g/(go*m2);p,1/2) - B(m1;p,1/2) ]
%   Region 2 (b<=g<c):  f = K*[ B(M1;p,1/2)       - B(m1;p,1/2) ]
%   Region 3 (c<=g<=d): f = K*[ B(M1;p,1/2) - B(g/(go*M2);p,1/2) ]
%   with B(x;a,b) the NON-regularized incomplete Beta function and the
%   DERIVED normalization constant
%       K = 1 / ( go*(M2-m2)*( B(M1;p+1,1/2)-B(m1;p+1,1/2) ) ).
%
%   Threshold:  tau(gE) = 2^(2*Rs)*gE + (3*pi*e/6)*(2^(2*Rs)-1)
%   For Rs = 0.5:  tau(gE) = 2*gE + pi*e/2.
%
%   SOP = int_0^{gE*} F_B(tau(g)) f_E(g) dg  +  [1 - F_E(gE*)],
%   gE* = ( d_B - (3*pi*e/6)*(2^(2Rs)-1) ) / 2^(2Rs).
%
%   F_B is the exact closed-form piecewise CDF (see cdf_link below),
%   built from the verified antiderivative identity
%       int B(x;p,1/2) dx = G(x) = x*B(x;p,1/2) - B(x;p+1,1/2).
%   The single remaining outer integral is evaluated by adaptive
%   quadrature with waypoints at every region breakpoint (the integrand
%   is exact/closed-form; only this 1-D quadrature is numerical, since
%   the cross terms J_{n,i} have no elementary primitive for general p).
%
%   Requires:  betainc, beta  (base MATLAB).  Octave-compatible.
% =========================================================================

    % ------------------ EDIT PARAMETERS HERE --------------------------
    Rs = 0.5;                                    % secrecy rate (b/s/Hz)
    %              go     p     m1    m2    M1    M2
    linkB = make_link(10.0, 1.5, 0.20, 2.00, 0.80, 9.0);   % Bob
    linkE = make_link( 6.0, 2.2, 0.15, 1.50, 0.70, 8.0);   % Eve
    % NOTE: model requires 0 < m1 < M1 <= 1 (Beta arguments in [0,1])
    %       and m2*M1 <= m1*M2 (Region-2 ordering b <= c).
    % -------------------------------------------------------------------

    % ---- self-checks (normalization + CDF endpoint) -------------------
    for L = [linkB, linkE]
        nrm = integral(@(g) pdf_link(g,L), L.a, L.d, ...
                       'Waypoints',[L.b L.c], 'AbsTol',1e-12,'RelTol',1e-12);
        fprintf('check: int pdf = %.12f (want 1),  F(d)=%.12f (want 1)\n', ...
                nrm, cdf_link(L.d, L));
    end

    % ---- SOP ----------------------------------------------------------
    [SOP, info] = compute_sop(linkB, linkE, Rs);
    fprintf('\ngamma_E^* = %.6f\n', info.gEstar);
    fprintf('outer-integral term  I = %.10f\n', info.I);
    fprintf('saturation term      S = %.10f\n', info.S);
    fprintf('SOP                    = %.10f\n', SOP);

    % ---- optional brute-force validation (double integral) ------------
    tau  = @(g) info.two2R.*g + info.cst;
    inner = @(gE) arrayfun(@(x) cdf_link(min(max(tau(x),linkB.a),linkB.d), linkB), gE);
    SOPbf = integral(@(g) inner(g).*pdf_link(g,linkE), linkE.a, linkE.d, ...
                     'Waypoints', info.brk_all, 'AbsTol',1e-12,'RelTol',1e-12);
    fprintf('brute-force check      = %.10f   (|diff| = %.2e)\n', ...
            SOPbf, abs(SOP-SOPbf));
end

% ========================================================================
function L = make_link(go, p, m1, m2, M1, M2)
% Build link parameter struct; compute breakpoints and closed-form K.
    assert(m1>0 && m1<M1 && M1<=1, 'need 0<m1<M1<=1');
    assert(m2>0 && m2<M2,          'need 0<m2<M2');
    L.go=go; L.p=p; L.m1=m1; L.m2=m2; L.M1=M1; L.M2=M2;
    L.a = go*m1*m2;  L.b = go*m2*M1;  L.c = go*m1*M2;  L.d = go*M1*M2;
    assert(L.a<L.b && L.b<=L.c && L.c<L.d, 'region ordering violated: need m2*M1 <= m1*M2');
    dB1  = Bx(M1,p+1) - Bx(m1,p+1);                 % Delta B_1
    L.K  = 1/( go*(M2-m2)*dB1 );                    % derived normalization
    L.B0m1 = Bx(m1,p);  L.B0M1 = Bx(M1,p);
    % continuity constants of the closed-form CDF:
    L.Fb = L.K*go*m2*( (M1*L.B0M1 - Bx(M1,p+1)) - (m1*L.B0m1 - Bx(m1,p+1)) ...
                       - L.B0m1*(M1-m1) );          % F(b) = K*go*m2*(M1*dB0 - dB1)
    L.Fc = L.Fb + L.K*(L.B0M1-L.B0m1)*(L.c-L.b);    % F(c)
end

function y = Bx(x, aa)
% Non-regularized incomplete Beta B(x; aa, 1/2), safe for x in [0,1].
    x = min(max(x,0),1);
    y = betainc(x, aa, 0.5) .* beta(aa, 0.5);
end

function G = Gfun(x, p)
% Antiderivative of B(x;p,1/2):  G(x) = x*B(x;p,1/2) - B(x;p+1,1/2).
    G = x.*Bx(x,p) - Bx(x,p+1);
end

function f = pdf_link(g, L)
% Piecewise PDF (vectorized).
    f  = zeros(size(g));
    r1 = g>=L.a & g<L.b;   r2 = g>=L.b & g<L.c;   r3 = g>=L.c & g<=L.d;
    f(r1) = L.K*( Bx(g(r1)/(L.go*L.m2), L.p) - L.B0m1 );
    f(r2) = L.K*( L.B0M1 - L.B0m1 );
    f(r3) = L.K*( L.B0M1 - Bx(g(r3)/(L.go*L.M2), L.p) );
end

function F = cdf_link(x, L)
% Exact closed-form piecewise CDF (vectorized), from the G identity:
%  R1: F = K*[ go*m2*( G(x/(go*m2)) - G(m1) ) - B0(m1)*(x-a) ]
%  R2: F = F(b) + K*dB0*(x-b)
%  R3: F = F(c) + K*[ B0(M1)*(x-c) - go*M2*( G(x/(go*M2)) - G(m1) ) ]
    F  = zeros(size(x));
    w1 = L.go*L.m2;  w3 = L.go*L.M2;
    r1 = x>=L.a & x<L.b;  r2 = x>=L.b & x<L.c;  r3 = x>=L.c & x<L.d;
    F(x>=L.d) = 1;
    F(r1) = L.K*( w1*(Gfun(x(r1)/w1,L.p)-Gfun(L.m1,L.p)) - L.B0m1*(x(r1)-L.a) );
    F(r2) = L.Fb + L.K*(L.B0M1-L.B0m1)*(x(r2)-L.b);
    F(r3) = L.Fc + L.K*( L.B0M1*(x(r3)-L.c) - w3*(Gfun(x(r3)/w3,L.p)-Gfun(L.m1,L.p)) );
end

function [SOP, info] = compute_sop(linkB, linkE, Rs)
% SOP = int_{Lo}^{Hi} F_B(tau(g)) f_E(g) dg + [1 - F_E(gE*)], with the
% outer integral restricted to where both factors are nonzero and split
% (via Waypoints) at every image of a region breakpoint.
    two2R = 2^(2*Rs);
    cst   = (3*pi*exp(1)/6)*(two2R - 1);        % = pi*e/2 for Rs = 0.5
    tau   = @(g) two2R.*g + cst;
    gEstar = (linkB.d - cst)/two2R;             % tau(gE*) = d_B

    info = struct('two2R',two2R,'cst',cst,'gEstar',gEstar);
    tb = (linkB.b - cst)/two2R;   tc = (linkB.c - cst)/two2R;
    info.brk_all = sort([tb tc linkE.b linkE.c gEstar]);
    info.brk_all = info.brk_all(info.brk_all>linkE.a & info.brk_all<linkE.d);

    if gEstar <= linkE.a                        % Bob's max below tau(a_E)
        info.I = 0; info.S = 1; SOP = 1; return
    end
    Lo = max([linkE.a, (linkB.a - cst)/two2R, 0]);
    Hi = min(linkE.d, gEstar);
    info.I = 0;
    if Hi > Lo
        wp = info.brk_all(info.brk_all>Lo & info.brk_all<Hi);
        info.I = integral(@(g) cdf_link(tau(g),linkB).*pdf_link(g,linkE), ...
                          Lo, Hi, 'Waypoints',wp,'AbsTol',1e-13,'RelTol',1e-12);
    end
    if gEstar >= linkE.d, info.S = 0;
    else,                 info.S = 1 - cdf_link(gEstar, linkE);
    end
    SOP = info.I + info.S;
end

% =========================================================================
% Kv_formula  Model-specified normalization constant K_v (Task 2 update):
%
%   K_v = sum_{i=1}^{3} sum_{l=0}^{N_i}  R*beta_{i+1}*alpha_i / (4*Psi*(m+2))
%         * nchoosek(N_i,l) * (-H2)^(N_i-l) * v^( -(m+3+l)/(m+2) )
%
%   q fields:  R, Psi, m, alpha (1x3), betap (1x3, = [beta_2 beta_3 beta_4]),
%              N (1x3 nonneg. integers), H2, v  — constants of link v.
%
%   NOTE: reconstructed from a flattened (copy-pasted) source formula;
%   verify grouping against your reference. Any valid K_v MUST satisfy the
%   normalization identity  K_v*go*(M2-m2)*[B(M1;p+1,1/2)-B(m1;p+1,1/2)]=1
%   (checked below), since the PDF must integrate to 1 on [a_v, d_v].
% =========================================================================
function K = Kv_formula(q)
    K = 0;
    for i = 1:3
        for l = 0:q.N(i)
            K = K + ( q.R * q.betap(i) * q.alpha(i) / (4*q.Psi*(q.m+2)) ) ...
                  * nchoosek(q.N(i), l) * (-q.H2)^(q.N(i)-l) ...
                  * q.v^( -(q.m+3+l)/(q.m+2) );
        end
    end
end

% Consistency check: compare a supplied Kv against the value forced by
% normalization for link struct L (fields go,p,m1,m2,M1,M2).
function check_Kv(Kv, L)
    Binc  = @(x,aa) betainc(min(max(x,0),1),aa,0.5).*beta(aa,0.5);
    Kreq  = 1/( L.go*(L.M2-L.m2)*( Binc(L.M1,L.p+1)-Binc(L.m1,L.p+1) ) );
    fprintf('Kv supplied = %.8g | Kv required by normalization = %.8g | rel.diff = %.2e\n', ...
            Kv, Kreq, abs(Kv-Kreq)/abs(Kreq));
    if abs(Kv-Kreq)/abs(Kreq) > 1e-6
        warning('Supplied Kv does not normalize the PDF; SOP would be biased. Check model constants.');
    end
end

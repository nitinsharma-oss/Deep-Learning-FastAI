%% SOP over generalized piecewise incomplete-Beta fading channels
%  Secrecy Outage Probability for a wiretap channel where gamma_B and
%  gamma_E each follow the 3-region incomplete-Beta piecewise PDF
%
%    Region 1 (a<=g<b):  f = K*( B(g/(go*m2); p,1/2) - B(m1; p,1/2) )
%    Region 2 (b<=g<c):  f = K*( B(M1; p,1/2)        - B(m1; p,1/2) )
%    Region 3 (c<=g<=d): f = K*( B(M1; p,1/2) - B(g/(go*M2); p,1/2) )
%
%  with a=go*m1*m2, b=go*m2*M1, c=go*m1*M2, d=go*M1*M2 (requires b<=c,
%  i.e. M2/m2 >= M1/m1, and m1,M1 in (0,1]).
%
%  Derived results used here (all verified numerically to <1e-9):
%    K   = 1/( go*(M2-m2)*(B(M1;p+1,1/2)-B(m1;p+1,1/2)) )
%    CDF piecewise via primitive G(x) = x*B(x;p,1/2) - B(x;p+1,1/2)
%    tau(gE) = 2*gE + pi*e/2                      (Rs = 0.5)
%    gE*     = (d_B - pi*e/2)/2
%    SOP     = int_{L}^{U} F_B(tau(g)) f_E(g) dg  +  1 - F_E(gE*)
%
%  Author: derivation + validation by Claude (Anthropic), 2026-07-09
% -------------------------------------------------------------------------
clear; clc;

%% ---------------- User parameters (edit freely) ------------------------
Rs = 0.5;                                  % target secrecy rate (bits/s/Hz)

% Bob's link
P.B = struct('go',30.0,'p',1.7,'m1',0.15,'m2',0.4,'M1',0.9,'M2',3.0);
% Eve's link
P.E = struct('go', 8.0,'p',2.3,'m1',0.20,'m2',0.5,'M1',0.8,'M2',2.5);

%% ---------------- Build both links -------------------------------------
Bob = build_link(P.B);
Eve = build_link(P.E);

fprintf('Bob breakpoints: a=%.4g  b=%.4g  c=%.4g  d=%.4g\n',Bob.a,Bob.b,Bob.c,Bob.d);
fprintf('Eve breakpoints: a=%.4g  b=%.4g  c=%.4g  d=%.4g\n',Eve.a,Eve.b,Eve.c,Eve.d);

% sanity: PDFs must integrate to 1 (validates K and parameter ordering)
fprintf('norm check Bob: %.10f | Eve: %.10f (both must be 1)\n', ...
    integral(Bob.pdf,Bob.a,Bob.d), integral(Eve.pdf,Eve.a,Eve.d));

%% ---------------- Threshold and critical point -------------------------
A  = 2^(2*Rs);                             % = 2 for Rs = 0.5
Cc = (3*pi*exp(1)/6)*(2^(2*Rs)-1);         % = pi*e/2 for Rs = 0.5
tau    = @(g) A.*g + Cc;
gEstar = (Bob.d - Cc)/A;
fprintf('tau(g) = %g*g + %.6f ,  gamma_E* = %.6f\n', A, Cc, gEstar);

%% ---------------- SOP via the validated split formula ------------------
% Term S (saturation):
if gEstar <= Eve.a
    SOP = 1.0;                             % Eve always strong enough
    fprintf('gE* <= a_E  ->  SOP = 1 exactly.\n');
else
    S = 1 - Eve.cdf(gEstar);               % closed form (=0 if gE*>=d_E)

    % Term I: integrate F_B(tau(g))*f_E(g) over the effective support,
    % partitioned at every breakpoint image so each piece is smooth.
    L = max([Eve.a, (Bob.a - Cc)/A, 0]);
    U = min(Eve.d, gEstar);
    I = 0.0;
    if U > L
        cuts = [(Bob.b-Cc)/A, (Bob.c-Cc)/A, Eve.b, Eve.c];
        cuts = cuts(cuts > L & cuts < U);
        pts  = unique([L, sort(cuts), U]);
        for k = 1:numel(pts)-1
            I = I + integral(@(g) Bob.cdf_v(tau(g)).*Eve.pdf(g), ...
                             pts(k), pts(k+1), 'AbsTol',1e-12,'RelTol',1e-10);
        end
    end
    SOP = I + S;
    fprintf('I (integral term) = %.10f\nS (saturation)    = %.10f\n', I, S);
end
fprintf('==> SOP = %.10f\n', SOP);

%% ---------------- Optional: brute-force verification --------------------
inner = @(gE) arrayfun(@(t) (t>=Bob.d) + (t>Bob.a && t<Bob.d)* ...
              integral(Bob.pdf,Bob.a,min(t,Bob.d)), tau(gE));
SOP_bf = integral(@(g) inner(g).*Eve.pdf(g), Eve.a, Eve.d, ...
                  'ArrayValued',true,'AbsTol',1e-10,'RelTol',1e-8);
fprintf('Brute-force SOP  = %.10f   (|diff| = %.3e)\n', SOP_bf, abs(SOP-SOP_bf));

%% ======================= local functions ================================
function L = build_link(q)
    % Breakpoints
    L.a = q.go*q.m1*q.m2;  L.b = q.go*q.m2*q.M1;
    L.c = q.go*q.m1*q.M2;  L.d = q.go*q.M1*q.M2;
    assert(L.b <= L.c + 1e-12, ...
        'Parameter ordering violated: need m1*M2 >= m2*M1 (Region 2 nonempty).');
    assert(q.m1>0 && q.M1<=1, 'Need 0 < m1 < M1 <= 1 (Beta arguments in [0,1]).');

    % Non-regularized incomplete Beta B(x;a,1/2) = betainc(x,a,0.5)*beta(a,0.5)
    Binc = @(x,aa) betainc(min(max(x,0),1),aa,0.5).*beta(aa,0.5);

    B0m1 = Binc(q.m1,q.p);  B0M1 = Binc(q.M1,q.p);
    dB1  = Binc(q.M1,q.p+1) - Binc(q.m1,q.p+1);

    % Normalization constant (derived; verified)
    L.K  = 1/( q.go*(q.M2-q.m2)*dB1 );

    % Vectorized PDF
    L.pdf = @(g) pdf_fun(g,q,L,Binc,B0m1,B0M1);

    % Primitive G(x) = x*B0(x) - B1(x)  =>  d/dx G = B0
    G  = @(x) x.*Binc(x,q.p) - Binc(x,q.p+1);
    Fb = L.K*( q.go*q.m2*(G(q.M1)-G(q.m1)) - B0m1*(L.b-L.a) );  % F at b
    Fc = Fb + L.K*(B0M1-B0m1)*(L.c-L.b);                         % F at c

    % Scalar CDF and vectorized wrapper
    L.cdf   = @(x) cdf_scalar(x,q,L,G,B0m1,B0M1,Fb,Fc);
    L.cdf_v = @(x) arrayfun(L.cdf,x);
end

function f = pdf_fun(g,q,L,Binc,B0m1,B0M1)
    f  = zeros(size(g));
    r1 = g>=L.a & g<L.b;   r2 = g>=L.b & g<L.c;   r3 = g>=L.c & g<=L.d;
    f(r1) = L.K*( Binc(g(r1)./(q.go*q.m2),q.p) - B0m1 );
    f(r2) = L.K*( B0M1 - B0m1 );
    f(r3) = L.K*( B0M1 - Binc(g(r3)./(q.go*q.M2),q.p) );
end

function F = cdf_scalar(x,q,L,G,B0m1,B0M1,Fb,Fc)
    if x < L.a,      F = 0;
    elseif x < L.b,  F = L.K*( q.go*q.m2*(G(x/(q.go*q.m2))-G(q.m1)) - B0m1*(x-L.a) );
    elseif x < L.c,  F = Fb + L.K*(B0M1-B0m1)*(x-L.b);
    elseif x < L.d,  F = Fc + L.K*( B0M1*(x-L.c) - q.go*q.M2*(G(x/(q.go*q.M2))-G(q.m1)) );
    else,            F = 1;
    end
end

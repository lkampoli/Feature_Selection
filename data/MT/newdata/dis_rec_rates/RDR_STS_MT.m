function RDR = RDR_STS_MT(sp_AB, sp_M, nABi, nA, nB, nM, T)
% function for calculation the relaxation rates of dissociation and
% recombination reactions 
% STS: state-to-state approach
% MT: Marrone-Treanor model

% input parameters: 
% sp_AB, sp_M are the species of AB and M particles
% nABi is vibrational distribution of AB molecules [m^-3]
% nA, nB, nM are the number densities of atoms A and B, and collision
% partner M [m^-3]
% T is the gas temperature [K]

% 21.04.2022
% Olga Kunova - kunova.olga@gmail.com

%% constants
h = 6.6261e-34; % Planck constant [J*sec]
k = 1.3807e-23; % Boltzmann constant [J/K]

%% load additional data
sp_AB = lower(sp_AB);
sp_M = lower(sp_M);

load([char(sp_AB), '_data.mat'], 'ei', 'm', 'D', 'sigma', 'theta_rot', ...
    'gi', 'arr_data')

mAB = m(1); mA = m(2); mB = m(3); % [kg]
gAB = gi(1); gA = gi(2); gB = gi(3);

AR = arr_data(sp_M);
CA = AR(1); bA = AR(2); Ea = AR(3);

%% rate coefficients 
% dissociation AB(i) + M -> A + B + M
% equilibrium rate coefficient [m^3/sec]
kd_eq = CA * T ^ bA * exp(-Ea / (k*T));

% parameter of model [K]
% U = Inf;
U = D/6;
% U = 3*T;

% vibr. partition functions
ZvT = sum(exp(-ei / (T*k)));
ZvU = sum(exp(ei / (U*k)));

% non-equilibrium factor
Z = ZvT / ZvU * exp(ei/k * (1/T + 1/U));

% state-depended rate coefficients [m^3/sec]
kd = kd_eq * Z;

% rotational partition function
Zr = T / (sigma * theta_rot);

% equilibrium ratio [m^3]
Kdr = gAB / (gA*gB) * (mAB * h^2 / (mA * mB * 2 * pi * k * T))^(3/2) * ...
    Zr * exp(-ei / (k*T)) * exp(D/T);

% recombination A + B + M -> AB(i) + M
% state-depended rate coefficients [m^6/sec]
kr = kd .* Kdr;

% state-depended dissociation/recombination rates [m^-3/sec]
RDR = nM * (nA * nB * kr - nABi .* kd);

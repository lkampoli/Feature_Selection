% Program for calling the calculation function of dissociation rates

% 21.04.2022
% Olga Kunova - kunova.olga@gmail.com
% Pressure Temperature X1X2X3X4X5X6X7X8X9X10X11X12X13X14X15X16X17X18X19X20X21X22X23X24X25X26X27X28X29X30X31X32X33X34X35X36X37X38X39X40X41X42X43X44X45X46X47X48X49 Viscosity
clear

%% constants
k = 1.3807e-23; % Boltzmann constant [J/K]

%% input parameters
% collision of species
sp_AB = 'N2'; 
sp_M = 'N';

sp_AB = lower(sp_AB);
load([char(sp_AB), '_data.mat'], 'ei');

% pressure [Pa]
p = 101325;

% temperature [K]
T = 1000;
Tv = 300;

% for i=0.1:0.5
% for j=0.1:0.5
% molar fraction of mixture components
x_AB = 0.1;
x_M = 0.3;
x_A = 0.1; %i;
x_B = 0.1; % j;

if isequal(sp_AB,'n2') || isequal(sp_AB,'o2')
    if x_A ~= x_B
        disp('Check xA and xB. Should be xA == xB!')
        return;
    end
end

% numbers densities of mixture components [m^-3]
n = p / (k*T);
nAB = x_AB * n;
nM = x_M * n;
nA = x_A * n;
nB = x_B * n;

% % Boltzmann distribution [m^-3]
% Zv = sum(exp(-ei / (k*Tv)));
% nABi = nAB / Zv * exp(-ei / (k*Tv));

% Treanor distribution [m^-3]
states = (0:length(ei)-1)';
Zv = sum(exp(-(ei-states*ei(2))/(k*T) - states*ei(2)/(k*Tv)));
nABi = nAB / Zv * exp(-(ei-states*ei(2))/(k*T) - states*ei(2)/(k*Tv));

%% output parameters
RDR = RDR_STS_MT(sp_AB, sp_M, nABi, nA, nB, nM, T);
% end
% end
clc

% === Extract variables ===
X = curves.X_mesh_tn;
Y = curves.X_mesh_jn;
Z = curves.pes;   % already log10(prob_sym_error_vec)

% === Plot ===
figure('Position',[100 100 800 600]);
contourf(X, Y, log10(Z), 100, 'LineStyle','none');  % filled contour with 100 levels
colormap('jet');
cbar = colorbar;
ylabel(cbar, 'P_{sym\_error} (log_{10})');
xlabel('SNR_{tn} (dB)');
ylabel('SNR_{jn} (dB)');
title('Symbol Error Probability (log_{10})');
grid on;
ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.3;
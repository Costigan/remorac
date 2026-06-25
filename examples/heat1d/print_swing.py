import numpy as np
data = np.loadtxt('h./eat1d_debug.csv', delimiter=',', skiprows=1)
time = data[:, 0]
T = data[:, 1:]
# Column indices: 0 = surface (z=0), -1 = bottom (z=0.658 m)
T_surf = T[:, 0]
T_bot = T[:, -1]
T_mid = T[:, 9]  # z ≈ 0.078 m

print(f'T_surf: min={T_surf.min():.2f}, max={T_surf.max():.2f}')
print(f'T_bot (z=0.658): min={T_bot.min():.2f}, max={T_bot.max():.2f}, swing={T_bot.max()-T_bot.min():.4f}')
print(f'T_mid (z=0.078): min={T_mid.min():.2f}, max={T_mid.max():.2f}, swing={T_mid.max()-T_mid.min():.4f}')

# Print swing at all depths
print()
print('z[m]    | T_min[K] | T_max[K] | swing[K]')
z_values = [0.0, 0.003, 0.0068, 0.0115, 0.0174, 0.0248, 0.034, 0.0455, 0.0599, 0.0779, 0.1003, 0.1284, 0.1636, 0.2075, 0.2624, 0.331, 0.4168, 0.524, 0.658]
for i in range(len(z_values)):
    Ti = T[:, i]
    print(f'{z_values[i]:<8.4f} | {Ti.min():9.4f} | {Ti.max():9.4f} | {Ti.max()-Ti.min():9.4f}')

"""
Plot sensor data from the CSV file and save visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create visualization folder if it doesn't exist
os.makedirs('visualisation', exist_ok=True)

# Load the data
data = pd.read_csv('DATA_[1,2V,1,4V;1,6;1.8]_TempSweep_2026-02-18_19-44-46.csv')

# Extract columns
time = data['time'].values
set_temp = data['set_temp'].values
act_temp = data['act_temp'].values
I_D1_1_2V = data['I_D1_1.2V'].values
I_D2_1_4V = data['I_D2_1.4V'].values
I_D3_1_6V = data['I_D3_1.6V'].values
I_D4_1_8V = data['I_D4_1.8V'].values

# Plot 1: Temperature sweep over time
plt.figure(figsize=(12, 6))
plt.plot(time, act_temp, 'b-', linewidth=2, label='Actual Temperature')
plt.plot(time, set_temp, 'r--', linewidth=2, label='Set Temperature')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Temperature [°C]', fontsize=12)
plt.title('Temperature Sweep Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('visualisation/01_temperature_sweep.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Sensor currents over time
plt.figure(figsize=(12, 6))
plt.plot(time, I_D1_1_2V * 1e9, 'r-', linewidth=1.5, label='D1 @ 1.2V', alpha=0.8)
plt.plot(time, I_D2_1_4V * 1e9, 'g-', linewidth=1.5, label='D2 @ 1.4V', alpha=0.8)
plt.plot(time, I_D3_1_6V * 1e9, 'b-', linewidth=1.5, label='D3 @ 1.6V', alpha=0.8)
plt.plot(time, I_D4_1_8V * 1e9, 'm-', linewidth=1.5, label='D4 @ 1.8V', alpha=0.8)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Current [nA]', fontsize=12)
plt.title('Sensor Currents Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('visualisation/02_sensor_currents_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Current vs Actual Temperature (scatter plot)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sensor Currents vs Temperature', fontsize=14, fontweight='bold')

colors = plt.cm.viridis(np.linspace(0, 1, len(act_temp)))

axes[0, 0].scatter(act_temp, I_D1_1_2V * 1e9, c=act_temp, cmap='viridis', s=20, alpha=0.7)
axes[0, 0].set_xlabel('Temperature [°C]', fontsize=11)
axes[0, 0].set_ylabel('Current [nA]', fontsize=11)
axes[0, 0].set_title('D1 @ 1.2V', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(act_temp, I_D2_1_4V * 1e9, c=act_temp, cmap='viridis', s=20, alpha=0.7)
axes[0, 1].set_xlabel('Temperature [°C]', fontsize=11)
axes[0, 1].set_ylabel('Current [nA]', fontsize=11)
axes[0, 1].set_title('D2 @ 1.4V', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(act_temp, I_D3_1_6V * 1e9, c=act_temp, cmap='viridis', s=20, alpha=0.7)
axes[1, 0].set_xlabel('Temperature [°C]', fontsize=11)
axes[1, 0].set_ylabel('Current [nA]', fontsize=11)
axes[1, 0].set_title('D3 @ 1.6V', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

scatter = axes[1, 1].scatter(act_temp, I_D4_1_8V * 1e9, c=act_temp, cmap='viridis', s=20, alpha=0.7)
axes[1, 1].set_xlabel('Temperature [°C]', fontsize=11)
axes[1, 1].set_ylabel('Current [nA]', fontsize=11)
axes[1, 1].set_title('D4 @ 1.8V', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Temperature [°C]')
plt.tight_layout()
plt.savefig('visualisation/03_current_vs_temperature.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: All sensors comparison (dual axis)
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.set_xlabel('Time [s]', fontsize=12)
ax1.set_ylabel('Sensor Currents [nA]', fontsize=12, color='black')
line1, = ax1.plot(time, I_D1_1_2V * 1e9, 'r-', linewidth=1.5, label='D1 @ 1.2V', alpha=0.8)
line2, = ax1.plot(time, I_D2_1_4V * 1e9, 'g-', linewidth=1.5, label='D2 @ 1.4V', alpha=0.8)
line3, = ax1.plot(time, I_D3_1_6V * 1e9, 'b-', linewidth=1.5, label='D3 @ 1.6V', alpha=0.8)
line4, = ax1.plot(time, I_D4_1_8V * 1e9, 'm-', linewidth=1.5, label='D4 @ 1.8V', alpha=0.8)
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('Temperature [°C]', fontsize=12, color='orange')
line5, = ax2.plot(time, act_temp, 'orange', linewidth=2.5, label='Temperature', linestyle='--')
ax2.tick_params(axis='y', labelcolor='orange')

lines = [line1, line2, line3, line4, line5]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=11)

plt.title('Sensor Currents and Temperature Over Time', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualisation/04_combined_currents_temperature.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Current ratios at different bias voltages
plt.figure(figsize=(12, 6))
current_ratio_1_4_vs_1_2 = I_D2_1_4V / I_D1_1_2V
current_ratio_1_6_vs_1_2 = I_D3_1_6V / I_D1_1_2V
current_ratio_1_8_vs_1_2 = I_D4_1_8V / I_D1_1_2V

plt.plot(time, current_ratio_1_4_vs_1_2, 'g-', linewidth=1.5, label='I(1.4V) / I(1.2V)', alpha=0.8)
plt.plot(time, current_ratio_1_6_vs_1_2, 'b-', linewidth=1.5, label='I(1.6V) / I(1.2V)', alpha=0.8)
plt.plot(time, current_ratio_1_8_vs_1_2, 'm-', linewidth=1.5, label='I(1.8V) / I(1.2V)', alpha=0.8)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Current Ratio', fontsize=12)
plt.title('Current Ratios at Different Bias Voltages', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('visualisation/05_current_ratios.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Temperature stability analysis
plt.figure(figsize=(12, 6))
temp_diff = act_temp - set_temp
plt.plot(time, temp_diff, 'purple', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Temperature Difference [°C]', fontsize=12)
plt.title('Temperature Difference (Actual - Set)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualisation/06_temperature_difference.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Visualizations saved to 'visualisation' folder:")
print("  - 01_temperature_sweep.png")
print("  - 02_sensor_currents_time.png")
print("  - 03_current_vs_temperature.png")
print("  - 04_combined_currents_temperature.png")
print("  - 05_current_ratios.png")
print("  - 06_temperature_difference.png")

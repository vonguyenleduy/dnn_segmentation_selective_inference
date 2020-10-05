import numpy as np
import matplotlib.pyplot as plt


line1 = [0.09, 0.31, 0.62, 0.79]
line2 = [0.04, 0.09, 0.22, 0.36]

index = ['0.5', '1.0', '1.5', '2.0']

xi = [1, 2, 3, 4]

plt.rcParams.update({'font.size': 18})

plt.title("Power")
plt.ylim(0, 1.03)

plt.plot(xi, line1, 'o-', label='proposed-method', linewidth=3)
plt.plot(xi, line2, 'o-', label='proposed-method-oc', linewidth=3)


plt.xticks([1, 2, 3, 4], index)
plt.xlabel("delta mu")
plt.ylabel("Power")
plt.legend()
plt.tight_layout()
plt.savefig('../results/power_plot.pdf')
plt.show()

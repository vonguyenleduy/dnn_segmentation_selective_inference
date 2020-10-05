import numpy as np
import matplotlib.pyplot as plt


line1 = [0.04, 0.04, 0.05, 0.04]
line2 = [0.04, 0.05, 0.05, 0.05]
line3 = [0.11, 0.33, 0.60, 0.77]

index = ['16', '64', '256', '1024']

xi = [1, 2, 3, 4]

plt.rcParams.update({'font.size': 17})

plt.title("False Positive Rate (FPR)")
plt.ylim(0, 1.03)

plt.plot(xi, line1, 'o-', label='proposed-method', linewidth=3)
plt.plot(xi, line2, 'o-', label='proposed-method-oc', linewidth=3)
plt.plot(xi, line3, 'o-', label='naive', linewidth=3)


plt.xticks([1, 2, 3, 4], index)
plt.xlabel("n")
plt.ylabel("FPR")
plt.legend()
plt.tight_layout()
plt.savefig('../results/fpr_plot.pdf')
plt.show()

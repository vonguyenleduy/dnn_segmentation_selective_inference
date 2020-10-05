import numpy as np
import matplotlib.pyplot as plt

# line1 = [0.06, 0.059, 0.06, 0.05]
# line2 = [0.11, 0.1, 0.1, 0.1]
#
# index = ['16', '64', '256', '1024']
#
# xi = [1, 2, 3, 4]
#
# # plt.rcParams.update({'font.size': 18})
# # plt.figure(figsize=(7, 4.5))
# plt.rcParams.update({'font.size': 18})
# plt.figure(figsize=(7, 5.2))
#
# plt.title("FPR (laplace distribution)")
# plt.ylim(0, 0.5)
#
# plt.plot(xi, line1, 'o-', label='alpha=0.05')
# plt.plot(xi, line2, 'o-', label='alpha=0.1')
#
# plt.xticks([1, 2, 3, 4], index)
# plt.xlabel("n")
# plt.ylabel("FPR")
# plt.legend()
# plt.tight_layout()
# plt.savefig('../results/fpr_laplace.pdf')
# plt.show()


# line1 = [0.05, 0.041, 0.05, 0.05]
# line2 = [0.1, 0.091, 0.09, 0.1]
#
# index = ['16', '64', '256', '1024']
#
# xi = [1, 2, 3, 4]
#
# # plt.rcParams.update({'font.size': 18})
# # plt.figure(figsize=(7, 4.5))
# plt.rcParams.update({'font.size': 18})
# plt.figure(figsize=(7, 5.2))
#
# plt.title("FPR (skew normal distribution)")
# plt.ylim(0, 0.5)
#
# plt.plot(xi, line1, 'o-', label='alpha=0.05')
# plt.plot(xi, line2, 'o-', label='alpha=0.1')
#
# plt.xticks([1, 2, 3, 4], index)
# plt.xlabel("n")
# plt.ylabel("FPR")
# plt.legend()
# plt.tight_layout()
# plt.savefig('../results/fpr_skew.pdf')
# plt.show()


# line1 = [0.05, 0.054, 0.04, 0.05]
# line2 = [0.1, 0.081, 0.09, 0.08]
#
# index = ['16', '64', '256', '1024']
#
# xi = [1, 2, 3, 4]
#
# # plt.rcParams.update({'font.size': 18})
# # plt.figure(figsize=(7, 4.5))
# plt.rcParams.update({'font.size': 18})
# plt.figure(figsize=(7, 5.2))
#
# plt.title("FPR (t20 distribution)")
# plt.ylim(0, 0.5)
#
# plt.plot(xi, line1, 'o-', label='alpha=0.05')
# plt.plot(xi, line2, 'o-', label='alpha=0.1')
#
# plt.xticks([1, 2, 3, 4], index)
# plt.xlabel("n")
# plt.ylabel("FPR")
# plt.legend()
# plt.tight_layout()
# plt.savefig('../results/fpr_t20.pdf')
# plt.show()


line1 = [0.05, 0.045, 0.05, .045]
line2 = [0.12, 0.1, 0.08, 0.1]

index = ['16', '64', '256', '1024']

xi = [1, 2, 3, 4]

# plt.rcParams.update({'font.size': 18})
# plt.figure(figsize=(7, 4.5))
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(7, 5.2))

plt.title("FPR (estimated sigma)")
plt.ylim(0, 0.5)

plt.plot(xi, line1, 'o-', label='alpha=0.05')
plt.plot(xi, line2, 'o-', label='alpha=0.1')

plt.xticks([1, 2, 3, 4], index)
plt.xlabel("n")
plt.ylabel("FPR")
plt.legend()
plt.tight_layout()
plt.savefig('../results/fpr_estimated_sigma.pdf')
plt.show()


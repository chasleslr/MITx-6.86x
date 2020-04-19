import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

with open('training_log.pkl', 'rb') as f:
    training_log = pkl.load(f)

n_epochs = len(training_log)

loss = np.zeros((n_epochs, 2))
val_loss = np.zeros((n_epochs, 2))
acc = np.zeros((n_epochs, 2))
val_acc = np.zeros((n_epochs, 2))

for i, epoch in enumerate(training_log):
    for j in range(2):
        loss[i, j] = epoch[0][j]
        val_loss[i, j] = epoch[1][j]
        acc[i, j] = epoch[2][j]
        val_acc[i, j] = epoch[3][j]

plt.style.use('ggplot')

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 6))
ax = ax.flatten()

ax[0].plot(loss)
ax[1].plot(val_loss)
ax[2].plot(acc)
ax[3].plot(val_acc)

ax[0].set_title("Training Loss")
ax[1].set_title("Validation Loss")
ax[2].set_title("Training Accuracy")
ax[3].set_title("Validation Accuracy")

ax[0].set_ylim((0, np.max(loss)))
ax[2].set_ylim((np.min(acc), 1))

plt.show();

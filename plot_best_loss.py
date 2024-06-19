import json
import matplotlib.pyplot as plt

with open('data/losess_best_model.json') as f:
  data = json.load(f)

epochs = range(1, len(data['losess_train']) + 1)
losses_train = data['losess_train']
losses_val = data['losess_val']
losses_test = data['losess_test']
lr = data['lr']
n_hidden = data['n_hidden']
wd = data['wd']

plt.plot(epochs, losses_train, 'b', label='Training loss')
plt.plot(epochs, losses_val, 'r', label='Validation loss')
plt.plot(epochs, losses_test, 'g', label='Test loss')
plt.title(f"Loss with {n_hidden} hidden units, lr={lr}, wd={wd}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_best_model.png')
plt.show()

# just val and test
plt.plot(epochs, losses_val, 'r', label='Validation loss')
plt.plot(epochs, losses_test, 'g', label='Test loss')
plt.title(f"Loss with {n_hidden} hidden units, lr={lr}, wd={wd}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_best_model_val_test.png')
plt.show()

#just val
plt.plot(epochs, losses_val, 'r', label='Validation loss')
plt.title(f"Loss with {n_hidden} hidden units, lr={lr}, wd={wd}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_best_model_val.png')
plt.show()

#just test
plt.plot(epochs, losses_test, 'g', label='Test loss')
plt.title(f"Loss with {n_hidden} hidden units, lr={lr}, wd={wd}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_best_model_test.png')
plt.show()
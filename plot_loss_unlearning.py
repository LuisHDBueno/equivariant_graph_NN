import json
import matplotlib.pyplot as plt

with open('data/losess_lr_0.0001_n_hidden_128_wd_1e-08.json') as f:
  data = json.load(f)

epochs = range(1, len(data['losess_train']) + 1)
losses_train = data['losess_train']
losses_val = data['losess_val']
lr = data['lr']
n_hidden = data['n_hidden']
wd = data['wd']

plt.plot(epochs, losses_train, 'b', label='Training loss')
plt.plot(epochs, losses_val, 'r', label='Validation loss')
plt.title(f"Loss with {n_hidden} hidden units, lr={lr}, wd={wd}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_unlearning_model.png')
plt.show()
import json
import matplotlib.pyplot as plt

with open('data/losess_lr0.001.json') as f:
  data = json.load(f)

plt.plot(data['epochs'], data['losess'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Epochs')
plt.show()

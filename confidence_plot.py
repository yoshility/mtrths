# 各tokenのconfidence scoreをプロットして、パターンを探す

import matplotlib.pyplot as plt
import shelve
import numpy as np

# TODO: shelveだと遅いのでzarrを使おう
raw_data = shelve.open("/data/yoshie/mtrths/raw_data_llama3_gsm8k")
print(f"len(raw_data): {len(raw_data)}")

fig, ax = plt.subplots(50, 1, figsize=(100, 100))
fig.suptitle('confidence scores of each trace')

for i in range(50):
    entropy = raw_data[str(i)]['entropy'].flatten() * (-1)
    token_id = np.arange(0, len(entropy), 1)
    color = 'green' if raw_data[str(i)]['is_correct'] else 'red'
    ax[i].set_xlabel('token id')
    ax[i].set_ylabel('confidence (-entropy)')
    ax[i].set_xlim(0, 270)
    ax[i].plot(token_id, entropy, 'o-', color=color)

plt.savefig('confidence_plot.png')
raw_data.close()
print(f"finished")
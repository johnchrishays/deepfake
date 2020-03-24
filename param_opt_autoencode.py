from matplotlib import rcParams
import matplotlib.pyplot as plt
import datetime
from models import Autoencoder
from datasets import DeepfakeDataset
import train_autoencoder
import test_autoencoder

rcParams.update({
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        'figure.figsize': (7.5,5),
})
def file_str():
    """ Auto-generates file name."""
    now = datetime.datetime.now()
    return now.strftime("H%HM%MS%S_%f_%m-%d-%y")
def pdf_savefig():
    fname = file_str()
    plt.savefig(f"./figs/{fname}.pdf")
    plt.close()

EPOCH_SIZE = 100
offset_range = range(0, 9, 2)

train_errs = list()
train_times = list()
test_errs = list()
test_times = list()
hidden_dims = list()

for offset in offset_range:
    train_err, train_exec_time = train_autoencoder.train_autoencoder(n_out_channels1=8+offset, n_out_channels2=8+offset, n_out_channels3=4+offset, kernel_size=5, epoch_size=EPOCH_SIZE)
    train_errs.append(train_err)
    train_times.append(train_exec_time)
    test_err, test_exec_time, hidden_dim = test_autoencoder.test_autoencoder(n_out_channels1=8+offset, n_out_channels2=8+offset, n_out_channels3=4+offset, kernel_size=5)
    test_errs.append(test_err)
    test_times.append(test_exec_time)
    hidden_dims.append(hidden_dim)

# NUM CHANNELS: ERR
plt.plot(list(offset_range), train_errs, label=f"Train")
plt.plot(list(offset_range), test_errs, label=f"Test")
plt.title(f"Train and test error vs. number of channels for epoch size {EPOCH_SIZE}")
plt.legend(loc=0)
plt.xlabel("Offset. (Num channels is {Layer1: 8+offset, Layer2: 8+offset, Layer3: 4+offset}).")
plt.ylabel("Mean squared error")
pdf_savefig()

# NUM CHANNELS: TIMES
plt.plot(list(offset_range), [train_time.total_seconds() for train_time in train_times], label=f"Train")
plt.title(f"Execution time vs. number of channels for epoch size {EPOCH_SIZE}")
plt.legend(loc=0)
plt.xlabel("Offset. (Num channels is {Layer1: 8+offset, Layer2: 8+offset, Layer3: 4+offset}).")
plt.ylabel("Execution time (s)")
pdf_savefig()

# DIM REDUCTION
plt.plot(list(offset_range), [hidden_dim/(3*1920*1080) for hidden_dim in hidden_dims])
plt.title(f"Dimension vs. number of channels for epoch size {EPOCH_SIZE}")
plt.xlabel("Number of channels. Layer1: 8+offset, Layer2: 8+offset, Layer3: 4+offset.")
plt.ylabel("Hidden dimension / original dimension")
pdf_savefig()

# LENGTH OF EPOCHS
epoch_range = range(20, 101, 20)

train_errs = list()
train_times = list()
test_errs = list()
test_times = list()

for epoch in epoch_range:
    train_err, train_exec_time = train_autoencoder.train_autoencoder(n_out_channels1=16, n_out_channels2=16, n_out_channels3=8, kernel_size=5, epoch_size=epoch)
    train_errs.append(train_err)
    train_times.append(train_exec_time)
    test_err, test_exec_time, hidden_dim = test_autoencoder.test_autoencoder(n_out_channels1=16, n_out_channels2=16, n_out_channels3=8, kernel_size=5)
    test_errs.append(test_err)
    test_times.append(test_exec_time)

# ERR
plt.plot(list(epoch_range), train_errs, label=f"Train")
plt.plot(list(epoch_range), test_errs, label=f"Test")
plt.title(f"Train and test error vs. epoch size")
plt.legend(loc=0)
plt.xlabel("Epoch size")
plt.ylabel("Mean squared error")
pdf_savefig()

# EXECUTION SIZE
plt.plot(list(epoch_range), [train_time.total_seconds() for train_time in train_times], label=f"Train")
plt.title(f"Execution time vs. epoch size")
plt.legend(loc=0)
plt.xlabel("Epoch size")
plt.ylabel("Execution time (s)")
pdf_savefig()

end_time = datetime.datetime.now()
exec_time = end_time - start_time
print(f"executed in: {str(exec_time)}, finished {str(end_time)}")



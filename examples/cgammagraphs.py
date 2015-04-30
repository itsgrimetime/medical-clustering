import numpy as np
import pylab as pl

ppcs = [(i, i) for i in range(2, 28, 2)]
cpbs = [(i, i) for i in range(1, 5)]

all_train_err = np.load('all_train_errs.npy')
all_train_err = 100.0 - all_train_err
all_val_err = np.load('all_val_errs.npy')
all_val_err = 100.0 - all_val_err

all_train_err = np.asarray(all_train_err).reshape(len(ppcs), len(cpbs))

print all_train_err[4][2]

pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
pl.imshow(all_train_err, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('Cells Per Block')
pl.ylabel('Pixels Per Cell')
pl.colorbar()
pl.xticks(np.arange(len(cpbs)), cpbs, rotation=45)
pl.yticks(np.arange(len(ppcs)), ppcs)
pl.show()

all_val_err = np.asarray(all_val_err).reshape(len(ppcs), len(cpbs))

print all_val_err[4][2]

pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
pl.imshow(all_val_err, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('Cells Per Block')
pl.ylabel('Pixels Per Cell')
pl.xticks(np.arange(len(cpbs)), cpbs, rotation=45)
pl.yticks(np.arange(len(ppcs)), ppcs)
pl.colorbar()
pl.show()

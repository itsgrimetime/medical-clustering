all_train_err = np.asarray(all_train_err).reshape(len(cs), len(gammas))
pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
pl.imshow(all_train_err, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('gamma')
pl.ylabel('C')
pl.colorbar()
pl.xticks(np.arange(len(gammas)), gammas, rotation=45)
pl.yticks(np.arange(len(cs)), cs)
pl.show()

all_val_err = np.asarray(all_val_err).reshape(len(cs), len(gammas))
pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
pl.imshow(all_val_err, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('gamma')
pl.ylabel('C')
pl.colorbar()
pl.xticks(np.arange(len(gammas)), gammas, rotation=45)
pl.yticks(np.arange(len(cs)), cs)
pl.show()

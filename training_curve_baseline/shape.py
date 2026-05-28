# import numpy as np, os
# p = '/home2/twu/SDDAL_best_version/SDDAL_InShaPe_randinit_syncUNet_ReparamCKL/Design_rec_6_123'
# tr = sorted(os.listdir(p+'/training_set/intensity/npy/'))[0]
# te = sorted(os.listdir(p+'/test_set/intensity/npy/'))[0]
# print('train:', np.load(p+'/training_set/intensity/npy/'+tr).shape)
# print('test: ', np.load(p+'/test_set/intensity/npy/'+te).shape)

import numpy as np, os
p = '/home2/twu/SDDAL_best_version/SDDAL_InShaPe_randinit_syncUNet_ReparamCKL/Design_rec_6_123/test_set/intensity/npy/'
bad = []
for f in sorted(os.listdir(p)):
  try:
      a = np.load(os.path.join(p, f))
      if a.shape != (427, 427):
          bad.append((f, a.shape))
  except Exception as e:
      bad.append((f, str(e)))
print(f'Checked {len(os.listdir(p))} files')
print('Bad files:', bad if bad else 'none')

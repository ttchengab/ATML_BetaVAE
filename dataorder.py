"""
Rename data files so they can be loaded in correct order
"""

import os
from natsort import natsorted

os.chdir('data/3dChairs_rotaion/images')

pics = os.listdir()



for i, pic in enumerate(natsorted(pics)):
    if pic == '.DS_Store':
        break
    print(f'pic = {pic}')
    idx_ = pic.replace('_', '', 1).index('_')+1
    real_idx = int(pic[idx_+1:idx_+4])
    if real_idx > 30:
        real_idx -= 1
    new_name = str(real_idx+62*int(i/62))+'.png'
    length = len(new_name)
    new_name = '0'*(12-length)+new_name
    # print(f'new name = {new_name}\n')
    os.rename(pic, new_name)











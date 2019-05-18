import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    entries = []
    tag = '_label.npy'
    for dirpath, _, filenames in os.walk('/local/Data/kitti_raw_formatted'):
        for filename in filenames:
            if len(filename) >= len(tag) and filename[-len(tag):] == tag:
                entries.append(os.path.join(dirpath, filename))

    for i, each in enumerate(entries):
        print('{}/{}'.format(i, len(entries)))
        seg = np.load(each)

        tgt = each.replace(tag, '_label.npy')
        print('saving to : {}'.format(tgt))
        np.save(tgt, seg.astype(np.uint8))
        # plt.imsave(tgt, seg[1, ...])
        # plt.imshow(seg[1, ...])
        # plt.show()

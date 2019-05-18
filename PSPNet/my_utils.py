import numpy as np
import matplotlib.pyplot as plt
import os
import glob

groups = {
    'flat': [0, 1],
    'human': [11, 12],
    'vehicle': [13, 14, 15, 16, 17, 18],
    'construction': [2, 3, 4],
    'object': [5, 6, 7],
    'nature': [8, 9],
    'sky': [10]
}

def grab_all_entries(dataroot, filetype='.png'):
    """ Grab absolute path of all images.
    Args:
        dataroot: Root directory of raw kitti dataset.
    Returns:
        a list of absolute paths
    """

    def expand_date_folder(date_folder_path):
        entries = []
        for seq_folder in os.listdir(date_folder_path):
            seq_folder_path = os.path.join(date_folder_path, seq_folder)
            if os.path.isdir(seq_folder_path):
                entries += glob.glob(
                    os.path.join(seq_folder_path, 'image_02', 'data', '*'+filetype))
                entries += glob.glob(
                    os.path.join(seq_folder_path, 'image_03', 'data', '*'+filetype))
        return entries

    entries = []
    for date_folder in os.listdir(dataroot):
        date_folder_path = os.path.join(dataroot, date_folder)
        if os.path.isdir(date_folder_path):
            entries += expand_date_folder(date_folder_path)
    return np.array(entries)


def batch_group_log_prob_map(lp):
    """ batch mode of function group_log_prob_map
    Args:
        lp: batched log probability map of shape BHW(19)
    Returns:
        merged log probability map of shape BHW7
    """
    assert len(lp.shape) == 4
    B, H, W, _ = lp.shape
    out = np.zeros([B, H, W, len(groups)])
    for i in range(lp.shape[0]):
        out[i] = group_log_prob_map(lp[i])
    return out


def group_log_prob_map(lp):
    """ Group log probability maps according to the following rule:
        0 flat: 0 (road), 1 (sidewalk)
        1 human: 12 (rider) 11 (person)
        2 vehicle: 13 (car) 14 (truck) 15 (bus) 16 (train) 17 (motorcycle) 18 (bicycle)
        3 construction: 2 (building) 3 (wall) 4 (fence)
        4 object: 5 (pole) 6 (traffic light) 7 (traffic sign)
        5 nature: 8 (vegetation) 9 (terrain)
        6 sky: 10 (sky)
    Args:
        lp: log probability map of shape HW(19)
    Returns:
        merged log probability map of shape HW7
    """
    assert len(lp.shape) == 3
    p = np.exp(lp)
    sum_p = np.sum(p, axis=2)
    p = p / sum_p[..., np.newaxis]
    out = np.zeros([p.shape[0], p.shape[1], len(groups)])
    for i, key in enumerate([
            'flat', 'human', 'vehicle', 'construction', 'object', 'nature',
            'sky'
    ]):
        for j in groups[key]:
            out[..., i] += p[..., j]
    return np.log(out)


if __name__ == '__main__':
    seg_files = grab_all_entries('/local/Data/NIPS18/kitti_raw_data_seg/', '.npy')
    output_dir = '/local/Data/NPIS18/kitti_raw_data/'

    for file_idx, path in enumerate(seg_files):
        print('{:6}/{:6}'.format(file_idx, len(seg_files)))
        p = group_log_prob_map(np.load(path))
        dest_path = path[:-4] + '_compact.npy'
        print('writing to {}'.format(dest_path))
        np.save(dest_path, p.astype(np.float16))

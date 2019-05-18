from inference import grab_all_entries
import os

if __name__ == '__main__':
    mask_files = grab_all_entries('/local/Data/NIPS18/kitti_stereo_video/', '*.npy')
    output_dir = '/local/Data/NPIS18/kitti_stereo_video_mask/'

    for i, path in enumerate(mask_files):
        print('{}/{}'.format(i, len(mask_files)))
        dest_path = path.replace('kitti_stereo_video', 'kitti_stereo_video_mask')
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(os.path.dirname(dest_path))
        print('moving {} to {}'.format(path, dest_path))
        os.rename(path, dest_path)


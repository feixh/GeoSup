import numpy as np
import cv2
from kitti_eval import depth_evaluation_utils as eval_utils
import os

def log(s, filepath=None, to_console=True):
    if to_console:
        print(s)

    if filepath is not None:
        try:
            with open(filepath, "a+") as o:
                o.write(s+'\n')
        except:
            raise IOError


class Validator(object):
    def __init__(self, val_dir, val_list, match_scale=False, log_path=None, key_metric='abs_rel', which_camera='imx'):
        """
        Args:
            val_dist: Root directory of the validation data.
            val_list: A list of filenames used for validation.
            match_scale: If trained with monocular video, need to set true to match scale of gt and pred.
            log_path: File to log results.
            key_metric: Keep best checkpoint according to the key metric.
        """
        which_camera = str.lower(which_camera)
        if which_camera not in ['imx', 'realsense', 'rs']:
            raise ValueError('which_camera = [imx|realsense|rs]')
        if which_camera in ['realsense', 'rs']:
            self.depth_height, self.depth_width = 480, 640
        else:
            self.depth_height, self.depth_width = 250, 480

        val_files = eval_utils.read_text_lines(val_list)
        filenames = [os.path.join(val_dir, p) for p in val_files]

        self.gt = []
        self.masks = []
        # load depth
        self.filenames = []
        for image_file in filenames:
            if which_camera == 'imx':
                depth_file = image_file[:-4] + '_depth.npy'
            else:
                depth_file = image_file.replace('rgb', 'depth')[:-4] + '.npy'
            depth = np.load(depth_file)

            mask = np.logical_and(depth < 5, depth > 0)
            if np.any(mask):
                self.masks.append(mask)
                self.gt.append(depth)
                self.filenames.append(image_file)

        print('{} effective ground truth loaded'.format(len(self.gt)))
        self.best_value = 10000
        self.best_step = -1
        self.log_path = log_path
        self.match_scale = match_scale
        if key_metric in ['log_rms', 'abs_rel']:
            self.key_metric = key_metric
        else: raise ValueError('key_metric =[log_rms|abs_rel]')

    def validate(self, preds, step, min_depth=1e-3, max_depth=5, mode='depth'):
        """ Compute error and accuracy metrics.
        Args:
            preds: prediction in the form of an NxHxWx1 numpy array.
            min_depth, max_depth: Min & Max depth to cap.
            mode: mode=[depth, invdepth, disparity]
        Returns:
            whether or not the current prediction is better than the previous best result.
        """
        assert mode in ['depth', 'invdepth'] # , 'invdepth', 'disparity']

        # create alias
        gt = self.gt
        masks = self.masks

        n_sample = len(gt)
        pred_depths = []
        for i, pred in enumerate(preds):
            disp_pred = cv2.resize(pred, (self.depth_width, self.depth_height), interpolation=cv2.INTER_LINEAR)
            if mode == 'invdepth':
                depth_pred = 1.0 / disp_pred
            elif mode == 'depth':
                depth_pred = disp_pred
            else: raise ValueError('unrecognized mode')

            depth_pred[np.isinf(depth_pred)] = 0
            pred_depths.append(depth_pred)

        rms     = np.zeros(n_sample, np.float32)
        log_rms = np.zeros(n_sample, np.float32)
        abs_rel = np.zeros(n_sample, np.float32)
        sq_rel  = np.zeros(n_sample, np.float32)
        d1_all  = np.zeros(n_sample, np.float32)
        a1      = np.zeros(n_sample, np.float32)
        a2      = np.zeros(n_sample, np.float32)
        a3      = np.zeros(n_sample, np.float32)

        for i in range(n_sample):
            gt_depth = gt[i]
            mask = masks[i]
            pred_depth = pred_depths[i]

            # Scale matching
            if self.match_scale:
                scalar = np.median(gt_depth[mask])/np.median(pred_depth[mask])
                pred_depth *= scalar
                # print('matching scale={}'.format(scalar))

            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth

            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = eval_utils.compute_errors(gt_depth[mask], pred_depth[mask])

        # pick metric to check
        if self.key_metric == 'log_rms':
            curr_value = log_rms.mean()
        elif self.key_metric == 'abs_rel':
            curr_value = abs_rel.mean()

        better_than_best = False
        if self.best_value > curr_value:
            self.best_step = step
            self.best_value = curr_value
            better_than_best = True

        log('Current(step)=%d Local Minima(step)=%d  AbsRel=%.4f  '
                % (step, self.best_step, self.best_value), self.log_path)
        log('abs_rel    sq_rel    rms    log_rms    d1_all    a1    a2    a3', self.log_path)
        log('%.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    '
                % (abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(),
                    d1_all.mean(), a1.mean(), a2.mean(), a3.mean()), self.log_path)

        return better_than_best


if __name__ == '__main__':

    val_dir = '/home/feixh/Data/VISMA_depth_stride5'
    val_list = 'data/visma/test.txt'

    validator = Validator(val_dir=val_dir, val_list=val_list, match_scale=True)
    dummy = np.random.random((100, 250, 480, 1))

    for i, filename in enumerate(validator.filenames):
        dummy[i, ..., 0] = np.load(filename[:-4]+'_depth.npy')

    validator.validate(dummy, 0)


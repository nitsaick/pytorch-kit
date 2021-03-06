import numpy as np

from utils.switch import switch


class Evaluator:
    def __init__(self, num_classes):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_classes = num_classes
        self.reset()

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def mean_intersection_over_union(self):
        mIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        mIoU = np.nanmean(mIoU)
        return mIoU

    def frequency_weighted_intersection_over_union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwIoU

    def dice_coef(self, class_, confusion_matrix):
        dc = confusion_matrix[class_][class_] * 2 / (
                np.sum(confusion_matrix, axis=0)[class_] + np.sum(confusion_matrix, axis=1)[class_])
        if np.isnan(dc):
            dc = 0
        return dc

    def _generate_matrix(self, pred, label):
        mask = (label >= 0) & (label < self.num_classes)
        label = self.num_classes * label[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, preds, labels):
        assert preds.shape == labels.shape
        self.num += len(preds)
        for i in range(len(preds)):
            matrix = self._generate_matrix(preds[i], labels[i])
            self.confusion_matrix += matrix
            for class_ in range(self.num_classes):
                self.dc_acc[class_] += self.dice_coef(class_, matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.dc_acc = np.zeros(self.num_classes)
        self.num = 0

    def eval(self, func, class_=1):
        for case in switch(func):
            if case('dc'):
                acc = (self.dc_acc / self.num)[class_]
                break
            if case('pixel_accuracy'):
                acc = self.pixel_accuracy()
                break
            if case('pixel_accuracy_class'):
                acc = self.pixel_accuracy_class()
                break
            if case('mIoU'):
                acc = self.mean_intersection_over_union()
                break
            if case('fwIoU'):
                acc = self.frequency_weighted_intersection_over_union()
                break
            if case():
                raise NotImplementedError('Unknown evaluation function.')

        return acc

    def log_acc(self, logger, epoch, prefix=''):
        logger.add_scalar(prefix + 'dc', (self.dc_acc / self.num)[1], epoch)
        logger.add_scalar(prefix + 'pixel_accuracy', self.pixel_accuracy(), epoch)
        logger.add_scalar(prefix + 'pixel_accuracy_class', self.pixel_accuracy_class(), epoch)
        logger.add_scalar(prefix + 'mIoU', self.mean_intersection_over_union(), epoch)
        logger.add_scalar(prefix + 'fwIoU', self.frequency_weighted_intersection_over_union(), epoch)


if __name__ == '__main__':
    evaluator = Evaluator(5)
    gt_image = np.zeros((2, 5), dtype=int)
    pre_image = np.zeros((2, 5), dtype=int)
    gt_image[0][0] = 0
    gt_image[0][1] = 1
    gt_image[0][2] = 0
    gt_image[0][3] = 1
    gt_image[0][4] = 0

    pre_image[0][0] = 1
    pre_image[0][1] = 1
    pre_image[0][2] = 1
    pre_image[0][3] = 0
    pre_image[0][4] = 0

    gt_image[1][0] = 0
    gt_image[1][1] = 1
    gt_image[1][2] = 0
    gt_image[1][3] = 0
    gt_image[1][4] = 0

    pre_image[1][0] = 1
    pre_image[1][1] = 1
    pre_image[1][2] = 1
    pre_image[1][3] = 0
    pre_image[1][4] = 0

    evaluator.add_batch(gt_image, pre_image)

    print(evaluator.eval('dc'))

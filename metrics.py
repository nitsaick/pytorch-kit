import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, predicts, labels):
        assert labels.shape == predicts.shape
        self.confusion_matrix += self._generate_matrix(labels, predicts)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

if __name__ == '__main__':
    evaluator = Evaluator(3)
    gt_image = np.zeros(5, dtype=int)
    pre_image = np.zeros(5, dtype=int)
    gt_image[0] = 0
    gt_image[1] = 1
    gt_image[2] = 2
    gt_image[3] = 2
    gt_image[4] = 3

    pre_image[0] = 1
    pre_image[1] = 1
    pre_image[2] = 2
    pre_image[3] = 0
    pre_image[4] = 0

    evaluator.add_batch(gt_image, pre_image)

    print(evaluator.Mean_Intersection_over_Union())
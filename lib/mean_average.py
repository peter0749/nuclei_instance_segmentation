import config as conf
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def IOU(x, y):
    x_f = x.ravel()
    y_f = y.ravel()
    return np.sum(np.logical_and(x_f,y_f)) / np.sum(np.logical_or(x_f,y_f))

def score_single_img(masks, labels, iou_t):
    # ap = TP / (TP+FP)
    TP = 0
    matched = np.zeros(len(masks), dtype=np.bool)
    for l in range(1, labels.max()+1):
        pred_mask = (labels==l).astype(np.bool)
        for i, mask in enumerate(masks):
            if matched[i]:
                continue
            iou = IOU(mask, pred_mask)
            if iou<iou_t:
                continue
            matched[i] = True
            TP += 1
    return TP, len(masks), labels.max()

def average_precision_recall(masks, labels, iou_t=0.5):
    assert len(masks)==len(labels)
    pool = mp.Pool(processes=conf.GENERATOR_WORKERS)
    scores = [pool.apply_async(score_single_img, (mask, label, iou_t)) for mask, label in zip(masks, labels)]
    TP, true_mask_n, pred_mask_n = 0,0,0
    for score in scores:
        hit, t, p = score.get()
        TP += hit
        true_mask_n += t
        pred_mask_n += p

    pool.close()
    pool.join()
    ap = TP / pred_mask_n
    ar = TP / true_mask_n
    return ap, ar

def average_scores(masks, labels, thresholds=np.arange(0.5,1.0,0.05)):
    assert len(masks)==len(labels)
    aps = []
    ars = []
    for t in tqdm(thresholds, total=len(thresholds), ascii=True):
        ap, ar = average_precision_recall(masks, labels, t)
        aps.append(ap)
        ars.append(ar)
    return aps, ars


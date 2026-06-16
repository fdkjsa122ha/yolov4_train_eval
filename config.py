from init import *
from easydict import EasyDict
Cfg = EasyDict()

# info
Cfg.track = 0
Cfg.w = 608
Cfg.h = 608
Cfg.train_data = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\label.txt"
Cfg.anchors_path = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\yolo_anchors.txt"
Cfg.classes_path = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\my_classes.txt"
Cfg.pth_path = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\check_point\Epoch_049_Loss_4.9219.pth"
Cfg.check = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\check_point"

# train
Cfg.batch = 128# 更大得batch采用更大的学习率
Cfg.subdivisions = 32
Cfg.min_batch = Cfg.batch//Cfg.subdivisions
Cfg.smoooth_label = False
Cfg.first_train = False
#cut param(first_train采用):
#431: backbone
#467: backbone+SPP
#509: backbone+SPP+1_concat
#557: 对应darknet的137
Cfg.cut=557
Cfg.cur_epoch = 0
Cfg.total_epoch = 80
Cfg.freeze_mode = False
Cfg.momentum = 0.949
Cfg.decay = 0.0005
# 优化器
Cfg.TRAIN_OPTIMIZER = 'adam'
# 学习率调整策略
# 是否使用余弦退火(cosine)或steps策略(steps)或自定义(other)
Cfg.policy='other'
Cfg.learning_rate = 0.001
Cfg.burn_in = 200
Cfg.max_batches = 8000
Cfg.steps = [2000, 4000]
Cfg.scales = [1, 0.1, 0.01]

# 数据增强
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1
Cfg.jitter = 0.3
Cfg.mosaic = True

#valid
Cfg.confidence = 0.3
Cfg.nms_thresh = 0.2
# NMS.IoU：0.3 ~ 0.5 是大部分目标检测模型默认值
# 如果目标本身不明显或者检测难 → 可以调低
# 如果想保证检测框准确 → 可以调高
# 4、0.1 ~ 0.3 → 严格去重，容易漏掉相邻目标
# 0.4 ~ 0.5 → 平衡
# 0.6 ~ 0.7 → 保留重叠目标，但可能重复检测
Cfg.draw_box = True
Cfg.input_dir = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\object-detection-crowdai"
Cfg.save_err_mis = True
from init import *
from config import *

def load_model_pth(model, pth, cut=None):
    print('Loading weights into state dict, name: %s'%(pth))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pth, map_location=device)
    match_dict = {}
    print_dict = {}
    #431: backbone
    #467: backbone+SPP
    #509: backbone+SPP+1_concat
    #557: 对应darknet的137
    if cut== None: cut = (len(pretrained_dict) - 1)
    try:
        for i, (k, v) in enumerate(pretrained_dict.items()):
            if i <= cut:
                assert np.shape(model_dict[k]) == np.shape(v)
                match_dict[k] = v
                print_dict[k] = v
            else:
                print_dict[k] = '[NO USE]'
    except:
        print('different shape with:', np.shape(model_dict[k]), np.shape(v), '  name:', k)
    for i, key in enumerate(print_dict):
        value = print_dict[key]
        print('items:', i, key, np.shape(value) if type(value) != str else value)
    model_dict.update(match_dict)
    model.load_state_dict(model_dict)
    print('model ready!')
    return model

def print_model(model):
    model_dict = model.state_dict()
    for i, key in enumerate(model_dict):
        print('model items:', i, key, '---->', np.shape(model_dict[key]))

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([3, 3, 2])

def get_train_lines(train_data):
    val_split = 0.1
    with open(train_data) as f:
        lines = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    return lines, num_train, num_val

def get_epoch_by_name(pth_path):
    try:
        pth = pth_path
        epoch = os.path.split(pth)[-1].split('_')[1]
        epoch = int(epoch)
    except Exception as e:
        print(e, 'start epoch: %d'%Cfg.cur_epoch)
        return Cfg.cur_epoch
    return epoch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_optim(lr,model):
    if Cfg.TRAIN_OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'initial_lr': lr}],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif Cfg.TRAIN_OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            [{'params': model.parameters(), 'initial_lr': lr}],
            lr=lr,
            momentum=Cfg.momentum,
            weight_decay=Cfg.decay,
        )
    else:
        print('optimizer must be adam or sgd...')
        return None,None
    return optimizer

def burnin_schedule(i):
    i = i+1
    if i < Cfg.burn_in:
        factor = pow(i / Cfg.burn_in, 4)
    elif i < Cfg.steps[0]:
        factor = Cfg.scales[0]
    elif i < Cfg.steps[1]:
        factor = Cfg.scales[1]
    else:
        factor = Cfg.scales[2]
    return factor

def gen_lr_scheduler(lr, cur_epoch, model):
    optimizer=get_optim(lr,model)
    if Cfg.policy=='other':
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule, last_epoch=cur_epoch-1)
        print('update learning rate:', lr_scheduler.get_last_lr()[0])
    else:
        init_lr = lr*pow(0.9, cur_epoch)
        print('init learning rate:', init_lr)
        if Cfg.policy=='cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        if Cfg.policy=='steps':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    return lr_scheduler,optimizer

def find_pth_by_epoch(epoch):
    pth=r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\check_point"
    pth_list = os.listdir(pth)
    for name in pth_list:
        curpo = name.split('_')[1]
        if int(curpo) == epoch:
            return os.path.join(pth, name)
    print('pth_path is error...')
    return False

if __name__=="__main__":
    print(get_classes(r"F:\1_project_store\2_dev_info\3_data_set\1_cv\10_yolo_v4\car_pedestrain\my_classes.txt"))
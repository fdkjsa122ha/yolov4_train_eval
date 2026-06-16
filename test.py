from get_info import *
from Evaluation.map_eval_pil import compute_map
from utils.dataloader import test_dataset_collate, TestDataset
from config import *

#调用Evaluation模块, 进行map计算和类别准召率计算
def make_labels_and_compute_map(infos, classes, input_dir, save_err_miss=False):
    out_lines, gt_lines = [], []
    eval_out, eval_gt= r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\evaluation\out.txt",r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\evaluation\true.txt"
    foutw = open(eval_out, 'w')
    fgtw = open(eval_gt, 'w')
    for info in infos:
        out, gt, shapes = info
        for i, images in enumerate(out):
            for box in images:
                bbx = str([box[0]*shapes[i][1], box[1]*shapes[i][0], box[2]*shapes[i][1], box[3]*shapes[i][0]])
                cls = classes[int(box[6])]
                prob = str(box[4])
                img_name = os.path.split(shapes[i][2])[-1]
                line = '\t'.join([img_name, 'Out:', cls, prob, bbx])+'\n'
                out_lines.append(line)
        for i, images in enumerate(gt):
            for box in images:
                bbx = str(box.tolist()[0:4])
                cls = classes[int(box[4])]
                img_name = os.path.split(shapes[i][2])[-1]
                line = '\t'.join([img_name, 'Out:', cls, '1.0', bbx])+'\n'
                gt_lines.append(line)
    foutw.writelines(out_lines)
    fgtw.writelines(gt_lines)
    foutw.close()
    fgtw.close()

    args = EasyDict()
    args.annotation_file = eval_gt
    args.detection_file = eval_out
    args.detect_subclass = False
    args.confidence = 0.35# 更关注准确率还是框定准确
    args.iou = 0.2
    args.record_mistake = True
    args.draw_full_img = save_err_miss
    args.draw_cut_box = False
    args.input_dir = input_dir
    args.out_dir = 'out_dir'
    Map = compute_map(args)
    return Map

def main():
    anchors, classes=get_anchors(Cfg.anchors_path), get_classes(Cfg.classes_path)
    model = YoloBody(len(anchors[0]), len(classes))
    yolo_decodes = [YoloLayer((Cfg.w, Cfg.h), [[0, 1, 2], [3, 4, 5], [6, 7, 8]], len(classes), anchors.reshape(-1)).eval() for _ in range(3)]
    lines, num_train, num_val = get_train_lines(Cfg.train_data)
    val_dataset = TestDataset(lines[num_train:], (Cfg.h, Cfg.w))
    gen_val = DataLoader(val_dataset, batch_size=Cfg.min_batch, num_workers=8, pin_memory=True, collate_fn=test_dataset_collate)

    epoch_size_val = num_val // Cfg.min_batch
    # writer = SummaryWriter(log_dir=r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\valid_logs",flush_secs=60)
        
    # for epo in range(18, Cfg.total_epoch + 1):
    for epo in range(54, 55):
        pth_path = find_pth_by_epoch(epo)
        if not pth_path:
            break
        model = load_model_pth(model, pth_path).to(device)
        model.eval()
        with tqdm(total=epoch_size_val, desc= f'Epoch {epo + 1}/{Cfg.total_epoch}', mininterval= 1) as pbar:
            infos = []
            for i, batch in enumerate(gen_val):
                images_src, images, targets, shapes = batch[0], batch[1], batch[2], batch[3]
                with torch.no_grad():
                    images_val= torch.from_numpy(images).float().to(device)
                    outputs = model(images_val)
                    output_list = [yolo_decodes[i](outputs[i]) for i in range(3)]
                    output = torch.cat(output_list, 1)
                    batch_detections = non_max_suppression(output, len(classes), conf_thres=Cfg.confidence, nms_thres=Cfg.nms_thresh)
                    boxs = [box.cpu().numpy() for box in batch_detections if box != None]
                    infos.append([boxs, targets, shapes])

                    if Cfg.draw_box:
                        for x in range(len(boxs)):
                            os.makedirs('result_%d'%epo, exist_ok=True)
                            savename = os.path.join('result_%d'%epo, os.path.split(shapes[x][2])[-1])
                            plot_boxes_cv2(images_src[x], boxs[x], savename=savename, class_names=classes)
                pbar.update(1)
            Map = make_labels_and_compute_map(infos, classes, Cfg.input_dir, save_err_miss=Cfg.save_err_mis)
            # writer.add_scalar('MAP/epoch', Map, epo)

if __name__=="__main__":
    main()
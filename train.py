from get_info import *
from utils.dataloader import train_dataset_collate, TrainDataset
from config import *

def main():
    # 初始化
    anchors = get_anchors(Cfg.anchors_path)
    classes = get_classes(Cfg.classes_path)
    model = YoloBody(len(anchors[0]), len(classes))
    model = load_model_pth(model, Cfg.pth_path, cut=Cfg.cut).to(device) if Cfg.first_train else load_model_pth(model, Cfg.pth_path).to(device)
    yolo_losses = [YOLOLoss(anchors.reshape(-1, 2), len(classes), (Cfg.w, Cfg.h), Cfg.smoooth_label) for _ in range(3)]

    lines, num_train, _ = get_train_lines(Cfg.train_data)
    train_dataset = TrainDataset(lines[:num_train], (Cfg.h, Cfg.w), mosaic=Cfg.mosaic)
    gen = DataLoader(train_dataset, batch_size=Cfg.min_batch, num_workers=8, pin_memory=True, drop_last=True, shuffle=True, collate_fn=train_dataset_collate)

    cur_epoch = get_epoch_by_name(Cfg.pth_path) if not Cfg.first_train else 0
    cur_batch = num_train * cur_epoch // Cfg.batch
    num_mini_batch = max(1, num_train // Cfg.min_batch)
    lr_scheduler, optimizer = gen_lr_scheduler(Cfg.learning_rate, cur_batch, model)
    writer = SummaryWriter(log_dir='train_logs',flush_secs=60)

    for epoch in range(cur_epoch, Cfg.total_epoch):
        total_loss = 0
        cur_step = 0
        with tqdm(total= num_train//Cfg.batch, desc= f'Epoch {epoch + 1}/{Cfg.total_epoch}', postfix= {}, mininterval= 5) as pbar:
            model.train()
            for iteration, batch in enumerate(gen):
                if iteration >= num_mini_batch:
                    break
                images, targets= torch.tensor(batch[0],dtype=torch.float32,device=device, requires_grad=False), [torch.tensor(ann,dtype=torch.float32,device=device, requires_grad=False) for ann in batch[1]]
                outputs = model(images)
                losses, losses_loc, losses_conf, losses_cls = [], [], [], []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets)
                    losses.append(loss_item[0])
                    losses_conf.append(loss_item[1])
                    losses_cls.append(loss_item[2])
                    losses_loc.append(loss_item[3])

                loss = sum(losses) / Cfg.min_batch
                loss.backward()
                total_loss += loss
                cur_step += 1
                #将第五个Epoch开始写入到tensorboard，每一步都写
                if epoch > 3:
                    writer.add_scalar('total_loss/gpu_batch', loss, (epoch * num_mini_batch + iteration))
                    writer.add_scalar('loss_loc/gpu_batch', sum(losses_loc) / Cfg.min_batch, (epoch * num_mini_batch + iteration))
                    writer.add_scalar('loss_conf/gpu_batch', sum(losses_conf) / Cfg.min_batch, (epoch * num_mini_batch + iteration))
                    writer.add_scalar('loss_cls/gpu_batch', sum(losses_cls) / Cfg.min_batch, (epoch * num_mini_batch + iteration))

                if cur_step % Cfg.subdivisions == 0:
                    optimizer.step()
                    lr_scheduler.step()# 每个batch更新一次学习率
                    model.zero_grad()
                    pbar.set_postfix(**{'loss_cur': loss.item(),
                                        'loss_total': total_loss.item() / (iteration + 1),
                                        'lr': get_lr(optimizer)})
                    pbar.update(1)

        print('Saving model %s...\nTotal Loss: %.5f || Last Loss: %.5f ' %  (str(epoch + 1), total_loss / num_mini_batch, loss.item()))
        torch.save(model.state_dict(), '%s/Epoch_%03d_Loss_%.4f.pth' % (Cfg.check, (epoch + 1), total_loss / num_mini_batch))

if __name__=="__main__":
    # Windows 没有 fork()，启动新进程时会重新import主脚本, 导致无限递归
    main()
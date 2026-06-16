import cv2
import os

# 配置路径
label_file = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\label.txt"          # 标签文件路径（每行：图片路径 框1 框2 ...）
output_dir = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\origin_label"             # 保存绘制后图片的文件夹
os.makedirs(output_dir, exist_ok=True)

# 可选：绘制参数
box_color = (0, 255, 0)            # 矩形框颜色 (BGR)，绿色
line_thickness = 2
text_color = (0, 255, 0)
text_scale = 0.5

# 读取所有行
with open(label_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    img_path = parts[0]
    # 处理路径分隔符（Windows反斜杠转正斜杠）
    img_path = img_path.replace('\\', '/')
    if not os.path.exists(img_path):
        print(f"警告：图片不存在，跳过 {img_path}")
        continue

    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图片，跳过 {img_path}")
        continue

    # 解析边界框
    boxes = parts[1:]
    for box_str in boxes:
        coords = list(map(int, box_str.split(',')))
        if len(coords) != 5:
            print(f"坐标格式错误: {box_str}")
            continue
        x1, y1, x2, y2, cls_id = coords
        # 绘制矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, line_thickness)
        # 绘制类别标签（可选）
        label_text = str(cls_id)
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)

    # 生成输出路径（保持原文件名）
    base_name = os.path.basename(img_path)
    out_path = os.path.join(output_dir, base_name)
    cv2.imwrite(out_path, img)
    print(f"已保存: {out_path}")

print("全部处理完成！")
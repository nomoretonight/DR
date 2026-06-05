简要说明（中文）：

- 数据格式（CSV）：必须包含列: image_path, patient_id, label, referable, gradable
  - image_path: 相对于 image_root 的路径或绝对路径
  - label: DR分级 0-4（整数）
  - referable: 0或1（>=2为referable）
  - gradable: 1表示图像可读，0表示不可读（可选过滤）
- 运行:
  - pip install -r requirements.txt
  - 修改 config.yaml 路径与参数
  - python train.py --config config.yaml
- 训练输出保存在 output.work_dir 中，包含 best_model.pth, logs 与 Grad-CAM 示例图片

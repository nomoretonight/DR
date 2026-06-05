import torch
import argparse
from model import DRMultiHeadModel

def export_torchscript(state_path, out_path='model_ts.pt', backbone='efficientnet_b3'):
    device = torch.device('cpu')
    model = DRMultiHeadModel(backbone_name=backbone, pretrained=False).to(device)
    ckpt = torch.load(state_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    # example input
    example = torch.randn(1,3,512,512)
    traced = torch.jit.trace(model, example)
    traced.save(out_path)
    print("Saved TorchScript to", out_path)

def export_onnx(state_path, out_path='model.onnx', backbone='efficientnet_b3'):
    import onnx
    device = torch.device('cpu')
    model = DRMultiHeadModel(backbone_name=backbone, pretrained=False).to(device)
    ckpt = torch.load(state_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    example = torch.randn(1,3,512,512)
    torch.onnx.export(model, example, out_path, opset_version=13, input_names=['input'], output_names=['multiclass_logits', 'referable_logits'], dynamic_axes={'input':{0:'batch_size'}, 'multiclass_logits':{0:'batch_size'}, 'referable_logits':{0:'batch_size'}})
    # validate
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("Saved ONNX to", out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--ts_out', default='model_ts.pt')
    parser.add_argument('--onnx_out', default='model.onnx')
    parser.add_argument('--backbone', default='efficientnet_b3')
    args = parser.parse_args()
    export_torchscript(args.ckpt, args.ts_out, backbone=args.backbone)
    try:
        export_onnx(args.ckpt, args.onnx_out, backbone=args.backbone)
    except Exception as e:
        print("ONNX export failed:", e)

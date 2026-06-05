import torch
import numpy as np
import cv2
import os
from torchvision.transforms.functional import to_pil_image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        handle_f = self.target_layer.register_forward_hook(forward_hook)
        handle_b = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle_f, handle_b])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        out_multi, out_ref = self.model(input_tensor)
        if class_idx is None:
            # if multiclass: pick max logit
            probs = torch.softmax(out_multi, dim=1)
            class_idx = torch.argmax(probs, dim=1).item()
        score = out_multi[:, class_idx].sum()
        score.backward(retain_graph=True)
        grads = self.gradients.cpu().numpy()[0]  # C,H,W
        activations = self.activations.cpu().numpy()[0]  # C,H,W
        weights = np.mean(grads, axis=(1,2))  # C
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam(img_np, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + img_np * (1 - alpha)).astype(np.uint8)
    return overlay

if __name__ == '__main__':
    # usage example:
    # python gradcam.py  --model outputs/best_model.pth --image data/images/xxx.jpg
    import argparse
    from model import DRMultiHeadModel
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', default='gradcam_out.png')
    parser.add_argument('--backbone', default='efficientnet_b3')
    args = parser.parse_args()
    cfg_backbone = args.backbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRMultiHeadModel(backbone_name=cfg_backbone, pretrained=False)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    # pick last conv layer from backbone (for timm models often .conv_head or features)
    # This is heuristic; adjust per backbone
    target_layer = None
    for name, module in model.backbone.named_modules():
        if 'conv_head' in name or 'conv' in name and 'head' in name:
            target_layer = module
    if target_layer is None:
        # fallback: take last module
        target_layer = list(model.backbone.children())[-1]
    gradcam = GradCAM(model, target_layer)
    img = Image.open(args.image).convert('RGB')
    import numpy as np
    img_np = np.array(img)
    from albumentations import Compose, Resize, Normalize
    import albumentations as A
    transform = A.Compose([A.Resize(512,512), A.Normalize()])
    inp = transform(image=img_np)['image']
    inp = np.transpose(inp, (2,0,1))[None, ...].astype(np.float32)
    inp_tensor = torch.tensor(inp).to(device)
    cam = gradcam(inp_tensor)
    overlay = overlay_cam(cv2.resize(img_np, (cam.shape[1], cam.shape[0])), cam)
    from PIL import Image
    Image.fromarray(overlay).save(args.out)
    print("Saved:", args.out)

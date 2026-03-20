from fastai.vision.all import xresnet18, resnet18

def get_arch(arch_str):
    arch = xresnet18 if arch_str == "xresnet18" else resnet18
    return arch

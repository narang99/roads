from mtrain.denorm import denormalize_imagenet
from pytorch_grad_cam import (
    GradCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def show_gradcam_for_image(learn, input_tensor, target_label_idx=None, layer_name="0.7.1.conv1"):
    target_layers = [learn.model.get_submodule(layer_name)]
    img_arr = denormalize_imagenet(input_tensor[0]).permute([1, 2, 0]).numpy()

    targets = [ClassifierOutputTarget(target_label_idx)]
    with GradCAM(model=learn.model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        print(img_arr.shape, grayscale_cam.shape)
        visualization = show_cam_on_image(img_arr, grayscale_cam, use_rgb=True)
        return visualization, img_arr

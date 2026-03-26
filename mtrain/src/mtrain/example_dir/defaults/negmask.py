from mtrain.neg_mask.model.datasets.blur_pad_dl import blur_overwriter
from mtrain.example_dir.learners import (
    NegmaskLearner,
    step_downer,
    get_raw_negmask_learner,
)


def default_negmask_learners(models_dir, labels, bs):
    res = {}

    if "md" in labels:
        path = (
            models_dir
            / "negmask"
            / "tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth"
        )
        res["md"] = get_configured_negmask_learner(
            "md", bs, 224, 10, step_downer, path
        )
    if "sm" in labels:
        # needs testing, I honestly dont have a good smallnet accepting model
        path = models_dir / "negmask" / "unblurred-iter100-arch-resnet18.pt"
        res["sm"] = get_configured_negmask_learner(
            "sm", bs, 128, 10, blur_overwriter(13, 4), path, "resnet18"
        )

    if "unblurred" in labels:
        # this needs testing, will be done later
        path = models_dir / "negmask" / "unblurred-iter100-arch-resnet18.pt"
        res["unblurred"] = get_configured_negmask_learner(
            "unblurred", bs, 128, 10, blur_overwriter(13,4), path, "resnet18"
        )
    if "high-recall" in labels:
        path = models_dir / "negmask" / "high-trash-recall-v2-tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth"
        res["high-recall"] = get_configured_negmask_learner(
            "high-recall", bs, 224, 10, step_downer, path, "xresnet18"
        )

    if "other-high-recall" in labels:
        path = models_dir / "successive-224" / "high-other-recall-v2-tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth"
        res["other-high-recall"] = get_configured_negmask_learner(
            "other-high-recall", bs, 224, 10, step_downer, path, "xresnet18"
        )
    return res


def get_configured_negmask_learner(
    label, bs, crop_size, bbox_pad, mutator, pth_path, arch="xresnet18"
) -> NegmaskLearner:
    return NegmaskLearner(
        label,
        get_raw_negmask_learner(bs, crop_size, pth_path, arch),
        bs,
        crop_size,
        bbox_pad,
        mutator,
    )

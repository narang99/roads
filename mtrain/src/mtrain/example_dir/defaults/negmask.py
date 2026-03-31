from mtrain.neg_mask.model.datasets.foviate_shrink import get_foviate_remaps
from torch.ao.ns.fx.mappings import get_base_name_to_sets_of_related_ops
from mtrain.neg_mask.model.datasets.blur_pad_dl import (
    blur_overwriter,
    BlurPadFoveateInferDataset,
)
from mtrain.example_dir.learners import (
    NegmaskLearner,
    step_downer,
    get_raw_negmask_learner,
    foveate_shrink_and_step_down,
    EnsembledNegmaskLearner,
)


def default_negmask_learners(models_dir, labels, bs):
    res = {}

    if "md" in labels:
        path = (
            models_dir
            / "negmask"
            / "tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth"
        )
        res["md"] = get_configured_negmask_learner("md", bs, 224, 10, step_downer, path)
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
            "unblurred", bs, 128, 10, blur_overwriter(13, 4), path, "resnet18"
        )
    if "high-recall" in labels:
        path = (
            models_dir
            / "negmask"
            / "high-trash-recall-v2-tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth"
        )
        res["high-recall"] = get_configured_negmask_learner(
            "high-recall", bs, 224, 10, step_downer, path, "xresnet18"
        )

    if "other-high-recall" in labels:
        path = (
            models_dir
            / "successive-224"
            / "high-other-recall-v2-tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth"
        )
        res["other-high-recall"] = get_configured_negmask_learner(
            "other-high-recall", bs, 224, 10, step_downer, path, "xresnet18"
        )
    if "latest" in labels:
        res["latest"] = get_baseline_foveated_model("latest", models_dir, bs)
    if "baseline" in labels:
        res["baseline"] = get_baseline_foveated_model("baseline", models_dir, bs)
        # res["baseline"] = EnsembledNegmaskLearner(
        #     "baseline",
        #     [
        #         get_baseline_foveated_model("latest", models_dir, bs),
        #         get_baseline_md_model("md", models_dir, bs),
        #     ],
        # )
    if "experimental" in labels:
        res["experimental"] = get_foveated_model(
            "experimental",
            models_dir / "foveated-224" / "iter-7-with-walls-v1-xresnet18.pth",
            bs,
        )

    return res


def get_baseline_md_model(label, models_dir, bs):
    path = (
        models_dir
        / "negmask"
        / "tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth"
    )
    return get_configured_negmask_learner("md", bs, 224, 10, step_downer, path)


def get_baseline_foveated_model(label, models_dir, bs):
    path = models_dir / "foveated-224" / "iter-7-xresnet18.pth"
    return get_configured_negmask_learner(
        label,
        bs,
        1024,
        3,
        foveate_shrink_and_step_down,
        path,
        "xresnet18",
        valid_tfms_crop_size=224,
        dataset_class=BlurPadFoveateInferDataset,
    )


def get_foveated_model(label, path, bs):
    return get_configured_negmask_learner(
        label,
        bs,
        1024,
        3,
        foveate_shrink_and_step_down,
        path,
        "xresnet18",
        valid_tfms_crop_size=224,
        dataset_class=BlurPadFoveateInferDataset,
    )

def get_configured_negmask_learner(
    label,
    bs,
    crop_size,
    bbox_pad,
    mutator,
    pth_path,
    arch="xresnet18",
    valid_tfms_crop_size=None,
    dataset_class=None,
) -> NegmaskLearner:
    return NegmaskLearner(
        label,
        get_raw_negmask_learner(bs, crop_size, pth_path, arch),
        bs,
        crop_size,
        bbox_pad,
        mutator,
        valid_tfms_crop_size,
        dataset_class,
    )

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
            / "tfm-stepedge_data-withtaco_iter-25_arch-xresnet18.pth"
        )
        res["md"] = get_configured_negmask_learner(
            "md", bs, 224, 10, step_downer, path
        )
    if "sm" in labels:
        # path = models_dir / "successive-224" / "sm-st_ed_tfm0-final-all-data-with-taco-iter-10.pth"
        # res["sm"] = get_configured_negmask_learner(
        #     "sm", bs, 224, 10, step_downer, path, "xresnet18"
        # )
        path = models_dir / "negmask" / "unblurred-iter100-arch-resnet18.pt"
        res["sm"] = get_configured_negmask_learner(
            "sm", bs, 128, 10, step_downer, path, "resnet18"
        )

    if "unblurred" in labels:
        path = models_dir / "negmask" / "unblurred-iter100-arch-resnet18.pt"
        res["unblurred"] = get_configured_negmask_learner(
            "unblurred", bs, 128, 10, step_downer, path, "resnet18"
        )
    if "step_edge_v1" in labels:
        path = models_dir / "negmask" / "step-edge-v1-xresnet18.pth"
        res["step_edge_v1"] = get_configured_negmask_learner(
            "step_edge_v1", bs, 128, 10, step_downer, path, "xresnet18"
        )
    if "latest" in labels:
        path = models_dir / "successive-224" / "tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth"
        res["latest"] = get_configured_negmask_learner(
            "latest", bs, 224, 10, step_downer, path, "xresnet18"
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

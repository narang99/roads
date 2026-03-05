def get_preds_for_ds(learn, ds, device=None, bs=4):
    from torch.utils.data import DataLoader as TorchDataLoader
    import torch.nn.functional as F
    import torch
    from fastai.vision.all import default_device

    if device is None:
        device = default_device()

    loader = TorchDataLoader(ds, batch_size=bs, shuffle=False)
    learn.model.eval()
    learn.model.to(device)
    all_preds, all_targs, all_losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = learn.model(x)
            preds = logits.softmax(dim=1)
            losses = F.cross_entropy(logits, y.to(device), reduction='none')
            all_preds.append(preds.cpu())
            all_targs.append(y.cpu())
            all_losses.append(losses.cpu())
    all_preds = torch.cat(all_preds)
    all_targs = torch.cat(all_targs)
    all_losses = torch.cat(all_losses)
    decoded = all_preds.argmax(dim=1)
    TRASH, OTHER = 0, 1
    fp_idxs = ((all_targs == OTHER) & (decoded == TRASH)).nonzero().squeeze()
    fn_idxs = ((all_targs == TRASH) & (decoded == OTHER)).nonzero().squeeze()
    return all_preds, all_targs, decoded, fp_idxs, fn_idxs, all_losses
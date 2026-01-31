import torch


def save_checkpoint(model, optimizer, iteration: int, out):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, out)


def load_checkpoint(src, model, optimizer) -> int:
    payload = torch.load(src, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["iteration"])

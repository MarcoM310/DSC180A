import numpy as np
import torch
from tqdm import tqdm

# testing
def test1Epoch(epoch_index, model, loss_fn, valid_loader, tb_writer=None):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    with torch.no_grad():
        for i, (image, bnpp, _) in tqdm(
            enumerate(valid_loader), total=len(valid_loader)
        ):
            image, bnpp = image.to(device, non_blocking=True), bnpp.to(
                device, non_blocking=True
            )

            pred = model(image)
            loss = loss_fn(torch.squeeze(pred, 1), bnpp).detach()
            losses = np.append(losses, loss.item())
            image.detach()
            bnpp.detach()
    return np.mean(losses)

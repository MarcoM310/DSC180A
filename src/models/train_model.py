import numpy as np
import torch
from tqdm import tqdm

# training
def train1Epoch(epoch_index, model, optimizer, loss_fn, training_loader, writer=None):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    for i, (image, bnpp, _) in tqdm(
        enumerate(training_loader), total=len(training_loader)
    ):
        # image, bnpp, _= data
        image, bnpp = image.to(device, non_blocking=True), bnpp.to(
            device, non_blocking=True
        )
        pred = model(image)
        loss = loss_fn(torch.squeeze(pred, 1), bnpp)
        # Backpropagation
        optimizer.zero_grad()  # set_to_none=True)
        loss.backward()
        optimizer.step()

        losses = np.append(losses, loss.item())
    return np.mean(losses)

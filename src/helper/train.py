from tqdm import tqdm
import numpy as np
import torch

# training
def train1Epoch(epoch_index, model, optimizer, loss_fn, training_loader, writer=None):
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (image, bnpp, _) in tqdm(
        enumerate(training_loader), total=len(training_loader)
    ):
        # Every data instance is an input + label pair

        # image, bnpp, _= data
        image, bnpp = image.to(device, non_blocking=True), bnpp.to(
            device, non_blocking=True
        )
        # bnpp = bnpp.float()

        # print('image size: ',image.shape)
        # print(torch.mean(image))
        # HERE

        pred = model(image)

        # print('BNPP size: ',bnpp.shape)
        # print('Pred size: ',pred.squeeze().shape)
        # print('BNPP: ',bnpp)
        # print('Preds: ',pred)
        # print('Preds: ',pred.squeeze())

        loss = loss_fn(torch.squeeze(pred, 1), bnpp)
        # print('loss: ',loss.item())
        # Backpropagation
        # <<<<<<< HEAD
        optimizer.zero_grad()  # set_to_none=True)
        # =======
        optimizer.zero_grad(set_to_none=True)
        # >>>>>>> 1960aca70cc080d2575d221d6bf6f58b6cdad5df
        loss.backward()
        optimizer.step()

        # HERE
        #         # Zero your gradients for every batch!
        #         optimizer.zero_grad()

        #         # Make predictions for this batch
        #         outputs = model(image)

        #         # Compute the loss and its gradients
        #         loss = loss_fn(outputs.squeeze(), bnpp)
        #         loss.backward()
        #         #print(f"{loss=}")

        #         # Adjust learning weights
        #         optimizer.step()

        # Gather data and report
        # running_loss += loss.item()
        losses = np.append(losses, loss.item())
        # writer.add_text('batch_losses', \
        #        f'''batch #{i}\n
        #        losses: {losses}
        #        ''',\
        #        0)
        # print(f"{running_loss=}")
    # last_loss += running_loss / (len(training_loader)) #number of batches
    # print('last_loss: ',last_loss)
    # if i % 1000 == 0:
    #   last_loss = running_loss / len(training_loader) # loss per batch
    # print('  batch {} loss: {}'.format(i + 1, last_loss))
    # tb_x = epoch_index * len(training_loader) + i + 1
    # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    #    running_loss = 0.
    # print('Running Losses: ',losses)
    return np.mean(losses)


def test1Epoch(epoch_index, model, loss_fn, valid_loader, tb_writer=None):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    with torch.no_grad():
        for i, (image, bnpp, _) in tqdm(
            enumerate(valid_loader), total=len(valid_loader)
        ):
            # Every data instance is an input + label pair

            # image, bnpp,_ = data
            image, bnpp = image.to(device, non_blocking=True), bnpp.to(
                device, non_blocking=True
            )
            # bnpp = bnpp.float()
            # print('BNPP: ',bnpp)
            # print(torch.mean(image))
            # HERE

            pred = model(image)
            # print('BNPP: ',bnpp)
            # print('Preds: ',torch.squeeze(pred,1))

            loss = loss_fn(torch.squeeze(pred, 1), bnpp).detach()
            # print('loss: ',loss.item())

            # HERE
            #         # Zero your gradients for every batch!
            #         optimizer.zero_grad()

            #         # Make predictions for this batch
            #         outputs = model(image)

            #         # Compute the loss and its gradients
            #         loss = loss_fn(outputs.squeeze(), bnpp)
            #         loss.backward()
            #         #print(f"{loss=}")

            #         # Adjust learning weights
            #         optimizer.step()

            # Gather data and report
            # running_loss += loss.item()
            losses = np.append(losses, loss.item())
            image.detach()
            bnpp.detach()

        # print(f"{running_loss=}")
    # last_loss += running_loss / (len(training_loader)) #number of batches
    # print('last_loss: ',last_loss)
    # if i % 1000 == 0:
    #   last_loss = running_loss / len(training_loader) # loss per batch
    # print('  batch {} loss: {}'.format(i + 1, last_loss))
    # tb_x = epoch_index * len(training_loader) + i + 1
    # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    #    running_loss = 0.
    # print('Running Losses: ',losses)
    return np.mean(losses)

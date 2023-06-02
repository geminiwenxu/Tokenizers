import numpy as np
import torch


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(torch.float32).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        outputs = outputs.reshape(-1)
        # softmax = F.softmax(logits, dim=-1)
        # outputs = torch.argmax(softmax, dim=-1)
        print("outputs: ", outputs, outputs.dtype, outputs.shape)
        preds = (outputs > 0.5).float()
        print("targets: ", targets, targets.dtype, targets.shape)
        print("predicts: ", preds, preds.dtype, preds.shape)
        loss = loss_fn(outputs.float(), targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(torch.float32).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            outputs = outputs.reshape(-1)
            preds = (outputs > 0.5).float()
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(data, seq_length, batch_size, device='cpu'):
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def eval(model, data_tensor, sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


@torch.no_grad()
def eval_sparse(model, data_tensor, sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext(), alpha_th=None, drop_k=None):
    assert model.training == False

    ce_loss_list_val, l1_loss_list_val, acc_list, sparcity_per_layer = [], [], [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
        with ctx:
            outputs = model(x, targets=y, alpha_th=alpha_th, drop_k=drop_k, get_logits=True, get_alphas=True)
        ce_loss_list_val.append(outputs['ce_loss'])
        l1_loss_list_val.append(outputs['l1_loss'])
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())
        sparcity_per_layer.append([(alphas < 1e-8).sum().float().cpu().item() / alphas.numel() for alphas in outputs['alphas']])

    val_acc = torch.stack(acc_list).mean().item()
    val_ce_loss = np.mean(ce_loss_list_val)
    val_l1_loss = np.mean(l1_loss_list_val)
    val_perplexity = 2.71828 ** val_ce_loss
    sparcity_per_layer = np.mean(np.array(sparcity_per_layer), axis=0)

    return val_acc, val_ce_loss, val_l1_loss, val_perplexity, sparcity_per_layer


@torch.no_grad()
def eval_sweep_dropk(model, data_tensor, sequence_length, batch_size, n_heads, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = torch.linspace(0.0,0.95,15), [], [], []
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches): 
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True)
            loss_list_val.append(outputs['ce_loss'])
            acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(model, data_tensor, sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1], [], [], []
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches): 
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True)
            nph, nh = outputs['num_head_pruned_per_layer'], outputs['num_heads_per_layer']
            frac_heads_pruned = np.sum(nph) / np.sum(nh) # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs['ce_loss'])
            acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)

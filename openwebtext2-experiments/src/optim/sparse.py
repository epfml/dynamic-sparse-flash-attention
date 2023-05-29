from contextlib import nullcontext
import torch
import torch.nn.functional as F
import wandb
import time 
import copy

from .utils import eval_sparse, get_batch, eval_sweep_dropk, save_checkpoint


def train_sparse(model, opt, data, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table, sparsity_plot = 0, 0, float('inf'), None, None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_ce_loss': [], 'train_l1_loss': [], 'val_l1_loss': [], 'val_ce_loss': [], 'val_pp': [], 'val_acc': []}

    num_substeps_per_epoch = len(data['train']) // (batch_size * sequence_length)
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        model = torch.compile(model) # requires pytorch 2.0+

    model.train()

    t0 = time.time()
    while itr < iterations:
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(data['train'], sequence_length, batch_size, device=extra_args.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                    outputs = model(x, targets=y)

            loss = outputs['loss']
            loss.backward()
            substep += 1

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # eval: from here it's pretty much only evaluation, all the training is above 
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                val_acc, val_ce_loss, val_l1_loss, val_perplexity, sparcity_per_layer = eval_sparse(model, data['val'], sequence_length, batch_size,
                                                                                                    extra_args.device, max_num_batches=24, ctx=type_ctx)

                print_string = f"{epoch}/{itr} [train] ce-loss={outputs['ce_loss']:.3f}, l1-loss={outputs['l1_loss']:.3f}"
                print_string += f" [val] ce-loss={val_ce_loss:.3f}, l1-loss={val_l1_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)

                if extra_args.wandb:
                    wandb.log({
                        "iter": itr,
                        "train/loss": outputs['ce_loss'],
                        "val/loss": val_ce_loss,
                        "train/l1-loss": outputs['l1_loss'],
                        "val/l1-loss": val_l1_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    })

                    if extra_args.eval_seq_prefix != 'none' and (itr % (eval_freq * 5) == 0 or itr == iterations):
                        if text_table is None:
                            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

                        out_str = distributed_backend.get_raw_model(model).generate_from_string(
                            extra_args.eval_seq_prefix, max_new_tokens=40, temperature=0.9, top_k=None)
                        text_table.add_data(itr, val_perplexity, out_str)
                        # why a copy? see github.com/wandb/wandb/issues/2981
                        wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})

                    if itr % (eval_freq * 5) == 0 or itr == iterations:
                        if sparsity_plot is None:
                            layer_idx, sparsity_plot, keys = list(range(1, extra_args.n_layer+1)), [], []
                        sparsity_plot.append(list(sparcity_per_layer))
                        keys.append(f"itr {itr}")
                        wandb.log({f"val-stats/sparsity-{wandb.run.name}": wandb.plot.line_series(xs=layer_idx,
                                  ys=sparsity_plot, keys=keys, title=f"sparsity-{wandb.run.name}")})

                    if itr % (eval_freq * 50) == 0 or itr == iterations:
                        x_axis, y_axis_acc, y_axis_pp, _ = eval_sweep_dropk(model, data['val'], sequence_length, batch_size, extra_args.n_head,
                                                                            extra_args.device, max_num_batches=24, ctx=ctx)
                        table_acc = wandb.Table(data=list(zip(x_axis, y_axis_acc)), columns=["n heads dropped", "acc"])
                        wandb.log({f"val-stats/Acc-drop-k-itr={itr}": wandb.plot.line(table_acc,
                                  "n heads dropped", "acc", title=f"Acc: drop-k-itr={itr}")})
                        table_pp = wandb.Table(data=list(zip(x_axis, y_axis_pp)),
                                               columns=["n heads dropped", "perplexity"])
                        wandb.log({f"val-stats/PP-drop-k-itr={itr}": wandb.plot.line(table_pp,
                                  "n heads dropped", "perplexity", title=f"Perplexity: drop-k-itr={itr}")})

                        x_axis, y_axis_acc, y_axis_pp, _ = eval_sweep_dropk(model, data['val'], sequence_length, batch_size, extra_args.n_head,
                                                                            extra_args.device, max_num_batches=24, ctx=ctx)
                        table_acc = wandb.Table(data=list(zip(x_axis, y_axis_acc)), columns=["n heads dropped", "acc"])
                        wandb.log({f"val-stats/Acc-alpha-th-sweep-itr={itr}": wandb.plot.line(table_acc,
                                  "n heads dropped", "acc", title=f"Acc: alpha-th-sweep-itr={itr}")})
                        table_pp = wandb.Table(data=list(zip(x_axis, y_axis_pp)),
                                               columns=["n heads dropped", "perplexity"])
                        wandb.log({f"val-stats/PP-alpha-th-sweep-itr={itr}": wandb.plot.line(table_pp,
                                  "n heads dropped", "perplexity", title=f"Perplexity: alpha-th-sweep-itr={itr}")})

                model.train()
                t0 = time.time()

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        ckpt_path=ckpt_path)

    return stats


import torch
import tqdm
from trace_tad.utils.misc import AverageMeter


def _swap_ema_weights(model, model_ema):
    """Swap model weights with EMA weights in-place (no allocation)."""
    for p_model, p_ema in zip(model.parameters(), model_ema.module.parameters()):
        tmp = p_model.data.clone()
        p_model.data.copy_(p_ema.data)
        p_ema.data.copy_(tmp)
    for b_model, b_ema in zip(model.buffers(), model_ema.module.buffers()):
        tmp = b_model.data.clone()
        b_model.data.copy_(b_ema.data)
        b_ema.data.copy_(tmp)


def _to_cuda(data_dict):
    """Move tensor values in data_dict to CUDA (non-blocking)."""
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cuda(non_blocking=True)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            data_dict[key] = [v.cuda(non_blocking=True) for v in value]
    return data_dict


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
    accumulation_steps=1,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True

    model.train()
    optimizer.zero_grad()

    for iter_idx, data_dict in enumerate(train_loader):
        data_dict = _to_cuda(data_dict)

        # forward pass
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)

        # scale loss by accumulation steps
        scaled_cost = losses["cost"] / accumulation_steps

        # compute the gradients
        if use_amp:
            scaler.scale(scaled_cost).backward()
        else:
            scaled_cost.backward()

        # step optimizer every accumulation_steps iterations or at the last iteration
        is_accumulation_step = ((iter_idx + 1) % accumulation_steps == 0) or ((iter_idx + 1) == num_iters)

        if is_accumulation_step:
            # gradient clipping (to stabilize training if necessary)
            if clip_grad_l2norm > 0.0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

            # update parameters
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # update scheduler
            scheduler.step()

            # update ema
            if model_ema is not None:
                model_ema.update(model)

        # track all losses (use unscaled values for logging)
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            curr_backbone_lr = None
            if hasattr(model, "backbone"):
                if model.backbone.freeze_backbone == False:
                    curr_backbone_lr = scheduler.get_last_lr()[0]
            curr_det_lr = scheduler.get_last_lr()[-1]

            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))


def val_one_epoch(
    val_loader,
    model,
    logger,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # swap model weights with EMA weights for evaluation
    if model_ema is not None:
        _swap_ema_weights(model, model_ema)

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader):
        data_dict = _to_cuda(data_dict)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            with torch.inference_mode():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # swap back to original model weights
    if model_ema is not None:
        _swap_ema_weights(model, model_ema)
    return losses_tracker["cost"].avg

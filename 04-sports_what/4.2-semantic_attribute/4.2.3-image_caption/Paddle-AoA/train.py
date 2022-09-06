import time
import traceback

import opts
import models
from dataloader import *
import misc.eval_utils as eval_utils
import misc.utils as utils
from modules.loss_wrapper import LossWrapper

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    acc_steps = getattr(opt, 'acc_steps', 1)

    loader = DataLoader(opt)
    opt.vocab_size = loader.get_vocab_size()
    opt.seq_length = loader.get_seq_length()

    tb_summary_writer = tb and tb.SummaryWriter(os.path.join(opt.checkpoint_path, 'log_runs/exp'))  # tensorboard --logdir=/

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                    "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['loader_state_dict'] = None
        infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.load_state_dict(infos['loader_state_dict'])

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    opt.vocab = loader.get_vocab()
    model = models.setup(opt)
    del opt.vocab
    dp_model = paddle.DataParallel(model)
    lw_model = LossWrapper(model, opt)
    dp_lw_model = paddle.DataParallel(lw_model)

    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pdopt")):
        optimizer.set_state_dict(paddle.load(os.path.join(opt.start_from, 'optimizer.pdopt')))

    try:
        while True:
            if epoch_done:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                optimizer.set_lr = opt.current_lr
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                else:
                    sc_flag = False

                epoch_done = False

            # Load data from train split (0)
            data = loader.get_batch('train')
            if iteration % acc_steps == 0:
                optimizer.clear_grad()

            start = time.time()
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'],
                                    paddle.arange(0, len(data['gts'])), sc_flag)

            loss = model_out['loss'].mean()
            loss_sp = loss / acc_steps

            loss_sp.backward()
            if (iteration + 1) % acc_steps == 0:
                optimizer.step()
            train_loss = loss.item()
            end = time.time()

            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, model_out['reward'].mean().item(), end - start))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if iteration % opt.losses_log_every == 0:
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tb_summary_writer, 'avg_reward', model_out['reward'].mean().item(), iteration)

                loss_history[iteration] = train_loss if not sc_flag else model_out['reward'].mean().item()
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()

            # make evaluation on validation set, and save model
            if iteration % opt.save_checkpoint_every == 0:
                # eval model
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                # Write validation result into summary
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k, v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                utils.save_checkpoint(opt, model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    utils.save_checkpoint(opt, model, infos, optimizer, append=str(iteration))

                if best_flag:
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best')

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs != -1:
                break
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


opt = opts.parse_opt()
train(opt)

import argparse
import paddle
import sys
import time
sys.path.append('core')
from raft import *
from datasets import *
from evaluate import *


class Logger:
    def __init__(self, filename, batch_size):
        self.filename = filename
        self.batch_size = batch_size
        self.init()

    def init(self):
        self.num_steps = 0
        self.loss = 0.0
        self.met = {
            'epe': 0.0, 
            'epe1':0.0, 
            'epe3':0.0, 
            'epe5':0.0
            }

    def update(self, metrics, loss):
        for k in metrics.keys():
            self.met[k] += float(metrics[k])
        self.num_steps += 1
        self.loss += loss

    def write(self, trained_numstep, lr): # , eval_epe
        t = time.localtime()
        time_str = "%i/%i %i:%i:%i " % (t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
        training_str = "[trained_numstep: %i  lr: %f]" % (trained_numstep, lr)
        loss_str = " loss:%.5f " % (self.loss / self.num_steps)
        # eval_str = " eval_epe: %f" % eval_epe
        metrics_str = str()
        for i in self.met.keys():
            metrics_str += ' %s:%.6f ' % (i, self.met[i] / (self.num_steps)) # self.batch_size * 
        print(time_str, training_str, loss_str, metrics_str) # , eval_str
        with open('log/%s' % self.filename, 'a') as f:
            f.write("%s%s%s%s\n" % (time_str, training_str, loss_str, metrics_str)) # , eval_str
        self.init()
            
    def write_eval(self, eval_str):
        print(eval_str)
        with open('log/%s' % self.filename, 'a') as f:
            f.write("%s\n" % eval_str)


class Sequence_loss(paddle.nn.Layer):
    def __init__(self, gamma, max_flow):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
    
    def forward(self, flow_preds, flow_gt, valid):
        # Loss function defined over sequence of flow predictions
        n_predictionos = len(flow_preds)
        flow_loss = paddle.tensor.zeros(shape=[1])

        # exlude invalid pixels and extremely large diplacements
        mag = paddle.sum(flow_gt**2, axis=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)
        valid1 = paddle.cast(valid, dtype='float32')


        for i in range(n_predictionos):
            i_weight = self.gamma**(n_predictionos - i - 1)
            i_loss = paddle.abs(flow_preds[i] - flow_gt)
            # flow_loss += paddle.mean(i_weight * (valid[:, None] * i_loss))
            flow_loss += i_weight * paddle.mean(valid1[:, None] * i_loss) 
            # flow_loss += paddle.mean(i_weight * i_loss) 
        
        epe = paddle.sqrt(paddle.sum((flow_preds[-1] - flow_gt)**2, axis=1))
        epe = paddle.reshape(epe, shape=[-1])[paddle.reshape(valid, shape=[-1])]
        # print(epe.shape)
        # 此处可以打印epe的信息
        metrics = {
            'epe': float(paddle.mean(epe)), 
            'epe1': float(paddle.mean(epe[epe <= 1])), # paddle.mean(paddle.cast(paddle.greater_equal(paddle.ones_like(epe), epe), dtype='float32') * epe), 
            'epe3': float(paddle.mean(epe[epe <= 3])), # paddle.mean(paddle.cast(paddle.greater_equal(paddle.ones_like(epe)*3, epe), dtype='float32') * epe), 
            'epe5': float(paddle.mean(epe[epe <= 5])), # paddle.mean(paddle.cast(paddle.greater_equal(paddle.ones_like(epe)*5, epe), dtype='float32') * epe)
        }
        # print(metrics)
        return flow_loss, metrics


def train():
    place = paddle.set_device('gpu')

    model = RAFT()
    paddle.summary(model, input_size=[(1, 3, 368, 496), (1, 3, 368, 496)])

    # ----------------------------------------------Hyperparameter begin--------------------------------------------------------
    num_steps = 120000                      # 训练次数
    batch_size = 16                         # 一批训练数量
    gamma = 0.85                            # 这个是求损失用到的gamma，官方默认是0.85
    max_flow = 400                          # 像素移动的最大值，超过最大值则忽略不计
    num_workers = 4                         # DataLoader读取数据的进程
    learning_rate = 0.00040                 # 学习率
    log_iter = 100                          # 每训练log_iter次打印一次日志
    VAL_FREQ = 2500                         # 每训练VAL_FREQ次验证一次
    log = Logger('test.txt', batch_size)    # 日志文件的名字，位置是work/log
    warmup_proportion = 0.3
    # 学习率策略：线性热启动+线性衰减
    polynomial_lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=learning_rate, decay_steps=int((1-warmup_proportion)*num_steps) + 100, end_lr = learning_rate / 10000)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=polynomial_lr, warmup_steps=int(warmup_proportion*num_steps), start_lr = 0.00001, end_lr=learning_rate)

    # 优化器：AdamW
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, weight_decay=0.00001, epsilon=1e-8, parameters=model.parameters())
    # ----------------------------------------------Hyperparameter end--------------------------------------------------------

    sequence_loss = Sequence_loss(gamma=gamma, max_flow=max_flow)
    train_loader = fetch_dataloader(stage='chairs', batch_size=batch_size, image_size=[368,496], TRAIN_DS=None, num_workers=num_workers, split='training')
    print("begin Training!")
    trained_numstep = 0
    stop_train = False
    while True:
        for batch_id, data in enumerate(train_loader):
            trained_numstep += 1
            img1 = data[0].cuda()
            img2 = data[1].cuda()
            flow = data[2].cuda()
            valid = data[3].cuda()
            flow_p = model(img1, img2)

            loss, metrics = sequence_loss(flow_p, flow, valid)
            loss.backward()

            log.update(metrics=metrics, loss=float(loss))
            optimizer.step()
            optimizer.clear_grad()

            if trained_numstep % log_iter == 0:
                print('trained_numstep:%i  lr:%f  loss:%f  epe:%f  epe1:%f  epe3:%f  epe5:%f ' % 
                (trained_numstep, scheduler.get_lr(), float(loss), metrics['epe'], metrics['epe1'], metrics['epe3'], metrics['epe5']))
            
            scheduler.step()

            if trained_numstep == num_steps:
                stop_train = True
            if stop_train:
                break 
            if trained_numstep % VAL_FREQ == 0 and trained_numstep != 1:
                log.write(trained_numstep=trained_numstep ,lr=scheduler.get_lr()) # ,eval_epe=0.0
                log.write_eval(eval_chair(model))
                paddle.save(model.state_dict(), "output/%i.pdparams" % trained_numstep)
                paddle.save(optimizer.state_dict(), "output/%i.pdopt" % trained_numstep)
        if stop_train:
            break


if __name__ == '__main__':
    train()
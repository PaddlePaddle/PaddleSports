import argparse
import paddle
import sys
import time
sys.path.append('core')
sys.path.append('models')
from datasets import *
from evaluate import *
from FlowNetS import *
from multiscaleloss import *


class Logger:
    def __init__(self, filename, batch_size):
        self.filename = filename
        self.batch_size = batch_size
        self.init()

    def init(self):
        self.num_steps = 0
        self.loss = 0.0
        self.met = {
            'epe': 0.0
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


def train():
    place = paddle.set_device('gpu')
    model = FlowNetS()
    params_info = paddle.summary(model, (1, 6, 368, 496))
    print(params_info)
    # ------------------------------------------------------------------------------------------------------
    batch_size = 32                         # 一批训练数量
    num_workers = 1                         # DataLoader读取数据的进程
    log_iter = 100                          # 每训练log_iter次打印一次日志
    log = Logger('test.txt', batch_size)    # 日志文件的名字，位置是work/log
    epoch_num = 300
    multiscale_weights = [0.005,0.01,0.02,0.08,0.32]
    div_flow = 20
    lr = 0.0001
    milestones = [100, 150, 200]
    multistep_lr = paddle.optimizer.lr.MultiStepDecay(learning_rate=lr, milestones=milestones, gamma=0.5)
    optimizer = paddle.optimizer.AdamW(learning_rate=multistep_lr, weight_decay=0.00001, epsilon=1e-8, parameters=model.parameters())
    # ------------------------------------------------------------------------------------------------------
    train_loader = fetch_dataloader(stage='chairs', batch_size=batch_size, image_size=[368,496], TRAIN_DS=None, num_workers=num_workers, split='training')

    for epoch in range(epoch_num):
        trained_numstep = 0
        for batch_id, data in enumerate(train_loader):
            trained_numstep += 1
            img1 = data[0].cuda()
            img2 = data[1].cuda()
            img = paddle.concat(x=[img1, img2], axis=1)
            # print(img.shape)
            flow = data[2].cuda()
            output = model(img)
            loss = multiscaleEPE(output, flow, weights=multiscale_weights, sparse=False)
            flow2_EPE = div_flow * realEPE(output[0], flow, sparse=False)

            log.update({'epe':flow2_EPE}, loss=float(loss))
            trained_numstep += 1
            if trained_numstep % log_iter == 0:
                log.write(trained_numstep=trained_numstep ,lr=multistep_lr.get_lr()) # ,eval_epe=0.0

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # break

        multistep_lr.step()
        log.write_eval(eval_chair(model))
        paddle.save(model.state_dict(), "output/%i.pdparams" % epoch)
        paddle.save(optimizer.state_dict(), "output/%i.pdopt" % epoch)


if __name__ == '__main__':
    train()


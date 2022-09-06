import paddle
import paddle.nn as nn
from stgcn_framework import STGCN_framework
from stgcn_dataset import Dataset
import os
from tqdm import tqdm
from visualdl import LogWriter
class STGCN_main():
    def __init__(self,args):
        self.log_writer = LogWriter(args.output_log_dir)
        self.bs = args.BATCH_SIZE
        if args.is_train == 1:
            self.is_train= True
        else:
            self.is_train=False
        self.epoches = args.epoches
        self.output_model_dir = args.output_model_dir
        self.dataset = Dataset(args)
        if self.is_train ==True:
            self.valid_dataset = Dataset(args,is_train=False)

        self.load_pretrain_model = args.load_pretrain_model
        self.data_loader = paddle.io.DataLoader(self.dataset,batch_size=self.bs,shuffle =True,drop_last=True)


        pass
    def valid_accurary(self,valid_loader,classifer_net):
        with paddle.set_grad_enabled(False):
            acc_all = 0
            num = 0 
            for one in valid_loader:
                img_data,cls=one
                # print()
                out = classifer_net(img_data)
                # print(out.shape)
                # out = nn.Softmax()(out)
                # out = paddle.multinomial(out, num_samples=1, replacement=False, name=None)
                acc = paddle.metric.accuracy(out,cls.unsqueeze(1))
                acc_all+=acc.numpy()[0]
                num+=1
            # if out[0] == cls:
                # right +=1
        # print("right",right)
        return acc_all/num
    def main(self):
        if self.is_train:
            self.train()
        else:
            self.test()
    def train(self):
        stgcn = STGCN_framework()
        valid_loader = paddle.io.DataLoader(self.valid_dataset,batch_size=self.bs,shuffle =True,drop_last=True)
        if self.load_pretrain_model:
            stgcn.set_state_dict(paddle.load(self.load_pretrain_model))
        crossEntropyLoss =nn.CrossEntropyLoss()
        scheduler_G = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate =0.05, T_max =60, eta_min=0, last_epoch=- 1, verbose=False)
        optimizer = paddle.optimizer.Momentum(learning_rate=scheduler_G, momentum=0.9, parameters=stgcn.parameters(), use_nesterov=False, weight_decay=1e-4, grad_clip=None, name=None)
        epoches =self.epoches
        i = 0

        v_acc_max = 0
        for epoch in range(epoches):
            print("epoch",epoch)
            for data in tqdm(self.data_loader):
                
                one_data,cls=data
                out = stgcn(one_data)
                optimizer.clear_grad()
                loss = crossEntropyLoss(out,cls)
                loss.backward()
                optimizer.step()


                self.log_writer.add_scalar(tag='train/loss', step=i, value=loss.numpy()[0])


                if i%100 == 3:
                    print("loss",loss.numpy()[0],v_acc_max)
                    
                i+=1
                # break

            if epoch%2 == 0:
                stgcn.eval()
                v_acc = self.valid_accurary(valid_loader,stgcn)
                stgcn.train()
                print("epoch loss",loss.numpy()[0],v_acc)
                self.log_writer.add_scalar(tag='train/v_acc', step=i, value=v_acc)
                if v_acc > v_acc_max:
                    v_acc_max = v_acc
                    save_param_path_model = os.path.join(self.output_model_dir, 'Gmodel_state'+str(v_acc_max)+'.pdparams')
                    paddle.save(stgcn.state_dict(), save_param_path_model)

            scheduler_G.step()
            # break
        pass

    def test(self):
        stgcn = STGCN_framework()
        stgcn.set_state_dict(paddle.load(self.load_pretrain_model))
        valid_acc = self.valid_accurary(self.data_loader,stgcn)# 0.52
        print("测试集上准确率为",valid_acc)
        return valid_acc
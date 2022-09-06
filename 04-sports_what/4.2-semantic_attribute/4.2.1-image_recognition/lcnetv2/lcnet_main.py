import paddle
import paddle.nn as nn
from lcnet_framework import PPLCNetV2_model
from dataset import Dataset
import os
from tqdm import tqdm
from paddle.regularizer import L2Decay
from visualdl import LogWriter
class LCNet_main():
    def __init__(self,args):
        self.log_writer = LogWriter(args.output_log_dir)
        self.bs = args.BATCH_SIZE
        self.class_num = args.class_num
        self.use_pretrained = args.use_pretrained
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
            for one in tqdm(valid_loader):
                img_data,cls=one
                img_data = img_data.astype("float32")/127.5-1
                img_data =paddle.transpose(x=img_data,perm=[0,3,1,2])
                out = classifer_net(img_data)
                acc = paddle.metric.accuracy(out,cls.unsqueeze(1))
                acc_all+=acc.numpy()[0]
                num+=1
                if num == 10 and self.is_train== True:
                    break
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
        model = PPLCNetV2_model(class_num=self.class_num,use_pretrained = self.use_pretrained)
        valid_loader = paddle.io.DataLoader(self.valid_dataset,batch_size=self.bs,shuffle =True,drop_last=True)
        if self.load_pretrain_model:
            model.set_state_dict(paddle.load(self.load_pretrain_model))
        crossEntropyLoss =nn.CrossEntropyLoss()
        scheduler_G = paddle.optimizer.lr.StepDecay(learning_rate=0.002, step_size=3, gamma=0.8, verbose=True)
        optimizer = paddle.optimizer.Momentum(learning_rate=scheduler_G, momentum=0.9, parameters=model.parameters(),weight_decay=L2Decay(0.00004))
        epoches =self.epoches

        i = 0

        v_acc_max = 0
        for epoch in range(epoches):
            # print("epoch",epoch)
            for data in tqdm(self.data_loader):
                
                img_data,cls=data
                img_data = img_data.astype("float32")/127.5-1
                img_data =paddle.transpose(x=img_data,perm=[0,3,1,2])
                out = model(img_data)
                optimizer.clear_grad()
                loss = crossEntropyLoss(out,cls)
                loss.backward()
                optimizer.step()


                self.log_writer.add_scalar(tag='train/loss', step=i, value=loss.numpy()[0])


                if i%100 == 3:
                    print("loss",loss.numpy()[0],v_acc_max)
                    
                i+=1
                # break
                # scheduler_G.step()
            if epoch%1 == 0:
                model.eval()
                v_acc = self.valid_accurary(valid_loader,model)
                model.train()
                print("epoch loss",loss.numpy()[0],v_acc)
                self.log_writer.add_scalar(tag='train/v_acc', step=i, value=v_acc)
                if v_acc > v_acc_max:
                    v_acc_max = v_acc
                    save_param_path_model = os.path.join("model", 'Gmodel_state'+str(v_acc_max)+'.pdparams')
                    paddle.save(model.state_dict(), save_param_path_model)

            scheduler_G.step()

    def test(self):
        model = PPLCNetV2_model(class_num=self.class_num,use_pretrained = self.use_pretrained)
        model.set_state_dict(paddle.load(self.load_pretrain_model))
        valid_acc = self.valid_accurary(self.data_loader,model)# 0.52
        print("测试集上准确率为",valid_acc)
        return valid_acc
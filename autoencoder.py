import numpy as np
import torch, torchvision
import torch.nn.functional as F


d2_data=np.load('2D_data_dataset.npy')

d2_data=torch.from_numpy(d2_data)
d2_data=torch.unsqueeze(d2_data,1)
d2_data=d2_data.type(torch.float)
train_loader=torch.utils.data.DataLoader(d2_data,batch_size=32,shuffle=True)

class autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(1,3,kernel_size=3,padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.co1_bn=torch.nn.BatchNorm2d(3)
        self.conv2=torch.nn.Conv2d(3,8,kernel_size=3,padding=1)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.co2_bn=torch.nn.BatchNorm2d(8)
        self.conv3=torch.nn.Conv2d(8,8,kernel_size=3,padding=1)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.co3_bn=torch.nn.BatchNorm2d(8)
        self.conv4=torch.nn.Conv2d(8,8,kernel_size=3,padding=1)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        self.co4_bn=torch.nn.BatchNorm2d(8)
        self.dense=torch.nn.Linear(10760,1024)
        self.deco1=torch.nn.ConvTranspose2d(1024,512,kernel_size=(5,5))
        torch.nn.init.kaiming_normal_(self.deco1.weight)
        self.deco1_bn=torch.nn.BatchNorm2d(512)
        self.deco2=torch.nn.ConvTranspose2d(512,256,kernel_size=(5,5))
        torch.nn.init.kaiming_normal_(self.deco2.weight)
        self.deco2_bn=torch.nn.BatchNorm2d(256)
        self.deco3=torch.nn.ConvTranspose2d(256,128,kernel_size=(3,3))
        torch.nn.init.kaiming_normal_(self.deco3.weight)
        self.deco3_bn=torch.nn.BatchNorm2d(128)
        self.deco4=torch.nn.ConvTranspose2d(128,64,kernel_size=(3,3))
        torch.nn.init.kaiming_normal_(self.deco4.weight)
        self.deco4_bn=torch.nn.BatchNorm2d(64)
        self.deco5=torch.nn.ConvTranspose2d(64,16,kernel_size=(3,3))
        torch.nn.init.kaiming_normal_(self.deco5.weight)
        self.deco5_bn=torch.nn.BatchNorm2d(16)
        self.deco6=torch.nn.ConvTranspose2d(16,3,kernel_size=(3,3))
        torch.nn.init.kaiming_normal_(self.deco6.weight)
        self.deco6_bn=torch.nn.BatchNorm2d(3)
        self.deco7=torch.nn.ConvTranspose2d(3,1,kernel_size=(3,3))
        torch.nn.init.kaiming_normal_(self.deco7.weight)

        self.upsample1=torch.nn.Upsample(size=(5,11),mode='bilinear')
        self.upsample2=torch.nn.Upsample(size=(10,44),mode='bilinear')
        self.upsample3=torch.nn.Upsample(size=(12,176),mode='bilinear')
        self.upsample4=torch.nn.Upsample(size=(14,350),mode='bilinear')
        self.upsample5=torch.nn.Upsample(size=(18,650),mode='bilinear')
        self.upsample6=torch.nn.Upsample(size=(20,1077),mode='bilinear')
        self.upsample7=torch.nn.Upsample(size=(20,1077),mode='bilinear')


    def forward(self,x):

        out=F.avg_pool2d(F.leaky_relu(self.co1_bn(self.conv1(x))),2)
        out=F.avg_pool2d(F.leaky_relu(self.co2_bn(self.conv2(out))),2)
        out=F.leaky_relu(self.co3_bn(self.conv3(out)))
        out=F.leaky_relu(self.co4_bn(self.conv4(out)))
        out=out.view(out.size(0),-1)
        out=F.leaky_relu(self.dense(out))
        out=out.view(-1,1024,1,1)
        out=F.leaky_relu(self.deco1_bn(self.deco1(out)))
        out=self.upsample1(out)
        out=F.leaky_relu(self.deco2_bn(self.deco2(out)))
        out=self.upsample2(out)
        out=F.leaky_relu(self.deco3_bn(self.deco3(out)))
        out=self.upsample3(out)
        out=F.leaky_relu(self.deco4_bn(self.deco4(out)))
        out=self.upsample4(out)
        out=F.leaky_relu(self.deco5_bn(self.deco5(out)))
        out=self.upsample5(out)
        out=F.leaky_relu(self.deco6_bn(self.deco6(out)))
        out=self.upsample6(out)
        out=self.deco7(out)
        out=self.upsample7(out)
        return out

device=torch.device('cuda')
model=autoencoder()
model.load_state_dict(torch.load('model_best_1024.pth'))
model=model.to(device)

criterion=torch.nn.MSELoss()
optim=torch.optim.Adam(model.parameters(),lr=1e-03)
#lambda1=lambda epoch:0.95**epoch
#scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lr_lambda=lambda1)
#encodings=np.zeros((512,1000))

for i in range(1000):
    epoch_loss=0
    for (b_idx,tensor) in enumerate(train_loader):
        model.train()
        input=tensor.to(device)
        output=model(input)
        #out=output.detach().cpu().numpy()
        #out=np.transpose(out)
        #encodings[:,b_idx]=out[0,:]
        loss=criterion(output,input)
        loss.backward()
        optim.step()
        epoch_loss+=loss
    print("Epoch:{},Loss:{}".format(i,epoch_loss))

torch.save(model.state_dict(),'model.pth')
#np.save('/code/CIS520/FinalProject/encodings_512.npy',encodings)

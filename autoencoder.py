import numpy as np
import torch, torchvision
import torch.nn.functional as F

d2_data=np.load('/media/rahul/927f87f9-ee25-4f34-a75c-7d5e3e893ad3/Penn_Courses/CIS520/FinalProject/2D_data_dataset.npy')
d2_data=torch.from_numpy(d2_data)
d2_data=torch.unsqueeze(d2_data,1)
d2_data=d2_data.type(torch.float)
train_loader=torch.utils.data.DataLoader(d2_data,batch_size=1,shuffle=True)

class autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(1,3,kernel_size=3,padding=1)
        self.conv2=torch.nn.Conv2d(3,8,kernel_size=3,padding=1)
        self.dense=torch.nn.Linear(10760,520)
        self.deco1=torch.nn.ConvTranspose2d(520,8,kernel_size=(5,11))
        self.deco2=torch.nn.ConvTranspose2d(8,3,kernel_size=(2,4),stride=(2,4))
        self.deco3=torch.nn.ConvTranspose2d(3,3,kernel_size=(2,4),stride=(2,4))
        self.deco4=torch.nn.ConvTranspose2d(3,3,kernel_size=(1,4),stride=(1,1))
        self.deco5=torch.nn.ConvTranspose2d(3,3,kernel_size=(1,2),stride=(1,2))
        self.deco6=torch.nn.ConvTranspose2d(3,3,kernel_size=(1,2),stride=(1,1))
        self.deco7=torch.nn.ConvTranspose2d(3,1,kernel_size=(1,3),stride=(1,3))

    def forward(self,x):
        out=F.avg_pool2d(F.relu(self.conv1(x)),2)
        out=F.avg_pool2d(F.relu(self.conv2(out)),2)
        out=out.view(out.size(0),-1)
        out=F.relu(self.dense(out))
        out=out.view(-1,520,1,1)
        out=F.relu(self.deco1(out))
        out=F.relu(self.deco2(out))
        out=F.relu(self.deco3(out))
        out=F.relu(self.deco4(out))
        out=F.relu(self.deco5(out))
        out=F.relu(self.deco6(out))
        out=F.relu(self.deco7(out))
        return out

device=torch.device('cuda')
model=autoencoder().to(device)

criterion=torch.nn.MSELoss()
optim=torch.optim.Adam(model.parameters())


for i in range(500):
    epoch_loss=0
    for (b_idx,tensor) in enumerate(train_loader):
        model.train()
        input=tensor.to(device)
        output=model(input)
        loss=criterion(output,input)
        loss.backward()
        optim.step()
        epoch_loss+=loss
    print("Epoch:{},Loss:{}".format(i,epoch_loss))

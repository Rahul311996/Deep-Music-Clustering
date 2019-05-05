import torch
import torchvision
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

#d2_data=np.load('data_3d_min_max.npy')
d2_data=np.load('data_train_3d.npy')
#d2_data=np.load('2D_data_dataset.npy')
#d2_data_new=np.zeros((700,20,1077))



class siamese_network(torch.nn.Module):

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
        self.dense=torch.nn.Linear(10760,128)

    def forward(self,x):
        out=F.avg_pool2d(F.leaky_relu(self.co1_bn(self.conv1(x))),2)
        out=F.avg_pool2d(F.leaky_relu(self.co2_bn(self.conv2(out))),2)
        out=F.leaky_relu(self.co3_bn(self.conv3(out)))
        out=F.leaky_relu(self.co4_bn(self.conv4(out)))
        out=out.view(out.size(0),-1)
        out=self.dense(out)
        return out


class contrastive_loss(torch.nn.Module):
    #code referred from https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
    def __init__(self,margin=1):
        super(contrastive_loss,self).__init__()
        self.margin=margin

    def sanity_check(self,in_types):
        assert len(in_types) == 3

        x0_type,x1_type,y_type=in_types
        assert x0_type.size()==x1_type.shape
        assert x1_type.size()[0]==y_type.shape[0]
        assert x1_type.size()[0]>0
        assert x0_type.dim()==2
        assert x1_type.dim()==2
        assert y_type.dim()==1

    def forward(self, x0, x1, y):
        self.sanity_check((x0, x1, y))

        # euclidian distance
        diff=x0-x1
        dist_sq=torch.sum(torch.pow(diff,2),1)
        dist=torch.sqrt(dist_sq)

        mdist=self.margin-dist
        dist=torch.clamp(mdist, min=0.0)
        loss=y*dist_sq+(1-y)*torch.pow(dist,2)
        loss=torch.sum(loss)/2.0/x0.size()[0]
        return loss

def create_pairs(data):
    x0_data=[]
    x1_data=[]
    label=[]

    n=68
    for d in range(10):
        for i in range(n):
            x0,x1=data[d*70+i],data[d*70+(i+1)]
            x0_data.append(x0)
            x1_data.append(x1)
            label.append(1)

            inc=random.randrange(1,10)
            dn=(d+inc)%10
            x0,x1=data[d*70+i],data[dn*70+(i)]
            x0_data.append(x0)
            x1_data.append(x1)
            label.append(0)

    x0_data=np.array(x0_data,dtype=np.float32)
    x0_data=x0_data.reshape([-1,1,20,1077])
    x1_data=np.array(x1_data,dtype=np.float32)
    x1_data=x1_data.reshape([-1,1,20,1077])

    label=np.array(label,dtype=np.float32)

    return x0_data,x1_data,label

class custom_dataset(Dataset):

    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return (self.x0[index],
                self.x1[index],
                self.label[index])

    def __len__(self):
        return self.size

def create_iterator(data):
    x0,x1,label=create_pairs(data)
    ret=custom_dataset(x0,x1,label)
    return ret

device=torch.device('cuda')
model=siamese_network().to(device)

criterion=contrastive_loss()
optimiser=torch.optim.Adam(model.parameters(),lr=1e-4)

train_iter=create_iterator(d2_data)
train_loader=torch.utils.data.DataLoader(train_iter,batch_size=8,shuffle=True)
model.load_state_dict(torch.load('siamese_new.pth'))

'''epochs=300
for i in range(epochs):
    epoch_loss=0
    model.train()
    for b_idx,(input1,input2,label) in enumerate(train_loader):
        input1=input1.to(device)
        input2=input2.to(device)
        label=label.to(device)

        encoding1=model(input1)
        encoding2=model(input2)
        loss=criterion(encoding1,encoding2,label)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss+=loss
    print("Epoch:{},Loss:{}".format(i,loss))

torch.save(model.state_dict(),'siamese_new.pth')'''

d2_data=np.load('2D_data_dataset.npy')
d2_data=torch.from_numpy(d2_data)
d2_data=torch.unsqueeze(d2_data,1)
d2_data=d2_data.type(torch.float)
train_loader=torch.utils.data.DataLoader(d2_data,batch_size=1,shuffle=False)

encodings=np.ones((128,1000),dtype=np.float32)
for (b_idx,tensor) in enumerate(train_loader):
    model.eval()
    input=tensor.to(device)
    out=model(input)
    out=out.cpu().detach().numpy().T
    print(b_idx)
    encodings[:,b_idx]=out[:,0]

np.save('siamese_encodings.npy',encodings)

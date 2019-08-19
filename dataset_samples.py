import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import warnings
import pandas as pd
import os
import numpy as np
import nibabel as nib
warnings.filterwarnings("ignore")

class FacelandmarksDataset(Dataset):
    def __init__(self,csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx,1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1,2)
        sample = {'image':image, 'landmarks': landmarks}

        # if self.transform:
        #     sample  = transform(sample)
        return sample

class OCT_Datasets(Dataset):
    def __init__(self,root_dir):
        self.namelist = os.listdir(root_dir+'original_images/')
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, idx):
        image_path = self.root_dir+'original_images/'+self.namelist[idx]
        label_name = self.namelist[idx].rstrip('.img')+'_labelMark'
        label_path = self.root_dir+'label_images/'+label_name
        image_list = os.listdir(image_path)
        sample_images = []
        sample_labels = []       
        for i, name in enumerate(image_list):
            image = io.imread(image_path+'/'+name)
            label = io.imread(label_path+'/'+name)
            sample_images.append(image)
            sample_labels.append(label)
        sample = {'image': np.array(sample_images),'label':np.array(sample_labels)}
        return sample

class OCT_Datasets_2(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.namelist = []
        for root, _, files in os.walk(self.root_dir+'original_images/'):
            if len(files)>0:
                for i in range(len(files)):
                    self.namelist.append([root,files[i]])
        # self.namelist = self.namelist[:100]
    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.namelist[idx][0],self.namelist[idx][1])
        label_path = self.namelist[idx][0].replace('original_images','label_images').rstrip('.img')+'_labelMark'
        label_name = os.path.join(label_path,self.namelist[idx][1])
        sample_image = io.imread(image_name)
        sample_label = io.imread(label_name)
        sample = {'image': np.array(sample_image),'label':np.array(sample_label)}
        return sample



class Brain_dataset(Dataset):
    def __init__(self,name_file,rootdir,sample_range=[0,None],padding=None,reshape=None):
        self.sample_range = sample_range
        f = open(name_file, 'r')
        namelist=[]
        for line in f:
            name = line.split('.')[0]
        #     print(name)
            namelist.append(name)
        self.namelist = namelist[sample_range[0]:sample_range[1]]
        self.rootdir = rootdir
        self.post = ['.nii','_4thVentrical.nii','_thalamus_right.nii','_thalamus_left.nii','_brainstem.nii' ]
        self.padding = padding
        self.reshape = reshape
    def __len__(self):
        return len(self.namelist)
    def __getitem__(self,idx):
        original_data_path = self.rootdir+self.namelist[idx]+self.post[0]
        label_data_path = [self.rootdir+self.namelist[idx]+poster for poster in self.post[1:]]
        original_data = nib.load(original_data_path).get_data()
        if idx in range(1,52):
            original_data = original_data.swapaxes(1,2).swapaxes(0,1)[:,::-1,:]
        label_data = []
        for i in range(4):
            mask_data = nib.load(label_data_path[i]).get_data()
            if idx in range(1,52):
                mask_data = mask_data.swapaxes(1,2).swapaxes(0,1)[:,::-1,:]
            label_data.append(mask_data)

        if self.padding:
            original_data = self.padding(original_data)
            label_data = [self.padding(mask_data) for mask_data in label_data]
        if self.reshape:
            original_data = self.reshape(original_data)
            label_data = [self.reshape(mask_data) for mask_data in label_data]
        return {'image':original_data,'label':np.array(label_data)}


class Zero_padding(object):
    def __init__(self, output_size):
        assert isinstance(output_size,(int,list))
        self.output_size = output_size
    def __call__(self,sample):
        if isinstance(self.output_size, int):
            length,width,height = self.output_size, self.output_size, self.output_size
        else:
            length,width,height = self.output_size[0],self.output_size[1],self.output_size[2]
        l,w,h = sample.shape
        dl,dw,dh = (length-l)//2,(width-w)//2,(height-h)//2
        dtype = type(sample[0,0,0])
        pad_image = np.zeros([length,width,height],dtype=dtype)
        pad_image[dl:dl+l,dw:dw+w,dh:dh+h]=sample
        return pad_image




def main():
    zero_padding = Zero_padding(256)
    brain_dataset = Brain_dataset('/DATABrainSeg/subject_list_good.txt' ,\
                                    '/DATA/BrainSeg/',padding=zero_padding)
    print(brain_dataset[10]['image'].shape)
    print(brain_dataset[10]['label'][0].shape)
    print(np.array(brain_dataset[10]['label']).shape)
    
    # OCT_sets = OCT_Datasets_2('D:/datasets_many/AIchallenger2018/ai_challenger_fl2018_trainingset/Edema_trainingset/')
    # print(len(OCT_sets))
    # print(OCT_sets[20]['image'].shape)
    # dataloader = DataLoader(OCT_sets, batch_size = 4, shuffle = True)
    # for i, sample_batched in enumerate(dataloader):
    #     print(i)
    #     print(sample_batched['image'].size())
    
    # face_dataset = FacelandmarksDataset('D:/datasets_many/faces/faces/face_landmarks.csv',\
    #                                 'D:/datasets_many/faces/faces/')
    # print(len(face_dataset))
    # dataloader = DataLoader(face_dataset, batch_size = 4, shuffle = True)
    # for i, sample_batched in enumerate(dataloader):
    #     print(i)
    #     print(sample_batched['image'].shape)
    # face_sample = face_dataset[10]
    # print(face_sample['image'].shape, face_sample['landmarks'].shape)

#     dir_path = 'd:/dataset_many/AIchallenger2018'
#     data_transform = transforms.Compose([
#             transforms.RandomSizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#         ])
#     hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                                 transform=None)
#     dataset_loader = torch.utils.data.DataLoader(hymenptera_dataset,
#                                                 batch_size=4, shuffle=True,
#                                                 num_workers=4)


if __name__=='__main__':
    main()
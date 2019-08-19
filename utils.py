import numpy as np
import tensorflow as tf
from keras import backend as K
import nibabel as nib
import os
class params(): 
    name_file = '/DATA/BrainSeg/subject_list_good.txt' 
    nii_directory = '/DATA/BrainSeg/'
    save_dirpath = '/DATA/BrainSeg/'
    post= ['.nii','_4thVentrical.nii','_thalamus_right.nii','_thalamus_left.nii','_brainstem.nii' ]

def central_zero_padding(vortex,length,width,height,dtype):
    '''
    vortex should be a 3-D shape
    return an array center on original array   
    dtype means how mang space to allocate 
    '''
    l,w,h = vortex.shape
    dl,dw,dh = (length-l)//2,(width-w)//2,(height-h)//2
#     print(l,w,h)
#     print(dl,dw,dh)
    pad_image = np.zeros([length,width,height],dtype=dtype)
    pad_image[dl:dl+l,dw:dw+w,dh:dh+h]=vortex
    return pad_image
def read_list():
    '''
    read the total list of brain files' name
    '''
    name_file = '/DATA/250/wyshi/Data/BrainSeg/subject_list_good.txt' 
    f = open(name_file, 'r')
    namelist=[]
    for line in f:
        name = line.split('.')[0]
        namelist.append(name)
    return namelist

def load_data(dirfile='/data/BrainSeg/', group=1 ):
    '''
    load data with 4D array (brain_num, x,y,z)
    '''
    if type(group)==int or type(group)==str:
        brain=np.load(dirfile+'brain'+str(group)+'.npy')
        label=np.load(dirfile+'label'+str(group)+'.npy')
        print('orignal brain shape: ',brain.shape)
        print('orignal mask shape: ',label.shape)
    if type(group)==list:
        Target_length,Target_width,Target_height = 256,256,256
        brain = np.zeros([0,Target_length,Target_width,Target_height,1],dtype='int32')
        label = np.zeros([0,Target_length,Target_width,Target_height,4],dtype=np.bool)
        for num in group:
            brain2=np.load(dirfile+'brain'+str(num)+'.npy')
            label2=np.load(dirfile+'label'+str(num)+'.npy')
            brain = np.concatenate((brain, brain2), axis = 0)
            label = np.concatenate((label, label2), axis = 0)
        print('orignal brain shape: ',brain.shape)
        print('orignal mask shape: ',label.shape)
    return brain,label
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def binary_crossentropy2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
def check_list(Y_data):
    '''
    shape(num,x,y,z)
    '''
    total_list = []
    for i in range(len(Y_data)):
        list_z = []
        for zz in range(Y_data.shape[3]):
            if True in Y_data[i,:,:,zz,:]:
                list_z.append(zz)
        range_single = np.arange(np.min(list_z)+(i-1)*256,np.max(list_z)+35+(i-1)*256)
        total_list.extend(range_single)
    return total_list
def reshape_dimension(X_train,Y_train,axis,zero_remove):
    '''
    accumulate in 'aixs' dimension return an array lost its 'axis' dimension.
    if axis = 2
    Input_shape(brain_num, x, y,z)
    Output_shape(z*brain_num,x,y)
    '''
    if zero_remove:
        use_list = check_list(Y_train)
        print(len(use_list))
    X_train = np.concatenate([brain for brain in X_train],axis=axis).swapaxes(0,axis)
    Y_train = np.concatenate([brain for brain in Y_train],axis=axis).swapaxes(0,axis)
    print('after reshape: ',X_train.shape)
    print('after reshape:',Y_train.shape)

        # use_list=[]
        # for j, slice in enumerate(X_train):
        #     if True in slice:  
        #     # if np.max(slice)>0:
        #         use_list.append(j)
    if zero_remove:
        X_train = X_train[use_list]
        Y_train = Y_train[use_list]
    return X_train,Y_train

def seperate_data(data,label,num):
    a=[i for i in range(len(data))]
    np.random.shuffle(a)
    X_train = data[a[:num],:,:,:]
    Y_train = label[a[:num],:,:,:]
    X_val = data[a[num:],:,:,:]
    Y_val = label[a[num:],:,:,:]
    return X_train,Y_train,X_val,Y_val


def cal_miou(ground_truth,prediction,threshold):
    # ground_truth,prediction = ground_truth.swapaxes(0,-1),ground_truth.swapaxes(0,-1)
    score=[]
    for i in range(4):
        tmp=prediction[:,:,:,i]>threshold[i]
        a= np.bitwise_and(tmp,ground_truth[:,:,:,i]).sum()/np.bitwise_or(tmp,ground_truth[:,:,:,i]).sum()
        score.append(a)
    return score

def find_optimal_threshold(ground_truth, prediction,key):
    '''
    ground_truth shape (:,:,:,num_classes)
    prediction shape (:,:,:,num_classes)
    '''
    num_class = prediction.shape[-1]
    score_list, optimal_threshold_list = [],[]
    class_range = [[np.min(prediction[:,:,:,i]),np.max(prediction[:,:,:,i])] for i in range(num_class)] 
    for i,dimrange in enumerate(class_range):
        score, optimal_threshold=0,0
        for threshold in np.linspace(dimrange[0],dimrange[1],20):
            tmp = prediction>threshold
            if key =='accuracy' or key=='acc':
                a=(tmp==ground_truth).sum()/ground_truth.size
            elif  key == 'iou':
                
                a = np.bitwise_and(tmp,ground_truth).sum()/np.bitwise_or(tmp,ground_truth).sum()
            else:
                print('Cannot find the corresponding key')
                return None
            if score < a :
                score = a
                optimal_threshold = threshold
        print('Numclass:',i)
        print('optimal_ '+key+': ',score,' optimal_threshold: ',optimal_threshold)
        score_list.append(score)
        optimal_threshold_list.append(optimal_threshold)
    return score_list,optimal_threshold_list



def read_standard_data(num):
    Target_length,Target_width,Target_height=256,256,256
    path = params.nii_directory
    namelist = read_list()
    filename = namelist[num]
    post = params.post
    X_data = nib.load(path+filename+post[0]).get_data()
    affine_matrix = nib.load(path+filename+post[0]).affine
    if num in range(1,52):
        X_data = X_data.swapaxes(1,2).swapaxes(0,1)[:,::-1,:]
    X_data = central_zero_padding(X_data,Target_length,Target_width,Target_height,'int32')
    X_data = np.expand_dims(X_data,0)
    X_data = np.expand_dims(X_data,-1)
    mask = np.zeros([Target_length,Target_width,Target_height,0],dtype=np.bool)
    for j,poster in enumerate(post[1:]):
        data =  nib.load(os.path.join(path+filename+poster)).get_data()
        if num in range(1,52):
            data = data.swapaxes(1,2).swapaxes(0,1)[:,::-1,:]
        data = central_zero_padding(data,256,256,256,np.bool)
        data = np.expand_dims(data,-1)
        mask = np.concatenate((mask, data), axis = 3)
    mask = np.expand_dims(mask,0)
    return X_data,mask,filename,affine_matrix
def get_affine_maxtrix(path):
    img = nib.load(path)
    affine_matrix = img.affine
    return affine_matrix

def reverse_sample_trans(X_data,Y_data,pre,axis):
    '''
    after the reshape function, willing to put it in a normal ones.
    Have nothing to do with the sample's size originally
    Input(z,x,y,:,:)
    output(x,y,z,:,:)
    '''
    X_data = np.squeeze(X_data)
    Y_data = np.squeeze(Y_data)
    pre = np.squeeze(pre)
    X_data = np.squeeze(np.expand_dims(X_data,axis+1).swapaxes(axis+1,0))
    Y_data = np.squeeze(np.expand_dims(Y_data,axis+1).swapaxes(axis+1,0))
    pre = np.squeeze(np.expand_dims(pre,axis+1).swapaxes(axis+1,0))
    # if num in range(1,52):
    #     X_data = X_data[:,::-1,:].swapaxes(0,1).swapaxes(1,2)
    #     Y_data = Y_data[:,::-1,:,:].swapaxes(0,1).swapaxes(1,2)
    return X_data,Y_data,pre

def array2nii(X_data,Y_data,pre,affine_matrix,savename):
    '''
    filename should not includ dirpath and post '.nii'
    X_data should be with shape of 3-D(x,y,z), and Y_data shape is (x,y,z,channels)
    '''
    if len(X_data.shape)>4:
        print('X_data dim is too many')
    elif len(X_data.shape)==4:
        X_data = X_data[:,:,:,0]
    dirpath = params.save_dirpath
    savepath = os.path.join(dirpath,savename)
    # affine_matrix = np.diag([1,1,1,1])
    post = params.post 
    img = nib.Nifti1Image(X_data, affine_matrix)
    img.to_filename(savepath+post[0])
    y_true = np.zeros([256,256,256,4],dtype='int32')
    for i,poster in enumerate(post[1:]):
        y_true[:,:,:,i] = Y_data[:,:,:,i] 
        img = nib.Nifti1Image(y_true[:,:,:,i], affine_matrix)
        img.to_filename(savepath+poster)
        img = nib.Nifti1Image(pre[:,:,:,i], affine_matrix)
        img.to_filename(savepath+'_pre'+poster)
    


# def show_result(X_train,Y_train,preds_val_t):
#     import matplotlib.pyplot as plt
#     # sample_id = 55
#     if x_train.ndim == 3:
#         x_train = x_train[:,:,0]
#     fig = plt.figure()
#     axe = fig.add_subplot(2,2,1)
#     plt.imshow(X_train)
#     for j in range(len(Y_train[0,0,:])):
#         plt.imshow(Y_train[:,:,i])
#     axe2 = fig.add_subplot(2,2,2)
#     plt.imshow(X_train)
#     for j in range(len(preds_val_t[0,0,:])):
#         plt.imshow(preds_val_t[:,:,i])
#     plt.imsave('test.jpg')
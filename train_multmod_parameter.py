# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, EnsureType, RandFlip
from sklearn.model_selection import KFold
# v11.02


from data import *
from utils import *
def main():
    with open('train_parameter_026.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
    print(json_data['GPU'])

    epoch_num = json_data['epoch']
    val_interval = json_data['val_per_epoch']
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #修改image路径
    taskname = json_data['taskname']
    modules_list = json_data['modules']

    sss = ''
    taskname = taskname+ '_' + sss.join(modules_list)
    print(taskname+"+++++++++++++++++++++++++++++++++++++++++++++++++++")

    save_dir = Path(json_data['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    fold = json_data['fold']
    task_dir = (save_dir/taskname/f"fold_{fold}")
    task_dir.mkdir(parents=True, exist_ok=True)
    t_batch_size = json_data['batch_size']
    data_path = Path(json_data['data_path'])
    num_class = json_data['class_num']
    modules_num = len(modules_list)
    dataclass = seg_dataset_mult(data_path , modules_list)
    resize_x = json_data['resize_x']
    resize_y = json_data['resize_y']
    resize_z = json_data['resize_z']
    with open(task_dir/'train_parameter.json','w') as file_obj:
        json.dump(json_data,file_obj,indent=4)



    #创建图像list
    image_list= []
    for i in range(modules_num):
        image_list.append(dataclass.data_list[i])


    # 2 binary labels for classification: 
    labels = dataclass.get_label_list()

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((resize_x, resize_y, resize_z)), RandRotate90(prob=0.5, spatial_axes=(0, 1)),RandFlip(prob=0.5), EnsureType()])
    val_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((resize_x, resize_y, resize_z)), EnsureType()])

    # Define image dataset, data loader
    # check_ds = ImageDataset(image_files=image_list[0], labels=labels, transform=train_transforms)
    # check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    # im, label = monai.utils.misc.first(check_loader)
    # print(type(im), im.shape, label)

    # val_num = 0-int(len(image_list[0])*json_data['val_percentage'])
    # print(f"data lenght : {len(image_list[0])} and the val data {val_num}")

    # kfold cal
    print(fold)
    kf = KFold(n_splits=5,shuffle=True, random_state=980)  # 初始化KFold
    train_index_k = [train_index for train_index , test_index in kf.split(image_list[0])][fold]  # 调用split方法切分数据
    val_index_k = [test_index for train_index , test_index in kf.split(image_list[0])][fold]  # 调用split方法切分数据
    print(f"Train data lenght : {len(train_index_k)} and the val data {len(val_index_k)}")
    print(train_index_k)
    print(val_index_k)
    kfold_t = {}
    kfold_t['train'] = train_index_k.tolist()
    kfold_t['val'] = val_index_k.tolist()
    with open(task_dir/'kflod.json','w') as file_obj:
        json.dump(kfold_t,file_obj,indent=4)

    # create a training data loader
    train_ds = []
    train_loader = []
    for i in range(modules_num):
        
        train_ds.append(ImageDataset(image_files=get_index(image_list[i],train_index_k), labels=get_index(labels,train_index_k), transform=train_transforms))
        if i == 0:
            s=torch.randperm(len(train_ds[0])).tolist()
            sampler=RandomSampler(train_ds[0],s1 = s) 

        train_loader.append(DataLoader(train_ds[i], batch_size=t_batch_size, shuffle=False,sampler =sampler, num_workers=2, pin_memory=torch.cuda.is_available()))

    print(train_loader[0])

    # create a validation data loader
    val_ds = []
    val_loader = []
    for i in range(modules_num):
        val_ds.append(ImageDataset(image_files=get_index(image_list[i],val_index_k), labels=get_index(labels,val_index_k), transform=val_transforms))
        val_loader.append(DataLoader(val_ds[i], batch_size=t_batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()))
    

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=modules_num, out_channels=num_class).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    metric_values_t = list()
    writer = SummaryWriter(str(task_dir),comment=taskname)
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        num_correct_t = 0.0
        metric_count_t = 0

        for batch_data in zip(*train_loader):
            #print(batch_data)
            batch = []
            for j in range(len(batch_data)):
                batch.append(batch_data[j][0])
            tt = torch.cat(batch,dim=1) 
           

            step += 1
            inputs, train_labels = tt.to(device), batch_data[0][1].to(device)
            #print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)

            value_t = torch.eq(outputs.argmax(dim=1), train_labels)
            metric_count_t += len(value_t)
            num_correct_t += value_t.sum().item()


            
            loss = loss_function(outputs, train_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds[0]) // train_loader[0].batch_size
            #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            #writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        metric_t = num_correct_t / metric_count_t
        metric_values_t.append(metric_t)
        writer.add_scalar("train_accuracy", metric_t, epoch + 1)


        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        writer.add_scalar("epoch_loss", epoch_loss, epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} train_accuracy: {metric_t:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                
                metric =val_metric(model, val_loader)
                metric_values.append(metric)

                # 如果验证集评估指标更好，保存模型，保存训练集和验证集的结果
                if metric >= best_metric:
                    
                    meta = metric_cal(model, train_loader)
                    meta_val = metric_cal(model, val_loader)
                    meta['val'] = meta_val
                    with open(task_dir/'train.json','w') as file_obj:
                        json.dump(meta,file_obj,indent=4)



                    best_metric_t = metric_t
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metric_model_save = task_dir/"best_metric_model_classification3d.pth"
                    torch.save(model.state_dict(), str(best_metric_model_save))
                    print(f"Saved new best metric {metric:.4f} model at epoch {epoch}")

                    output(model, train_loader)


                print(
                    "{} current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} best train accuracy: {:.4f} at epoch {}".format(
                        taskname, epoch + 1, metric, best_metric,best_metric_t, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", metric, epoch + 1)
    latest_model_save = task_dir/"latest_metric_model_classification3d.pth"

    torch.save(model.state_dict(), latest_model_save)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

    meta['best_val_acc'] = best_metric
    meta['best_epoch'] = best_metric_epoch
    with open(task_dir/'train_end.json','w') as file_obj:
        json.dump(meta,file_obj,indent=4)


if __name__ == "__main__":

    main()
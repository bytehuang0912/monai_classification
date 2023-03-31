import os

import pandas
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score , recall_score , confusion_matrix
from sklearn.metrics import precision_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def val_metric(model, dataloder):
    num_correct = 0.0
    metric_count = 0
    for batch_datav in zip(*dataloder):
        batchv = []
        for k in range(len(batch_datav)):
            batchv.append(batch_datav[k][0])
        vv = torch.cat(batchv,dim=1) 

        val_images, val_labels = vv.to(device), batch_datav[0][1].to(device)
        val_outputs = model(val_images)
        value = torch.eq(val_outputs.argmax(dim=1), val_labels)
        metric_count += len(value)
        num_correct += value.sum().item()
    metric = num_correct / metric_count

    return metric

def val_metric_multiclass(model, dataloder):
    num_correct = 0.0
    metric_count = 0
    for batch_datav in zip(*dataloder):
        batchv = []
        for k in range(len(batch_datav)):
            batchv.append(batch_datav[k][0])
        vv = torch.cat(batchv,dim=1) 

        val_images, val_labels = vv.to(device), batch_datav[0][1].to(device)
        val_outputs = model(val_images)
        value = torch.eq(val_outputs.argmax(dim=1), val_labels)
        metric_count += len(value)
        num_correct += value.sum().item()
    metric = num_correct / metric_count

    return metric



def output(model, dataloder):
    pass 
    for batch_datav in zip(*dataloder):
        batchv = []
        for k in range(len(batch_datav)):
            batchv.append(batch_datav[k][0])
        vv = torch.cat(batchv,dim=1) 


def metric_mult(label_gt, label_predict):
    md = {}
    AUC = roc_auc_score(label_gt, label_predict)
    recall = recall_score(label_gt, label_predict, average='micro')
    tn, fp, fn, tp = confusion_matrix(label_gt, label_predict).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+ fn)
    accuracy = (tp+tn)/(tn+fp+fn+tp)

    return(accuracy, AUC, recall, specificity, sensitivity)
    
def metric_mult_mulclass(label_gt, label_predict):
    md = {}
    recall_micro = recall_score(label_gt, label_predict, average='micro')
    recall_macro = recall_score(label_gt, label_predict, average='macro')
    precision_micro = precision_score(label_gt, label_predict, average='micro')
    precision_macro = precision_score(label_gt, label_predict, average='macro')
    f1_score_micro = f1_score(label_gt, label_predict, average='micro')
    f1_score_macro = f1_score(label_gt, label_predict, average='macro')
    AUC = roc_auc_score(label_gt, label_predict, average='micro',multi_class='ovo')
    # tn, fp, fn, tp = confusion_matrix(label_gt, label_predict).ravel()
    # specificity = tn / (tn+fp)
    # sensitivity = tp/ (tp+ fn)
    # accuracy = (tp+tn)/(tn+fp+fn+tp)

    return(precision_micro, precision_macro, recall_micro, recall_macro, f1_score_micro, f1_score_macro)

def metric_cal(model, train_loader):
    # do train metrics
    outpus_list = []
    label_lst = []
    val_outputs_raw_list = []
    num_correctt = 0.0
    metric_countt = 0

    for batch_datat in zip(*train_loader):
        batcht = []
        for k in range(len(batch_datat)):
            batcht.append(batch_datat[k][0])
        vv = torch.cat(batcht,dim=1) 

        val_images, val_labels = vv.to(device), batch_datat[0][1].to(device)
        val_outputs_raw = model(val_images)
        val_outputs_raw_list.extend(val_outputs_raw.cpu().numpy().tolist())
        val_outputs = val_outputs_raw.argmax(dim=1)
        outpus_list.extend(val_outputs.tolist())
        value = torch.eq(val_outputs, val_labels)
        label_lst.extend(val_labels.tolist())
        metric_countt += len(value)
        num_correctt += value.sum().item()
    a,b,c,d,e = metric_mult(label_lst, outpus_list)
    print("**accuracy: {}, AUC: {}, recall: {}, specificity: {}, sensitivity: {}. ".format(a,b,c,d,e))
    meta = {}
    meta['raw_label'] = val_outputs_raw_list
    meta['outpus_list'] = outpus_list
    meta['label_lst'] = label_lst                    
    meta['accuracy'] = a
    meta['AUC'] = b
    meta['recall'] = c
    meta['specificity'] = d
    meta['sensitivity'] = e
    return(meta)
    
    with open(task_dir/'train.json','w') as file_obj:
        json.dump(meta,file_obj)

def metric_cal_mutclass(model, train_loader):
    # do train metrics
    outpus_list = []
    label_lst = []
    val_outputs_raw_list = []
    num_correctt = 0.0
    metric_countt = 0

    for batch_datat in zip(*train_loader):
        batcht = []
        for k in range(len(batch_datat)):
            batcht.append(batch_datat[k][0])
        vv = torch.cat(batcht,dim=1) 

        val_images, val_labels = vv.to(device), batch_datat[0][1].to(device)
        val_outputs_raw = model(val_images)
        val_outputs_raw_list.extend(val_outputs_raw.cpu().numpy().tolist())
        val_outputs = val_outputs_raw.argmax(dim=1)
        outpus_list.extend(val_outputs.tolist())
        value = torch.eq(val_outputs, val_labels)
        label_lst.extend(val_labels.tolist())
        metric_countt += len(value)
        num_correctt += value.sum().item()
    # a,b,c,d,e = metric_mult_mulclass(label_lst, outpus_list)
    # print("**accuracy: {}, AUC: {}, recall: {}, specificity: {}, sensitivity: {}. ".format(a,b,c,d,e))
    meta = {}
    meta['raw_label'] = val_outputs_raw_list
    meta['outpus_list'] = outpus_list
    meta['label_lst'] = label_lst                    
    # meta['accuracy'] = a
    # meta['AUC'] = b
    # meta['recall'] = c
    # meta['specificity'] = d
    # meta['sensitivity'] = e
    return(meta)
    
    with open(task_dir/'train.json','w') as file_obj:
        json.dump(meta,file_obj)


def get_index(input_list, index):
    return([input_list[i] for i in index])
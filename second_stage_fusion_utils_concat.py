# encoding:utf-8
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score, roc_curve,auc,average_precision_score,precision_recall_curve
import cv2
import pandas as pd
from utils import encode_test_label,Logger,encode_meta_choosed_label,encode_meta_label
import pandas as pd
from dependency import *
import torch


def create_cosine_learning_schedule(epochs, lr):
    cosine_learning_schedule = []

    for epoch in range(epochs):
        cos_inner = np.pi * (epoch % epochs)
        cos_inner /= epochs
        cos_out = np.cos(cos_inner) + 1
        final_lr = float(lr / 2 * cos_out)
        cosine_learning_schedule.append(final_lr)

    return cosine_learning_schedule


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


# multi-classification

from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.utils import to_categorical
test_index_df = pd.read_csv(test_index_path)
train_index_df = pd.read_csv(train_index_path)
val_index_df = pd.read_csv(val_index_path)

train_index_list = list(train_index_df['indexes'])
val_index_list = list(val_index_df['indexes'])
test_index_list = list(test_index_df['indexes'])

train_index_list_1 = train_index_list[0:206]
train_index_list_2 = train_index_list[206:]

df = pd.read_csv(img_info_path)

def get_label_list(image_index_list):
    diag_label_list = []
    pn_label_list = []
    str_label_list = []
    pig_label_list = []
    rs_label_list = []
    dag_label_list = []
    bwv_label_list = []
    vs_label_list = []
    meta_list = []

    img_feature = []
    img_hf_feature = []
    img_vf_feature = []
    img_vhf_feature = []

    from sklearn.decomposition import PCA

    from tqdm import tqdm_notebook, tqdm
    for index_num in tqdm(image_index_list):
        # index_num = test_index_list[100]
        img_info = df[index_num:index_num + 1]
        clinic_path = img_info['clinic']
        dermoscopy_path = img_info['derm']
        source_dir = '../release_v0/release_v0/images/'
        clinic_img = cv2.imread(source_dir + clinic_path[index_num])
        dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])
        meta_vector = encode_meta_choosed_label(img_info, index_num)

        [diagnosis_label, pigment_network_label, streaks_label, pigmentation_label, regression_structures_label,
         dots_and_globules_label, blue_whitish_veil_label, vascular_structures_label], [diagnosis_label_one_hot,
                                                                                        pigment_network_label_one_hot,
                                                                                        streaks_label_one_hot,
                                                                                        pigmentation_label_one_hot,
                                                                                        regression_structures_label_one_hot,
                                                                                        dots_and_globules_label_one_hot,
                                                                                        blue_whitish_veil_label_one_hot,
                                                                                        vascular_structures_label_one_hot] = encode_test_label(
            img_info, index_num)

        diag_label_list.append(diagnosis_label)
        pn_label_list.append(pigment_network_label)
        str_label_list.append(streaks_label)
        pig_label_list.append(pigmentation_label)
        rs_label_list.append(regression_structures_label)
        dag_label_list.append(dots_and_globules_label)
        bwv_label_list.append(blue_whitish_veil_label)
        vs_label_list.append(vascular_structures_label)
        meta_list.append(meta_vector)

    label_dict = {'diag': diag_label_list,
                  'pn': pn_label_list,
                  'str': str_label_list,
                  'pig': pig_label_list,
                  'rs': rs_label_list,
                  'dag': dag_label_list,
                  'bwv': bwv_label_list,
                  'vs': vs_label_list}

    return label_dict, meta_list

#################get labels#####################
print('fusing predictions from two-modality images and metadata......')
train_label_dict, train_meta_list = get_label_list(train_index_list)
val_label_dict, val_meta_list = get_label_list(val_index_list)
test_label_dict, test_meta_list = get_label_list(test_index_list)

############################这里的P2应该改成metadata直接通过一个multi-classification输出的P2#####################
#############################################然后后续是P2和P1的search##############################################

train_meta_array = np.array(train_meta_list)
print("train_meta_array.shape:", train_meta_array.shape)

meta_total_feature = train_meta_array
print(meta_total_feature.shape)

train_diag_label = np.array(train_label_dict['diag'])
print(train_diag_label.shape)

val_meta_array = np.array(val_meta_list)
val_meta_total_feature = val_meta_array
print(val_meta_total_feature.shape)

val_diag_label = np.array(val_label_dict['diag'])
print(val_diag_label.shape)

test_meta_array = np.array(test_meta_list)
test_meta_total_feature = test_meta_array
print(test_meta_total_feature.shape)

test_diag_label = np.array(test_label_dict['diag'])
print(test_diag_label.shape)
print('Done!')

img_pn_label = np.array(train_label_dict['pn'])
img_str_label = np.array(train_label_dict['str'])
img_pig_label = np.array(train_label_dict['pig'])
img_rs_label = np.array(train_label_dict['rs'])
img_dag_label = np.array(train_label_dict['dag'])
img_bwv_label = np.array(train_label_dict['bwv'])
img_vs_label = np.array(train_label_dict['vs'])
img_diag_label = np.array(train_label_dict['diag'])

test_img_pn_label = np.array(test_label_dict['pn'])
test_img_str_label = np.array(test_label_dict['str'])
test_img_pig_label = np.array(test_label_dict['pig'])
test_img_rs_label = np.array(test_label_dict['rs'])
test_img_dag_label = np.array(test_label_dict['dag'])
test_img_bwv_label = np.array(test_label_dict['bwv'])
test_img_vs_label = np.array(test_label_dict['vs'])
test_img_diag_label = np.array(test_label_dict['diag'])

val_img_pn_label = np.array(val_label_dict['pn'])
val_img_str_label = np.array(val_label_dict['str'])
val_img_pig_label = np.array(val_label_dict['pig'])
val_img_rs_label = np.array(val_label_dict['rs'])
val_img_dag_label = np.array(val_label_dict['dag'])
val_img_bwv_label = np.array(val_label_dict['bwv'])
val_img_vs_label = np.array(val_label_dict['vs'])
val_img_diag_label = np.array(val_label_dict['diag'])

img_pn_label_one_hot = to_categorical(np.array(train_label_dict['pn']))
img_str_label_one_hot = to_categorical(np.array(train_label_dict['str']))
img_pig_label_one_hot = to_categorical(np.array(train_label_dict['pig']))
img_rs_label_one_hot = to_categorical(np.array(train_label_dict['rs']))
img_dag_label_one_hot = to_categorical(np.array(train_label_dict['dag']))
img_bwv_label_one_hot = to_categorical(np.array(train_label_dict['bwv']))
img_vs_label_one_hot = to_categorical(np.array(train_label_dict['vs']))
img_diag_label_one_hot = to_categorical(np.array(train_label_dict['diag']))

test_img_pn_label_one_hot = to_categorical(np.array(test_label_dict['pn']))
test_img_str_label_one_hot = to_categorical(np.array(test_label_dict['str']))
test_img_pig_label_one_hot = to_categorical(np.array(test_label_dict['pig']))
test_img_rs_label_one_hot = to_categorical(np.array(test_label_dict['rs']))
test_img_dag_label_one_hot = to_categorical(np.array(test_label_dict['dag']))
test_img_bwv_label_one_hot = to_categorical(np.array(test_label_dict['bwv']))
test_img_vs_label_one_hot = to_categorical(np.array(test_label_dict['vs']))
test_img_diag_label_one_hot = to_categorical(np.array(test_label_dict['diag']))

forest = RandomForestClassifier(random_state=1)
clf = MultiOutputClassifier(forest, n_jobs=2)
meta_label = np.array(
        [img_diag_label, img_pn_label, img_str_label, img_pig_label, img_rs_label, img_dag_label, img_bwv_label,
         img_vs_label]).T
clf.fit(meta_total_feature, meta_label)


def multi_classifier_predict2(test_meta_feature):
    forest = RandomForestClassifier(random_state=1)
    clf = MultiOutputClassifier(forest, n_jobs=2)

    meta_label = np.array(
        [img_diag_label, img_pn_label, img_str_label, img_pig_label, img_rs_label, img_dag_label, img_bwv_label,
         img_vs_label]).T

    clf.fit(meta_total_feature, meta_label)
    test_preds_all = clf.predict(test_meta_feature)
    print("test_preds_all.shape:", test_preds_all.shape)

    test_preds_prob_all = clf.predict_proba(test_meta_feature)
    rows = test_meta_feature.shape[0]
    print("rows:", rows)

    test_preds = test_preds_all[:, 0]
    pn_test_preds = test_preds_all[:, 1]
    str_test_preds = test_preds_all[:, 2]
    pig_test_preds = test_preds_all[:, 3]
    rs_test_preds = test_preds_all[:, 4]
    dag_test_preds = test_preds_all[:, 5]
    bwv_test_preds = test_preds_all[:, 6]
    vs_test_preds = test_preds_all[:, 7]

    uncertainty1 = (test_preds == img_diag_label)
    uncertainty2 = (pn_test_preds == img_pn_label)
    uncertainty3 = (str_test_preds == img_str_label)
    uncertainty4 = (pig_test_preds == img_pig_label)
    uncertainty5 = (rs_test_preds == img_rs_label)
    uncertainty6 = (dag_test_preds == img_dag_label)
    uncertainty7 = (bwv_test_preds == img_bwv_label)
    uncertainty8 = (vs_test_preds == img_vs_label)

    # 再次训练
    meta_label = np.array(
        [img_diag_label, img_pn_label, img_str_label, img_pig_label, img_rs_label, img_dag_label, img_bwv_label,
         img_vs_label,
         uncertainty1, uncertainty2, uncertainty3, uncertainty4, uncertainty5, uncertainty6, uncertainty7,
         uncertainty8]).T
    clf.fit(meta_total_feature, meta_label)
    test_preds_all = clf.predict(test_meta_feature)
    print("test_preds_all.shape:", test_preds_all.shape)
    test_preds_prob_all = clf.predict_proba(test_meta_feature)
    rows = test_meta_feature.shape[0]
    print("rows:", rows)
    test_preds = test_preds_all[:, 0]
    pn_test_preds = test_preds_all[:, 1]
    str_test_preds = test_preds_all[:, 2]
    pig_test_preds = test_preds_all[:, 3]
    rs_test_preds = test_preds_all[:, 4]
    dag_test_preds = test_preds_all[:, 5]
    bwv_test_preds = test_preds_all[:, 6]
    vs_test_preds = test_preds_all[:, 7]
    test_preds_prob = np.array(test_preds_prob_all[0])
    pn_test_preds_prob = np.array(test_preds_prob_all[1])
    str_test_preds_prob = np.array(test_preds_prob_all[2])
    pig_test_preds_prob = np.array(test_preds_prob_all[3])
    rs_test_preds_prob = np.array(test_preds_prob_all[4])
    dag_test_preds_prob = np.array(test_preds_prob_all[5])
    bwv_test_preds_prob = np.array(test_preds_prob_all[6])
    vs_test_preds_prob = np.array(test_preds_prob_all[7])
    uncertainty1_prob = np.array(test_preds_prob_all[8])
    uncertainty2_prob = np.array(test_preds_prob_all[9])
    uncertainty3_prob = np.array(test_preds_prob_all[10])
    uncertainty4_prob = np.array(test_preds_prob_all[11])
    uncertainty5_prob = np.array(test_preds_prob_all[12])
    uncertainty6_prob = np.array(test_preds_prob_all[13])
    uncertainty7_prob = np.array(test_preds_prob_all[14])
    uncertainty8_prob = np.array(test_preds_prob_all[15])

    return [[test_preds, pn_test_preds, str_test_preds, pig_test_preds, rs_test_preds, dag_test_preds, bwv_test_preds,
             vs_test_preds],
            [test_preds_prob, pn_test_preds_prob, str_test_preds_prob, pig_test_preds_prob, rs_test_preds_prob,
             dag_test_preds_prob, bwv_test_preds_prob, vs_test_preds_prob,
             uncertainty1_prob, uncertainty2_prob, uncertainty3_prob, uncertainty4_prob, uncertainty5_prob,
             uncertainty6_prob, uncertainty7_prob, uncertainty8_prob]]

def multi_classifier_predict(test_meta_feature):
    test_preds_all = clf.predict(test_meta_feature)
    #print("test_preds_all.shape:", test_preds_all.shape)

    test_preds_prob_all = clf.predict_proba(test_meta_feature)
    rows = test_meta_feature.shape[0]
    #print("rows:", rows)

    test_preds = test_preds_all[:, 0]
    pn_test_preds = test_preds_all[:, 1]
    str_test_preds = test_preds_all[:, 2]
    pig_test_preds = test_preds_all[:, 3]
    rs_test_preds = test_preds_all[:, 4]
    dag_test_preds = test_preds_all[:, 5]
    bwv_test_preds = test_preds_all[:, 6]
    vs_test_preds = test_preds_all[:, 7]

    test_preds_prob = np.array(test_preds_prob_all[0])
    pn_test_preds_prob = np.array(test_preds_prob_all[1])
    str_test_preds_prob = np.array(test_preds_prob_all[2])
    pig_test_preds_prob = np.array(test_preds_prob_all[3])
    rs_test_preds_prob = np.array(test_preds_prob_all[4])
    dag_test_preds_prob = np.array(test_preds_prob_all[5])
    bwv_test_preds_prob = np.array(test_preds_prob_all[6])
    vs_test_preds_prob = np.array(test_preds_prob_all[7])

    return [[test_preds, pn_test_preds, str_test_preds, pig_test_preds, rs_test_preds, dag_test_preds, bwv_test_preds,
             vs_test_preds],
            [test_preds_prob, pn_test_preds_prob, str_test_preds_prob, pig_test_preds_prob, rs_test_preds_prob,
             dag_test_preds_prob, bwv_test_preds_prob, vs_test_preds_prob]]

# 若采用平均权重，则暂时不需要考虑搜索最佳权重

def predict(net, test_index_list, df, model_name, out_dir, mode, TTA=4, size=224,img_type="clinic",data_mode="img"):
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir + 'log.multi_modality_{}_{}_sinlesion.txt'.format(mode, model_name), mode='w')
    log.write('\n--- [START %s] %s\n\n' % ('IDENTIFIER', '-' * 64))
    net.set_mode('valid')


    # 7-point score
    # prob #pred

    # 1 pigment_network
    pn_prob_typ_list = [];
    pn_prob_asp_list = [];
    pn_prob_asb_list = [];
    pn_pred_list = []
    pn_prob_list = []
    # 2 streak
    str_prob_asb_list = [];
    str_prob_reg_list = [];
    str_prob_irg_list = [];
    str_pred_list = []
    str_prob_list = []
    # 3 pigmentation
    pig_prob_asb_list = [];
    pig_prob_reg_list = [];
    pig_prob_irg_list = [];
    pig_pred_list = []
    pig_prob_list = []
    # 4 regression structure
    rs_prob_asb_list = [];
    rs_prob_prs_list = [];
    rs_pred_list = []
    rs_prob_list = []
    # 5 dots and globules
    dag_prob_asb_list = [];
    dag_prob_reg_list = [];
    dag_prob_irg_list = [];
    dag_pred_list = []
    dag_prob_list = []
    # 6 blue whitish veil 1
    bwv_prob_asb_list = [];
    bwv_prob_prs_list = [];
    bwv_pred_list = []
    bwv_prob_list = []
    # 7 vascular strucuture
    vs_prob_asb_list = [];
    vs_prob_reg_list = [];
    vs_prob_irg_list = [];
    vs_pred_list = []
    vs_prob_list = []

    # label
    # 1 pigment_network
    pn_label_typ_list = [];
    pn_label_asp_list = [];
    pn_label_list = [];
    pn_label_asb_list = []
    # 2 streak
    str_label_asb_list = [];
    str_label_reg_list = [];
    str_label_irg_list = [];
    str_label_list = []
    # 3 pigmentation
    pig_label_asb_list = [];
    pig_label_reg_list = [];
    pig_label_irg_list = [];
    pig_label_list = []
    # 4 regression structure
    rs_label_asb_list = [];
    rs_label_prs_list = [];
    rs_label_list = []
    # 5 dots and globules
    dag_label_asb_list = [];
    dag_label_reg_list = [];
    dag_label_irg_list = [];
    dag_label_list = []
    # 6 blue whitish veil l
    bwv_label_asb_list = [];
    bwv_label_prs_list = [];
    bwv_label_list = []
    # 7vascular structure
    vs_label_asb_list = [];
    vs_label_reg_list = [];
    vs_label_irg_list = [];
    vs_label_list = []

    # total
    pred_list = [];
    prob_list = [];
    gt_list = []

    # diagnositic_prob and diagnositic_label
    nevu_prob_list = [];
    bcc_prob_list = [];
    mel_prob_list = [];
    misc_prob_list = [];
    sk_prob_list = []
    nevu_label_list = [];
    bcc_label_list = [];
    mel_label_list = [];
    misc_label_list = [];
    sk_label_list = []
    seven_point_feature_list = []

    for index_num in tqdm(test_index_list):
        img_info = df[index_num:index_num + 1]
        clinic_path = img_info['clinic']
        dermoscopy_path = img_info['derm']
        source_dir = './release_v0/release_v0/images/'
        clinic_img = cv2.imread(source_dir + clinic_path[index_num])
        dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])

        meta_data = encode_meta_choosed_label(img_info, index_num)

        if TTA == 0:
            meta_data = torch.from_numpy(np.array([meta_data]))
        elif TTA == 4:
            meta_data = torch.from_numpy(np.array([meta_data, meta_data, meta_data, meta_data]))
        elif TTA == 6:
            meta_data = torch.from_numpy(np.array([meta_data, meta_data, meta_data, meta_data, meta_data, meta_data]))

        clinic_img = cv2.resize(clinic_img, (size, size))
        clinic_img_hf = cv2.flip(clinic_img, 0)
        clinic_img_vf = cv2.flip(clinic_img, 1)
        clinic_img_vhf = cv2.flip(clinic_img, -1)
        clinic_img_90 = cv2.rotate(clinic_img, 0)
        clinic_img_270 = cv2.rotate(clinic_img, 2)

        dermoscopy_img = cv2.resize(dermoscopy_img, (size, size))
        dermoscopy_img_hf = cv2.flip(dermoscopy_img, 0)
        dermoscopy_img_vf = cv2.flip(dermoscopy_img, 1)
        dermoscopy_img_vhf = cv2.flip(dermoscopy_img, -1)
        dermoscopy_img_90 = cv2.rotate(dermoscopy_img, 0)
        dermoscopy_img_270 = cv2.rotate(dermoscopy_img, 2)

        clinic_img_total = np.array([clinic_img])
        dermoscopy_img_total = np.array([dermoscopy_img])
        if TTA == 4:
            dermoscopy_img_total = np.array([dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf])
            clinic_img_total = np.array([clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf])
        elif TTA == 6:
            dermoscopy_img_total = np.array(
                [dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf, dermoscopy_img_90,
                 dermoscopy_img_270])
            clinic_img_total = np.array(
                [clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf, clinic_img_90, clinic_img_270])

        dermoscopy_img_tensor = torch.from_numpy(
            np.transpose(dermoscopy_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255
        clinic_img_tensor = torch.from_numpy(np.transpose(clinic_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255

        if data_mode == "img":
            if img_type == "clic":
                [logit_diagnosis11, logit_pn11, logit_str11, logit_pig11, logit_rs11, logit_dag11, logit_bwv11,
                logit_vs11] = net((clinic_img_tensor).cuda())
            elif img_type == "derm":
                [logit_diagnosis22, logit_pn22, logit_str22, logit_pig22, logit_rs22, logit_dag22, logit_bwv22,
                logit_vs22] = net(((dermoscopy_img_tensor).cuda()))
            else:
                [(logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic,
                  logit_bwv_clic, logit_vs_clic, logit_uncertainty1_clic, logit_uncertainty2_clic,
                  logit_uncertainty3_clic, logit_uncertainty4_clic, logit_uncertainty5_clic, logit_uncertainty6_clic,
                  logit_uncertainty7_clic, logit_uncertainty8_clic),
                 (logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm,
                  logit_bwv_derm, logit_vs_derm, logit_uncertainty1_derm, logit_uncertainty2_derm,
                  logit_uncertainty3_derm, logit_uncertainty4_derm, logit_uncertainty5_derm, logit_uncertainty6_derm,
                  logit_uncertainty7_derm, logit_uncertainty8_derm),
                 (logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs,
                  logit_uncertainty1, logit_uncertainty2, logit_uncertainty3, logit_uncertainty4, logit_uncertainty5,
                  logit_uncertainty6, logit_uncertainty7, logit_uncertainty8),
                 (x_clic_cos, x_derm_cos)] = net((clinic_img_tensor.cuda(), dermoscopy_img_tensor.cuda()))

            if img_type == "clic":
                logit_diagnosis = logit_diagnosis11
                logit_pn = logit_pn11
                logit_str = logit_str11
                logit_pig = logit_pig11
                logit_rs = logit_rs11
                logit_dag = logit_dag11
                logit_bwv = logit_bwv11
                logit_vs = logit_vs11
            elif img_type == "derm":
                logit_diagnosis = logit_diagnosis22
                logit_pn = logit_pn22
                logit_str = logit_str22
                logit_pig = logit_pig22
                logit_rs = logit_rs22
                logit_dag = logit_dag22
                logit_bwv = logit_bwv22
                logit_vs = logit_vs22
            else:
                logit_diagnosis = logit_diagnosis
                logit_pn = logit_pn
                logit_str = logit_str
                logit_pig = logit_pig
                logit_rs = logit_rs
                logit_dag = logit_dag
                logit_bwv = logit_bwv
                logit_vs = logit_vs

        if data_mode == "metadata":
            _, [prob_2, pn_prob_2, str_prob_2, pig_prob_2, rs_prob_2, dag_prob_2, bwv_prob_2,
                vs_prob_2] = multi_classifier_predict(meta_data)
            logit_diagnosis = torch.tensor(prob_2)
            logit_pn = torch.tensor(pn_prob_2)
            logit_str = torch.tensor(str_prob_2)
            logit_pig = torch.tensor(pig_prob_2)
            logit_rs = torch.tensor(rs_prob_2)
            logit_dag = torch.tensor(dag_prob_2)
            logit_bwv = torch.tensor(bwv_prob_2)
            logit_vs = torch.tensor(vs_prob_2)

        # diagnostic_pred
        #print("pred1:", logit_diagnosis.shape)
        pred = softmax(logit_diagnosis.detach().cpu().numpy());
        #print("pred1_1:", pred.shape)
        pred = np.mean(pred, 0);
        #print("pred1_2:", pred.shape)
        pred_ = np.argmax(pred)
        nevu_prob = pred[0];
        bcc_prob = pred[1];
        mel_prob = pred[2];
        misc_prob = pred[3];
        sk_prob = pred[4];
        # pn_prob
        pn_pred = softmax(logit_pn.detach().cpu().numpy());
        pn_pred = np.mean(pn_pred, 0);
        pn_pred_ = np.argmax(pn_pred)
        pn_prob_asb = pn_pred[0];
        pn_prob_typ = pn_pred[1];
        pn_prob_asp = pn_pred[2];
        # str_prob
        str_pred = softmax(logit_str.detach().cpu().numpy())
        str_pred = np.mean(str_pred, 0);
        str_pred_ = np.argmax(str_pred)
        str_prob_asb = str_pred[0];
        str_prob_reg = str_pred[1];
        str_prob_irg = str_pred[2];
        # pig_prob
        pig_pred = softmax(logit_pig.detach().cpu().numpy())
        pig_pred = np.mean(pig_pred, 0);
        pig_pred_ = np.argmax(pig_pred)
        pig_prob_asb = pig_pred[0];
        pig_prob_reg = pig_pred[1];
        pig_prob_irg = pig_pred[2];
        # rs_prob
        rs_pred = softmax(logit_rs.detach().cpu().numpy())
        rs_pred = np.mean(rs_pred, 0);
        rs_pred_ = np.argmax(rs_pred)
        rs_prob_asb = rs_pred[0];
        rs_prob_prs = rs_pred[1];
        # dag_prob
        dag_pred = softmax(logit_dag.detach().cpu().numpy());
        dag_pred = np.mean(dag_pred, 0);
        dag_pred_ = np.argmax(dag_pred)
        dag_prob_asb = dag_pred[0];
        dag_prob_reg = dag_pred[1];
        dag_prob_irg = dag_pred[2]
        # bwv_prob
        bwv_pred = softmax(logit_bwv.detach().cpu().numpy());
        bwv_pred = np.mean(bwv_pred, 0);
        bwv_pred_ = np.argmax(bwv_pred)
        bwv_prob_asb = bwv_pred[0];
        bwv_prob_prs = bwv_pred[1]
        # vs_prob
        vs_pred = softmax(logit_vs.detach().cpu().numpy());
        vs_pred = np.mean(vs_pred, 0);
        vs_pred_ = np.argmax(vs_pred)
        vs_prob_asb = vs_pred[0];
        vs_prob_reg = vs_pred[1];
        vs_prob_irg = vs_pred[2]

        seven_point_feature_list.append(
            np.concatenate([pred, pn_pred, str_pred, pig_pred, rs_pred, dag_pred, bwv_pred, vs_pred], 0))
        # encode label
        [diagnosis_label, pigment_network_label, streaks_label, pigmentation_label, regression_structures_label,
         dots_and_globules_label, blue_whitish_veil_label, vascular_structures_label], [diagnosis_label_one_hot,
                                                                                        pigment_network_label_one_hot,
                                                                                        streaks_label_one_hot,
                                                                                        pigmentation_label_one_hot,
                                                                                        regression_structures_label_one_hot,
                                                                                        dots_and_globules_label_one_hot,
                                                                                        blue_whitish_veil_label_one_hot,
                                                                                        vascular_structures_label_one_hot] = encode_test_label(
            img_info, index_num)

        # diagnostic_label
        pred_list.append(pred_);
        prob_list.append(pred);
        gt_list.append(diagnosis_label);
        nevu_prob_list.append(nevu_prob);
        bcc_prob_list.append(bcc_prob);
        mel_prob_list.append(mel_prob);
        misc_prob_list.append(misc_prob);
        sk_prob_list.append(sk_prob);
        nevu_label_list.append(diagnosis_label_one_hot[0]);
        bcc_label_list.append(diagnosis_label_one_hot[1]);
        mel_label_list.append(diagnosis_label_one_hot[2]);
        misc_label_list.append(diagnosis_label_one_hot[3]);
        sk_label_list.append(diagnosis_label_one_hot[4]);

        # pn_label
        pn_pred_list.append(pn_pred_);
        pn_prob_list.append(pn_pred);
        pn_label_list.append(pigment_network_label);
        pn_prob_typ_list.append(pn_prob_typ);
        pn_prob_asp_list.append(pn_prob_asp);
        pn_prob_asb_list.append(pn_prob_asb);
        pn_label_asb_list.append(pigment_network_label_one_hot[0]);
        pn_label_typ_list.append(pigment_network_label_one_hot[1]);
        pn_label_asp_list.append(pigment_network_label_one_hot[2]);

        # str_label
        str_pred_list.append(str_pred_);
        str_prob_list.append(str_pred);

        str_label_list.append(streaks_label)
        str_prob_reg_list.append(str_prob_reg);
        str_prob_irg_list.append(str_prob_irg);
        str_prob_asb_list.append(str_prob_asb)
        str_label_asb_list.append(streaks_label_one_hot[0]);
        str_label_reg_list.append(streaks_label_one_hot[1]);
        str_label_irg_list.append(streaks_label_one_hot[2])

        # pig_label
        pig_pred_list.append(pig_pred_);
        pig_prob_list.append(pig_pred);

        pig_label_list.append(pigmentation_label)
        pig_prob_reg_list.append(pig_prob_reg);
        pig_prob_irg_list.append(pig_prob_irg);
        pig_prob_asb_list.append(pig_prob_asb)
        pig_label_asb_list.append(pigmentation_label_one_hot[0]);
        pig_label_reg_list.append(pigmentation_label_one_hot[1]);
        pig_label_irg_list.append(pigmentation_label_one_hot[2])

        # rs_label
        rs_pred_list.append(rs_pred_);
        rs_prob_list.append(rs_pred);

        rs_label_list.append(regression_structures_label)
        rs_prob_asb_list.append(rs_prob_asb);
        rs_prob_prs_list.append(rs_prob_prs)
        rs_label_asb_list.append(regression_structures_label_one_hot[0]);
        rs_label_prs_list.append(regression_structures_label_one_hot[1])

        # dag_label
        dag_pred_list.append(dag_pred_);
        dag_prob_list.append(dag_pred);

        dag_label_list.append(dots_and_globules_label)
        dag_prob_reg_list.append(dag_prob_reg);
        dag_prob_irg_list.append(dag_prob_irg);
        dag_prob_asb_list.append(dag_prob_asb)
        dag_label_asb_list.append(dots_and_globules_label_one_hot[0]);
        dag_label_reg_list.append(dots_and_globules_label_one_hot[1]);
        dag_label_irg_list.append(dots_and_globules_label_one_hot[2])

        # bwv_label
        bwv_pred_list.append(bwv_pred_);
        bwv_prob_list.append(bwv_pred);

        bwv_label_list.append(blue_whitish_veil_label)
        bwv_prob_asb_list.append(bwv_prob_asb);
        bwv_prob_prs_list.append((bwv_prob_prs))
        bwv_label_asb_list.append(blue_whitish_veil_label_one_hot[0]);
        bwv_label_prs_list.append(blue_whitish_veil_label_one_hot[1])

        # vs_label
        vs_pred_list.append(vs_pred_);
        vs_prob_list.append(vs_pred);

        vs_label_list.append(vascular_structures_label)
        vs_prob_reg_list.append(vs_prob_reg);
        vs_prob_irg_list.append(vs_prob_irg);
        vs_prob_asb_list.append(vs_prob_asb)
        vs_label_asb_list.append(vascular_structures_label_one_hot[0]);
        vs_label_reg_list.append(vascular_structures_label_one_hot[1]);
        vs_label_irg_list.append(vascular_structures_label_one_hot[2])

    pred = np.array(pred_list).squeeze();
    prob = np.array(prob_list).squeeze();

    gt = np.array(gt_list)
    nevu_prob = np.array(nevu_prob_list);
    bcc_prob = np.array(bcc_prob_list);
    mel_prob = np.array(mel_prob_list);
    misc_prob = np.array(misc_prob_list);
    sk_prob = np.array(sk_prob_list)
    nevu_label = np.array(nevu_label_list);
    bcc_label = np.array(bcc_label_list);
    mel_label = np.array(mel_label_list);
    misc_label = np.array(misc_label_list);
    sk_label = np.array(sk_label_list)

    pn_pred = np.array(pn_pred_list).squeeze();
    pn_prob = np.array(pn_prob_list).squeeze();

    pn_gt = np.array(pn_label_list)
    pn_prob_typ = np.array(pn_prob_typ_list);
    pn_prob_asp = np.array(pn_prob_asp_list);
    pn_prob_asb = np.array(pn_prob_asb_list)

    pn_label_typ = np.array(pn_label_typ_list);
    pn_label_asp = np.array(pn_label_asp_list);
    pn_label_asb = np.array(pn_label_asb_list)

    str_pred = np.array(str_pred_list).squeeze();
    str_prob = np.array(str_prob_list).squeeze();

    str_gt = np.array(str_label_list)
    str_prob_asb = np.array(str_prob_asb_list);
    str_prob_reg = np.array(str_prob_reg_list);
    str_prob_irg = np.array(str_prob_irg_list)
    str_label_asb = np.array(str_label_asb_list);
    str_label_reg = np.array(str_label_reg_list);
    str_label_irg = np.array(str_label_irg_list)

    pig_pred = np.array(pig_pred_list).squeeze();
    pig_prob = np.array(pig_prob_list).squeeze();

    pig_gt = np.array(pig_label_list)
    pig_prob_asb = np.array(pig_prob_asb_list);
    pig_prob_reg = np.array(pig_prob_reg_list);
    pig_prob_irg = np.array(pig_prob_irg_list)
    pig_label_asb = np.array(pig_label_asb_list);
    pig_label_reg = np.array(pig_label_reg_list);
    pig_label_irg = np.array(pig_label_irg_list)

    rs_pred = np.array(rs_pred_list).squeeze();
    rs_prob = np.array(rs_prob_list).squeeze();

    rs_gt = np.array(rs_label_list)
    rs_prob_asb = np.array(rs_prob_asb_list);
    rs_prob_prs = np.array(rs_prob_prs_list)
    rs_label_asb = np.array(rs_label_asb_list);
    rs_label_prs = np.array(rs_label_prs_list)

    dag_pred = np.array(dag_pred_list).squeeze();
    dag_prob = np.array(dag_prob_list).squeeze();

    dag_gt = np.array(dag_label_list)
    dag_prob_asb = np.array(dag_prob_asb_list);
    dag_prob_reg = np.array(dag_prob_reg_list);
    dag_prob_irg = np.array(dag_prob_irg_list)
    dag_label_asb = np.array(dag_label_asb_list);
    dag_label_reg = np.array(dag_label_reg_list);
    dag_label_irg = np.array(dag_label_irg_list)

    bwv_pred = np.array(bwv_pred_list).squeeze();
    bwv_prob = np.array(bwv_prob_list).squeeze();

    bwv_gt = np.array(bwv_label_list)
    bwv_prob_asb = np.array(bwv_prob_asb_list);
    bwv_prob_prs = np.array(bwv_prob_prs_list)
    bwv_label_asb = np.array(bwv_label_asb_list);
    bwv_label_prs = np.array(bwv_label_prs_list)

    vs_pred = np.array(vs_pred_list).squeeze();
    vs_prob = np.array(vs_prob_list).squeeze();

    vs_gt = np.array(vs_label_list)
    vs_prob_asb = np.array(vs_prob_asb_list);
    vs_prob_reg = np.array(vs_prob_reg_list);
    vs_prob_irg = np.array(vs_prob_irg_list)
    vs_label_asb = np.array(vs_label_asb_list);
    vs_label_reg = np.array(vs_label_reg_list);
    vs_label_irg = np.array(vs_label_irg_list)

    vs_acc = np.mean(vs_pred == vs_gt)
    bwv_acc = np.mean(bwv_pred == bwv_gt)
    dag_acc = np.mean(dag_pred == dag_gt)
    rs_acc = np.mean(rs_pred == rs_gt)
    pig_acc = np.mean(pig_pred == pig_gt)
    str_acc = np.mean(str_pred == str_gt)
    pn_acc = np.mean(pn_pred == pn_gt)
    diag_acc = np.mean(pred == gt)

    avg_acc = (vs_acc + bwv_acc + dag_acc + rs_acc + pig_acc + str_acc + pn_acc + diag_acc) / 8
    log.write('-' * 15 + '\n')
    log.write('avg_acc : {}\n'.format(avg_acc))
    log.write('vs_acc : {}\n'.format(np.mean(vs_pred == vs_gt)))
    log.write('bwv_acc : {}\n'.format(np.mean(bwv_pred == bwv_gt)))
    log.write('dag_acc : {}\n'.format(np.mean(dag_pred == dag_gt)))
    log.write('rs_acc : {}\n'.format(np.mean(rs_pred == rs_gt)))
    log.write('pig_acc : {}\n'.format(np.mean(pig_pred == pig_gt)))
    log.write('str_acc : {}\n'.format(np.mean(str_pred == str_gt)))
    log.write('pn_acc : {}\n'.format(np.mean(pn_pred == pn_gt)))
    log.write('diag_acc : {}\n'.format(np.mean(pred == gt)))

    nevu_auc = roc_auc_score((np.array(nevu_label) * 1).flatten(), nevu_prob.flatten())
    bcc_auc = roc_auc_score((np.array(bcc_label) * 1).flatten(), bcc_prob.flatten())
    mel_auc = roc_auc_score((np.array(mel_label) * 1).flatten(), mel_prob.flatten())
    misc_auc = roc_auc_score((np.array(misc_label) * 1).flatten(), misc_prob.flatten())
    sk_auc = roc_auc_score((np.array(sk_label) * 1).flatten(), sk_prob.flatten())
    log.write('-' * 15 + "\n")
    log.write('nevu_auc: {}\n'.format(nevu_auc))
    log.write('bcc_auc: {}\n'.format(bcc_auc))
    log.write('mel_auc: {}\n'.format(mel_auc))
    log.write('misc_auc: {}\n'.format(misc_auc))
    log.write('sk_auc: {}\n'.format(sk_auc))

    vs_asb_auc = roc_auc_score((np.array(vs_label_asb) * 1).flatten(), vs_prob_asb.flatten())
    vs_reg_auc = roc_auc_score((np.array(vs_label_reg) * 1).flatten(), vs_prob_reg.flatten())
    vs_irg_auc = roc_auc_score((np.array(vs_label_irg) * 1).flatten(), vs_prob_irg.flatten())
    log.write('-' * 15 + "\n")
    log.write('vs_asb_auc: {}\n'.format(vs_asb_auc))
    log.write('vs_reg_auc: {}\n'.format(vs_reg_auc))
    log.write('vs_irg_auc: {}\n'.format(vs_irg_auc))

    bwv_asb_auc = roc_auc_score((np.array(bwv_label_asb) * 1).flatten(), bwv_prob_asb.flatten())
    bwv_prs_auc = roc_auc_score((np.array(bwv_label_prs) * 1).flatten(), bwv_prob_prs.flatten())
    log.write('-' * 15 + '\n')
    log.write('bwv_asb_auc: {}\n'.format(bwv_asb_auc))
    log.write('bwv_prs_auc: {}\n'.format(bwv_prs_auc))

    dag_asb_auc = roc_auc_score((np.array(dag_label_asb) * 1).flatten(), dag_prob_asb.flatten())
    dag_reg_auc = roc_auc_score((np.array(dag_label_reg) * 1).flatten(), dag_prob_reg.flatten())
    dag_irg_auc = roc_auc_score((np.array(dag_label_irg) * 1).flatten(), dag_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('dag_asb_auc: {}\n'.format(dag_asb_auc))
    log.write('dag_reg_auc: {}\n'.format(dag_reg_auc))
    log.write('dag_irg_auc: {}\n'.format(dag_irg_auc))

    rs_asb_auc = roc_auc_score((np.array(rs_label_asb) * 1).flatten(), rs_prob_asb.flatten())
    rs_prs_auc = roc_auc_score((np.array(rs_label_prs) * 1).flatten(), rs_prob_prs.flatten())
    log.write('-' * 15 + '\n')
    log.write('rs_asb_auc: {}\n'.format(rs_asb_auc))
    log.write('rs_prs_auc: {}\n'.format(rs_prs_auc))

    pig_asb_auc = roc_auc_score((np.array(pig_label_asb) * 1).flatten(), pig_prob_asb.flatten())
    pig_reg_auc = roc_auc_score((np.array(pig_label_reg) * 1).flatten(), pig_prob_reg.flatten())
    pig_irg_auc = roc_auc_score((np.array(pig_label_irg) * 1).flatten(), pig_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('pig_asb_auc: {}\n'.format(pig_asb_auc))
    log.write('pig_reg_auc: {}\n'.format(pig_reg_auc))
    log.write('pig_irg_auc: {}\n'.format(pig_irg_auc))

    str_asb_auc = roc_auc_score((np.array(str_label_asb) * 1).flatten(), str_prob_asb.flatten())
    str_reg_auc = roc_auc_score((np.array(str_label_reg) * 1).flatten(), str_prob_reg.flatten())
    str_irg_auc = roc_auc_score((np.array(str_label_irg) * 1).flatten(), str_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('str_asb_auc: {}\n'.format(str_asb_auc))
    log.write('str_reg_auc: {}\n'.format(str_reg_auc))
    log.write('str_irg_auc: {}\n'.format(str_irg_auc))

    pn_typ_auc = roc_auc_score((np.array(pn_label_typ) * 1).flatten(), pn_prob_typ.flatten())
    pn_asp_auc = roc_auc_score((np.array(pn_label_asp) * 1).flatten(), pn_prob_asp.flatten())
    pn_asb_auc = roc_auc_score((np.array(pn_label_asb) * 1).flatten(), pn_prob_asb.flatten())
    log.write('-' * 15 + '\n')
    log.write('pn_typ_auc: {}\n'.format(pn_typ_auc))
    log.write('pn_asp_auc: {}\n'.format(pn_asp_auc))
    log.write('pn_asb_auc: {}\n'.format(pn_asb_auc))
    avg_auc = (vs_asb_auc + vs_reg_auc + vs_irg_auc
               + bwv_asb_auc + bwv_prs_auc
               + dag_asb_auc + dag_reg_auc + dag_irg_auc
               + rs_asb_auc + rs_prs_auc
               + pig_asb_auc + pig_reg_auc + pig_irg_auc
               + str_asb_auc + str_reg_auc + str_irg_auc
               + pn_typ_auc + pn_asp_auc + pn_asb_auc) / 19
    log.write('-' * 15 + '\n')
    log.write('avg_auc:{}\n'.format(avg_auc))
    log.close()

    return avg_acc, [prob, pn_prob, str_prob, pig_prob, rs_prob, dag_prob, bwv_prob, vs_prob], [
        np.array(nevu_label), np.array(bcc_label), np.array(mel_label), np.array(misc_label), np.array(sk_label)], [
               nevu_prob, bcc_prob, mel_prob, misc_prob, sk_prob], seven_point_feature_list, [gt, pn_gt,
                                                                                              str_gt, pig_gt,
                                                                                              rs_gt, dag_gt,
                                                                                              bwv_gt, vs_gt]


def predict3(net, test_index_list, df, model_name, out_dir, mode,weight, TTA=4, size=224, img_type="clinic"):
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir + 'log.multi_modality_{}_{}_sinlesion.txt'.format(mode, model_name), mode='w')
    log.write('\n--- [START %s] %s\n\n' % ('IDENTIFIER', '-' * 64))
    net.set_mode('valid')

    # 7-point score
    # prob #pred

    # 1 pigment_network
    pn_prob_typ_list = [];
    pn_prob_asp_list = [];
    pn_prob_asb_list = [];
    pn_pred_list = []
    pn_prob_list = []
    # 2 streak
    str_prob_asb_list = [];
    str_prob_reg_list = [];
    str_prob_irg_list = [];
    str_pred_list = []
    str_prob_list = []
    # 3 pigmentation
    pig_prob_asb_list = [];
    pig_prob_reg_list = [];
    pig_prob_irg_list = [];
    pig_pred_list = []
    pig_prob_list = []
    # 4 regression structure
    rs_prob_asb_list = [];
    rs_prob_prs_list = [];
    rs_pred_list = []
    rs_prob_list = []
    # 5 dots and globules
    dag_prob_asb_list = [];
    dag_prob_reg_list = [];
    dag_prob_irg_list = [];
    dag_pred_list = []
    dag_prob_list = []
    # 6 blue whitish veil 1
    bwv_prob_asb_list = [];
    bwv_prob_prs_list = [];
    bwv_pred_list = []
    bwv_prob_list = []
    # 7 vascular strucuture
    vs_prob_asb_list = [];
    vs_prob_reg_list = [];
    vs_prob_irg_list = [];
    vs_pred_list = []
    vs_prob_list = []

    # label
    # 1 pigment_network
    pn_label_typ_list = [];
    pn_label_asp_list = [];
    pn_label_list = [];
    pn_label_asb_list = []
    # 2 streak
    str_label_asb_list = [];
    str_label_reg_list = [];
    str_label_irg_list = [];
    str_label_list = []
    # 3 pigmentation
    pig_label_asb_list = [];
    pig_label_reg_list = [];
    pig_label_irg_list = [];
    pig_label_list = []
    # 4 regression structure
    rs_label_asb_list = [];
    rs_label_prs_list = [];
    rs_label_list = []
    # 5 dots and globules
    dag_label_asb_list = [];
    dag_label_reg_list = [];
    dag_label_irg_list = [];
    dag_label_list = []
    # 6 blue whitish veil l
    bwv_label_asb_list = [];
    bwv_label_prs_list = [];
    bwv_label_list = []
    # 7vascular structure
    vs_label_asb_list = [];
    vs_label_reg_list = [];
    vs_label_irg_list = [];
    vs_label_list = []

    # total
    pred_list = [];
    prob_list = [];
    gt_list = []

    # diagnositic_prob and diagnositic_label
    nevu_prob_list = [];
    bcc_prob_list = [];
    mel_prob_list = [];
    misc_prob_list = [];
    sk_prob_list = []
    nevu_label_list = [];
    bcc_label_list = [];
    mel_label_list = [];
    misc_label_list = [];
    sk_label_list = []
    seven_point_feature_list = []

    for index_num in tqdm(test_index_list):
        img_info = df[index_num:index_num + 1]
        clinic_path = img_info['clinic']
        dermoscopy_path = img_info['derm']
        source_dir = './release_v0/release_v0/images/'
        clinic_img = cv2.imread(source_dir + clinic_path[index_num])
        dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])

        meta_data = encode_meta_choosed_label(img_info, index_num)

        if TTA == 0:
            meta_data = torch.from_numpy(np.array([meta_data]))
        elif TTA == 4:
            meta_data = torch.from_numpy(np.array([meta_data, meta_data, meta_data, meta_data]))
        elif TTA == 6:
            meta_data = torch.from_numpy(np.array([meta_data, meta_data, meta_data, meta_data, meta_data, meta_data]))

        clinic_img = cv2.resize(clinic_img, (size, size))
        clinic_img_hf = cv2.flip(clinic_img, 0)
        clinic_img_vf = cv2.flip(clinic_img, 1)
        clinic_img_vhf = cv2.flip(clinic_img, -1)
        clinic_img_90 = cv2.rotate(clinic_img, 0)
        clinic_img_270 = cv2.rotate(clinic_img, 2)

        dermoscopy_img = cv2.resize(dermoscopy_img, (size, size))
        dermoscopy_img_hf = cv2.flip(dermoscopy_img, 0)
        dermoscopy_img_vf = cv2.flip(dermoscopy_img, 1)
        dermoscopy_img_vhf = cv2.flip(dermoscopy_img, -1)
        dermoscopy_img_90 = cv2.rotate(dermoscopy_img, 0)
        dermoscopy_img_270 = cv2.rotate(dermoscopy_img, 2)

        clinic_img_total = np.array([clinic_img])
        dermoscopy_img_total = np.array([dermoscopy_img])
        if TTA == 4:
            dermoscopy_img_total = np.array([dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf])
            clinic_img_total = np.array([clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf])
        elif TTA == 6:
            dermoscopy_img_total = np.array(
                [dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf, dermoscopy_img_90,
                 dermoscopy_img_270])
            clinic_img_total = np.array(
                [clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf, clinic_img_90, clinic_img_270])

        dermoscopy_img_tensor = torch.from_numpy(
            np.transpose(dermoscopy_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255
        clinic_img_tensor = torch.from_numpy(np.transpose(clinic_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255

        if img_type == "clic":
            [logit_diagnosis11, logit_pn11, logit_str11, logit_pig11, logit_rs11, logit_dag11, logit_bwv11,
             logit_vs11] = net((clinic_img_tensor).cuda())
        elif img_type == "derm":
            [logit_diagnosis22, logit_pn22, logit_str22, logit_pig22, logit_rs22, logit_dag22, logit_bwv22,
             logit_vs22] = net(((dermoscopy_img_tensor).cuda()))
        else:
            [(logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic,
              logit_bwv_clic, logit_vs_clic),
             (logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm,
              logit_bwv_derm, logit_vs_derm),
             (logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs),
             (x_clic_cos, x_derm_cos)] = net((clinic_img_tensor.cuda(), dermoscopy_img_tensor.cuda()))

        if img_type == "clic":
            logit_diagnosis3_1 = logit_diagnosis11
            logit_pn3_1 = logit_pn11
            logit_str3_1 = logit_str11
            logit_pig3_1 = logit_pig11
            logit_rs3_1 = logit_rs11
            logit_dag3_1 = logit_dag11
            logit_bwv3_1 = logit_bwv11
            logit_vs3_1 = logit_vs11
        elif img_type == "derm":
            logit_diagnosis3_1 = logit_diagnosis22
            logit_pn3_1 = logit_pn22
            logit_str3_1 = logit_str22
            logit_pig3_1 = logit_pig22
            logit_rs3_1 = logit_rs22
            logit_dag3_1 = logit_dag22
            logit_bwv3_1 = logit_bwv22
            logit_vs3_1 = logit_vs22
        else:
            logit_diagnosis3_1 = logit_diagnosis
            logit_pn3_1 = logit_pn
            logit_str3_1 = logit_str
            logit_pig3_1 = logit_pig
            logit_rs3_1 = logit_rs
            logit_dag3_1 = logit_dag
            logit_bwv3_1 = logit_bwv
            logit_vs3_1 = logit_vs

        _, [prob_2, pn_prob_2, str_prob_2, pig_prob_2, rs_prob_2, dag_prob_2, bwv_prob_2,
            vs_prob_2] = multi_classifier_predict(meta_data)
        logit_diagnosis3_2 = torch.tensor(prob_2)
        logit_pn3_2 = torch.tensor(pn_prob_2)
        logit_str3_2 = torch.tensor(str_prob_2)
        logit_pig3_2 = torch.tensor(pig_prob_2)
        logit_rs3_2 = torch.tensor(rs_prob_2)
        logit_dag3_2 = torch.tensor(dag_prob_2)
        logit_bwv3_2 = torch.tensor(bwv_prob_2)
        logit_vs3_2 = torch.tensor(vs_prob_2)

        logit_diagnosis3_1 = logit_diagnosis3_1.cpu()
        logit_pn3_1 = logit_pn3_1.cpu()
        logit_str3_1 = logit_str3_1.cpu()
        logit_pig3_1 = logit_pig3_1.cpu()
        logit_rs3_1 = logit_rs3_1.cpu()
        logit_dag3_1 = logit_dag3_1.cpu()
        logit_bwv3_1 = logit_bwv3_1.cpu()
        logit_vs3_1 = logit_vs3_1.cpu()

        logit_diagnosis3_2 = logit_diagnosis3_2.cpu()
        logit_pn3_2 = logit_pn3_2.cpu()
        logit_str3_2 = logit_str3_2.cpu()
        logit_pig3_2 = logit_pig3_2.cpu()
        logit_rs3_2 = logit_rs3_2.cpu()
        logit_dag3_2 = logit_dag3_2.cpu()
        logit_bwv3_2 = logit_bwv3_2.cpu()
        logit_vs3_2 = logit_vs3_2.cpu()
            # diagnostic_pred
        pred3_3 = weight*logit_diagnosis3_2 + (1 - weight) * logit_diagnosis3_1
        pred3_3 = softmax(pred3_3.detach().cpu().numpy());
        pred = np.mean(pred3_3, 0);
        pred_ = np.argmax(pred)
        nevu_prob = pred[0];
        bcc_prob = pred[1];
        mel_prob = pred[2];
        misc_prob = pred[3];
        sk_prob = pred[4];

        # pn_prob
        pn_pred3_3 = weight * logit_pn3_2 + (1 - weight) * logit_pn3_1
        pn_pred3_3 = softmax(pn_pred3_3.detach().cpu().numpy());
        pn_pred = np.mean(pn_pred3_3, 0);
        pn_pred_ = np.argmax(pn_pred)
        pn_prob_asb = pn_pred[0];
        pn_prob_typ = pn_pred[1];
        pn_prob_asp = pn_pred[2];

        # str_prob
        str_pred3_3 = weight * logit_str3_2 + (1-weight) * logit_str3_1
        str_pred3_3 = softmax(str_pred3_3.detach().cpu().numpy());
        str_pred = np.mean(str_pred3_3, 0);
        str_pred_ = np.argmax(str_pred)
        str_prob_asb = str_pred[0];
        str_prob_reg = str_pred[1];
        str_prob_irg = str_pred[2];

        # pig_prob
        pig_pred3_3 = weight * logit_pig3_2 + (1-weight) * logit_pig3_1
        pig_pred3_3 = softmax(pig_pred3_3.detach().cpu().numpy())
        pig_pred = np.mean(pig_pred3_3, 0);
        pig_pred_ = np.argmax(pig_pred)
        pig_prob_asb = pig_pred[0];
        pig_prob_reg = pig_pred[1];
        pig_prob_irg = pig_pred[2];

        # rs_prob
        rs_pred3_3 = weight * logit_rs3_2 + (1 - weight) * logit_rs3_1
        rs_pred3_3 = softmax(rs_pred3_3.detach().cpu().numpy())
        rs_pred = np.mean(rs_pred3_3, 0);
        rs_pred_ = np.argmax(rs_pred)
        rs_prob_asb = rs_pred[0];
        rs_prob_prs = rs_pred[1];

        # dag_prob
        dag_pred3_3 = weight * logit_dag3_2 + (1 - weight) * logit_dag3_1
        dag_pred3_3 = softmax(dag_pred3_3.detach().cpu().numpy())
        dag_pred = np.mean(dag_pred3_3, 0);
        dag_pred_ = np.argmax(dag_pred)
        dag_prob_asb = dag_pred[0];
        dag_prob_reg = dag_pred[1];
        dag_prob_irg = dag_pred[2]

        # bwv_prob
        bwv_pred3_3 = weight * logit_bwv3_2 + (1 - weight) * logit_bwv3_1
        bwv_pred3_3 = softmax(bwv_pred3_3.detach().cpu().numpy())
        bwv_pred = np.mean(bwv_pred3_3, 0);
        bwv_pred_ = np.argmax(bwv_pred)
        bwv_prob_asb = bwv_pred[0];
        bwv_prob_prs = bwv_pred[1]

        # vs_prob
        vs_pred3_3 = weight * logit_vs3_2 + (1 - weight) * logit_vs3_1
        vs_pred3_3 = softmax(vs_pred3_3.detach().cpu().numpy())
        vs_pred = np.mean(vs_pred3_3, 0);
        vs_pred_ = np.argmax(vs_pred)
        vs_prob_asb = vs_pred[0];
        vs_prob_reg = vs_pred[1];
        vs_prob_irg = vs_pred[2]

        # encode label
        [diagnosis_label, pigment_network_label, streaks_label, pigmentation_label, regression_structures_label,
         dots_and_globules_label, blue_whitish_veil_label, vascular_structures_label], [diagnosis_label_one_hot,
                                                                                        pigment_network_label_one_hot,
                                                                                        streaks_label_one_hot,
                                                                                        pigmentation_label_one_hot,
                                                                                        regression_structures_label_one_hot,
                                                                                        dots_and_globules_label_one_hot,
                                                                                        blue_whitish_veil_label_one_hot,
                                                                                        vascular_structures_label_one_hot] = encode_test_label(
            img_info, index_num)

        # diagnostic_label
        pred_list.append(pred_);
        prob_list.append(pred);
        gt_list.append(diagnosis_label);
        nevu_prob_list.append(nevu_prob);
        bcc_prob_list.append(bcc_prob);
        mel_prob_list.append(mel_prob);
        misc_prob_list.append(misc_prob);
        sk_prob_list.append(sk_prob);
        nevu_label_list.append(diagnosis_label_one_hot[0]);
        bcc_label_list.append(diagnosis_label_one_hot[1]);
        mel_label_list.append(diagnosis_label_one_hot[2]);
        misc_label_list.append(diagnosis_label_one_hot[3]);
        sk_label_list.append(diagnosis_label_one_hot[4]);

        # pn_label
        pn_pred_list.append(pn_pred_);
        pn_prob_list.append(pn_pred);
        pn_label_list.append(pigment_network_label);
        pn_prob_typ_list.append(pn_prob_typ);
        pn_prob_asp_list.append(pn_prob_asp);
        pn_prob_asb_list.append(pn_prob_asb);
        pn_label_asb_list.append(pigment_network_label_one_hot[0]);
        pn_label_typ_list.append(pigment_network_label_one_hot[1]);
        pn_label_asp_list.append(pigment_network_label_one_hot[2]);

        # str_label
        str_pred_list.append(str_pred_);
        str_prob_list.append(str_pred);

        str_label_list.append(streaks_label)
        str_prob_reg_list.append(str_prob_reg);
        str_prob_irg_list.append(str_prob_irg);
        str_prob_asb_list.append(str_prob_asb)
        str_label_asb_list.append(streaks_label_one_hot[0]);
        str_label_reg_list.append(streaks_label_one_hot[1]);
        str_label_irg_list.append(streaks_label_one_hot[2])

        # pig_label
        pig_pred_list.append(pig_pred_);
        pig_prob_list.append(pig_pred);

        pig_label_list.append(pigmentation_label)
        pig_prob_reg_list.append(pig_prob_reg);
        pig_prob_irg_list.append(pig_prob_irg);
        pig_prob_asb_list.append(pig_prob_asb)
        pig_label_asb_list.append(pigmentation_label_one_hot[0]);
        pig_label_reg_list.append(pigmentation_label_one_hot[1]);
        pig_label_irg_list.append(pigmentation_label_one_hot[2])

        # rs_label
        rs_pred_list.append(rs_pred_);
        rs_prob_list.append(rs_pred);

        rs_label_list.append(regression_structures_label)
        rs_prob_asb_list.append(rs_prob_asb);
        rs_prob_prs_list.append(rs_prob_prs)
        rs_label_asb_list.append(regression_structures_label_one_hot[0]);
        rs_label_prs_list.append(regression_structures_label_one_hot[1])

        # dag_label
        dag_pred_list.append(dag_pred_);
        dag_prob_list.append(dag_pred);

        dag_label_list.append(dots_and_globules_label)
        dag_prob_reg_list.append(dag_prob_reg);
        dag_prob_irg_list.append(dag_prob_irg);
        dag_prob_asb_list.append(dag_prob_asb)
        dag_label_asb_list.append(dots_and_globules_label_one_hot[0]);
        dag_label_reg_list.append(dots_and_globules_label_one_hot[1]);
        dag_label_irg_list.append(dots_and_globules_label_one_hot[2])

        # bwv_label
        bwv_pred_list.append(bwv_pred_);
        bwv_prob_list.append(bwv_pred);

        bwv_label_list.append(blue_whitish_veil_label)
        bwv_prob_asb_list.append(bwv_prob_asb);
        bwv_prob_prs_list.append((bwv_prob_prs))
        bwv_label_asb_list.append(blue_whitish_veil_label_one_hot[0]);
        bwv_label_prs_list.append(blue_whitish_veil_label_one_hot[1])

        # vs_label
        vs_pred_list.append(vs_pred_);
        vs_prob_list.append(vs_pred);

        vs_label_list.append(vascular_structures_label)
        vs_prob_reg_list.append(vs_prob_reg);
        vs_prob_irg_list.append(vs_prob_irg);
        vs_prob_asb_list.append(vs_prob_asb)
        vs_label_asb_list.append(vascular_structures_label_one_hot[0]);
        vs_label_reg_list.append(vascular_structures_label_one_hot[1]);
        vs_label_irg_list.append(vascular_structures_label_one_hot[2])

    pred = np.array(pred_list).squeeze();
    prob = np.array(prob_list).squeeze();

    gt = np.array(gt_list)
    nevu_prob = np.array(nevu_prob_list);
    bcc_prob = np.array(bcc_prob_list);
    mel_prob = np.array(mel_prob_list);
    misc_prob = np.array(misc_prob_list);
    sk_prob = np.array(sk_prob_list)
    nevu_label = np.array(nevu_label_list);
    bcc_label = np.array(bcc_label_list);
    mel_label = np.array(mel_label_list);
    misc_label = np.array(misc_label_list);
    sk_label = np.array(sk_label_list)

    pn_pred = np.array(pn_pred_list).squeeze();
    pn_prob = np.array(pn_prob_list).squeeze();

    pn_gt = np.array(pn_label_list)
    pn_prob_typ = np.array(pn_prob_typ_list);
    pn_prob_asp = np.array(pn_prob_asp_list);
    pn_prob_asb = np.array(pn_prob_asb_list)

    pn_label_typ = np.array(pn_label_typ_list);
    pn_label_asp = np.array(pn_label_asp_list);
    pn_label_asb = np.array(pn_label_asb_list)

    str_pred = np.array(str_pred_list).squeeze();
    str_prob = np.array(str_prob_list).squeeze();

    str_gt = np.array(str_label_list)
    str_prob_asb = np.array(str_prob_asb_list);
    str_prob_reg = np.array(str_prob_reg_list);
    str_prob_irg = np.array(str_prob_irg_list)
    str_label_asb = np.array(str_label_asb_list);
    str_label_reg = np.array(str_label_reg_list);
    str_label_irg = np.array(str_label_irg_list)

    pig_pred = np.array(pig_pred_list).squeeze();
    pig_prob = np.array(pig_prob_list).squeeze();

    pig_gt = np.array(pig_label_list)
    pig_prob_asb = np.array(pig_prob_asb_list);
    pig_prob_reg = np.array(pig_prob_reg_list);
    pig_prob_irg = np.array(pig_prob_irg_list)
    pig_label_asb = np.array(pig_label_asb_list);
    pig_label_reg = np.array(pig_label_reg_list);
    pig_label_irg = np.array(pig_label_irg_list)

    rs_pred = np.array(rs_pred_list).squeeze();
    rs_prob = np.array(rs_prob_list).squeeze();

    rs_gt = np.array(rs_label_list)
    rs_prob_asb = np.array(rs_prob_asb_list);
    rs_prob_prs = np.array(rs_prob_prs_list)
    rs_label_asb = np.array(rs_label_asb_list);
    rs_label_prs = np.array(rs_label_prs_list)

    dag_pred = np.array(dag_pred_list).squeeze();
    dag_prob = np.array(dag_prob_list).squeeze();

    dag_gt = np.array(dag_label_list)
    dag_prob_asb = np.array(dag_prob_asb_list);
    dag_prob_reg = np.array(dag_prob_reg_list);
    dag_prob_irg = np.array(dag_prob_irg_list)
    dag_label_asb = np.array(dag_label_asb_list);
    dag_label_reg = np.array(dag_label_reg_list);
    dag_label_irg = np.array(dag_label_irg_list)

    bwv_pred = np.array(bwv_pred_list).squeeze();
    bwv_prob = np.array(bwv_prob_list).squeeze();

    bwv_gt = np.array(bwv_label_list)
    bwv_prob_asb = np.array(bwv_prob_asb_list);
    bwv_prob_prs = np.array(bwv_prob_prs_list)
    bwv_label_asb = np.array(bwv_label_asb_list);
    bwv_label_prs = np.array(bwv_label_prs_list)

    vs_pred = np.array(vs_pred_list).squeeze();
    vs_prob = np.array(vs_prob_list).squeeze();

    vs_gt = np.array(vs_label_list)
    vs_prob_asb = np.array(vs_prob_asb_list);
    vs_prob_reg = np.array(vs_prob_reg_list);
    vs_prob_irg = np.array(vs_prob_irg_list)
    vs_label_asb = np.array(vs_label_asb_list);
    vs_label_reg = np.array(vs_label_reg_list);
    vs_label_irg = np.array(vs_label_irg_list)

    vs_acc = np.mean(vs_pred == vs_gt)
    bwv_acc = np.mean(bwv_pred == bwv_gt)
    dag_acc = np.mean(dag_pred == dag_gt)
    rs_acc = np.mean(rs_pred == rs_gt)
    pig_acc = np.mean(pig_pred == pig_gt)
    str_acc = np.mean(str_pred == str_gt)
    pn_acc = np.mean(pn_pred == pn_gt)
    diag_acc = np.mean(pred == gt)

    avg_acc = (vs_acc + bwv_acc + dag_acc + rs_acc + pig_acc + str_acc + pn_acc + diag_acc) / 8
    log.write('-' * 15 + '\n')
    log.write('avg_acc : {}\n'.format(avg_acc))
    log.write('vs_acc : {}\n'.format(np.mean(vs_pred == vs_gt)))
    log.write('bwv_acc : {}\n'.format(np.mean(bwv_pred == bwv_gt)))
    log.write('dag_acc : {}\n'.format(np.mean(dag_pred == dag_gt)))
    log.write('rs_acc : {}\n'.format(np.mean(rs_pred == rs_gt)))
    log.write('pig_acc : {}\n'.format(np.mean(pig_pred == pig_gt)))
    log.write('str_acc : {}\n'.format(np.mean(str_pred == str_gt)))
    log.write('pn_acc : {}\n'.format(np.mean(pn_pred == pn_gt)))
    log.write('diag_acc : {}\n'.format(np.mean(pred == gt)))

    nevu_auc = roc_auc_score((np.array(nevu_label) * 1).flatten(), nevu_prob.flatten())
    bcc_auc = roc_auc_score((np.array(bcc_label) * 1).flatten(), bcc_prob.flatten())
    mel_auc = roc_auc_score((np.array(mel_label) * 1).flatten(), mel_prob.flatten())
    misc_auc = roc_auc_score((np.array(misc_label) * 1).flatten(), misc_prob.flatten())
    sk_auc = roc_auc_score((np.array(sk_label) * 1).flatten(), sk_prob.flatten())
    log.write('-' * 15 + "\n")
    log.write('nevu_auc: {}\n'.format(nevu_auc))
    log.write('bcc_auc: {}\n'.format(bcc_auc))
    log.write('mel_auc: {}\n'.format(mel_auc))
    log.write('misc_auc: {}\n'.format(misc_auc))
    log.write('sk_auc: {}\n'.format(sk_auc))

    vs_asb_auc = roc_auc_score((np.array(vs_label_asb) * 1).flatten(), vs_prob_asb.flatten())
    vs_reg_auc = roc_auc_score((np.array(vs_label_reg) * 1).flatten(), vs_prob_reg.flatten())
    vs_irg_auc = roc_auc_score((np.array(vs_label_irg) * 1).flatten(), vs_prob_irg.flatten())
    log.write('-' * 15 + "\n")
    log.write('vs_asb_auc: {}\n'.format(vs_asb_auc))
    log.write('vs_reg_auc: {}\n'.format(vs_reg_auc))
    log.write('vs_irg_auc: {}\n'.format(vs_irg_auc))

    bwv_asb_auc = roc_auc_score((np.array(bwv_label_asb) * 1).flatten(), bwv_prob_asb.flatten())
    bwv_prs_auc = roc_auc_score((np.array(bwv_label_prs) * 1).flatten(), bwv_prob_prs.flatten())
    log.write('-' * 15 + '\n')
    log.write('bwv_asb_auc: {}\n'.format(bwv_asb_auc))
    log.write('bwv_prs_auc: {}\n'.format(bwv_prs_auc))

    dag_asb_auc = roc_auc_score((np.array(dag_label_asb) * 1).flatten(), dag_prob_asb.flatten())
    dag_reg_auc = roc_auc_score((np.array(dag_label_reg) * 1).flatten(), dag_prob_reg.flatten())
    dag_irg_auc = roc_auc_score((np.array(dag_label_irg) * 1).flatten(), dag_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('dag_asb_auc: {}\n'.format(dag_asb_auc))
    log.write('dag_reg_auc: {}\n'.format(dag_reg_auc))
    log.write('dag_irg_auc: {}\n'.format(dag_irg_auc))

    rs_asb_auc = roc_auc_score((np.array(rs_label_asb) * 1).flatten(), rs_prob_asb.flatten())
    rs_prs_auc = roc_auc_score((np.array(rs_label_prs) * 1).flatten(), rs_prob_prs.flatten())
    log.write('-' * 15 + '\n')
    log.write('rs_asb_auc: {}\n'.format(rs_asb_auc))
    log.write('rs_prs_auc: {}\n'.format(rs_prs_auc))

    pig_asb_auc = roc_auc_score((np.array(pig_label_asb) * 1).flatten(), pig_prob_asb.flatten())
    pig_reg_auc = roc_auc_score((np.array(pig_label_reg) * 1).flatten(), pig_prob_reg.flatten())
    pig_irg_auc = roc_auc_score((np.array(pig_label_irg) * 1).flatten(), pig_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('pig_asb_auc: {}\n'.format(pig_asb_auc))
    log.write('pig_reg_auc: {}\n'.format(pig_reg_auc))
    log.write('pig_irg_auc: {}\n'.format(pig_irg_auc))

    str_asb_auc = roc_auc_score((np.array(str_label_asb) * 1).flatten(), str_prob_asb.flatten())
    str_reg_auc = roc_auc_score((np.array(str_label_reg) * 1).flatten(), str_prob_reg.flatten())
    str_irg_auc = roc_auc_score((np.array(str_label_irg) * 1).flatten(), str_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('str_asb_auc: {}\n'.format(str_asb_auc))
    log.write('str_reg_auc: {}\n'.format(str_reg_auc))
    log.write('str_irg_auc: {}\n'.format(str_irg_auc))

    pn_typ_auc = roc_auc_score((np.array(pn_label_typ) * 1).flatten(), pn_prob_typ.flatten())
    pn_asp_auc = roc_auc_score((np.array(pn_label_asp) * 1).flatten(), pn_prob_asp.flatten())
    pn_asb_auc = roc_auc_score((np.array(pn_label_asb) * 1).flatten(), pn_prob_asb.flatten())
    log.write('-' * 15 + '\n')
    log.write('pn_typ_auc: {}\n'.format(pn_typ_auc))
    log.write('pn_asp_auc: {}\n'.format(pn_asp_auc))
    log.write('pn_asb_auc: {}\n'.format(pn_asb_auc))
    avg_auc = (vs_asb_auc + vs_reg_auc + vs_irg_auc
               + bwv_asb_auc + bwv_prs_auc
               + dag_asb_auc + dag_reg_auc + dag_irg_auc
               + rs_asb_auc + rs_prs_auc
               + pig_asb_auc + pig_reg_auc + pig_irg_auc
               + str_asb_auc + str_reg_auc + str_irg_auc
               + pn_typ_auc + pn_asp_auc + pn_asb_auc) / 19
    log.write('-' * 15 + '\n')
    log.write('avg_auc:{}\n'.format(avg_auc))
    log.close()

    return avg_acc, [prob, pn_prob, str_prob, pig_prob, rs_prob, dag_prob, bwv_prob, vs_prob], [
        np.array(nevu_label), np.array(bcc_label), np.array(mel_label), np.array(misc_label), np.array(sk_label)], [
               nevu_prob, bcc_prob, mel_prob, misc_prob, sk_prob], seven_point_feature_list, [gt, pn_gt,
                                                                                              str_gt, pig_gt,
                                                                                              rs_gt, dag_gt,
                                                                                              bwv_gt, vs_gt]
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
import torch.nn as nn
import sklearn
from model import Resnet_with_uncertainty


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
    meta_list_one_hot = []

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
        '''这里对于metadata需要变更'''
        meta_vector_one_hot, meta_vector = encode_meta_choosed_label(img_info, index_num)

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
        meta_list_one_hot.append(meta_vector_one_hot)

    label_dict = {'diag': diag_label_list,
                  'pn': pn_label_list,
                  'str': str_label_list,
                  'pig': pig_label_list,
                  'rs': rs_label_list,
                  'dag': dag_label_list,
                  'bwv': bwv_label_list,
                  'vs': vs_label_list}

    return label_dict, meta_list, meta_list_one_hot


# multi-classification
from tensorflow.keras.utils import to_categorical
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

test_index_df = pd.read_csv(test_index_path)
train_index_df = pd.read_csv(train_index_path)
val_index_df = pd.read_csv(val_index_path)

train_index_list = list(train_index_df['indexes'])
val_index_list = list(val_index_df['indexes'])
test_index_list = list(test_index_df['indexes'])

train_index_list_1 = train_index_list[0:206]
train_index_list_2 = train_index_list[206:]
df = pd.read_csv(img_info_path)

train_label_dict,train_meta_list,train_meta_list_one_hot = get_label_list(train_index_list)
val_label_dict,val_meta_list,val_meta_list_one_hot = get_label_list(val_index_list)
test_label_dict,test_meta_list,test_meta_list_one_hot = get_label_list(test_index_list)

train_meta_array = np.array(train_meta_list)
val_meta_array = np.array(val_meta_list)
test_meta_array = np.array(test_meta_list)
train_meta_array_one_hot = np.array(train_meta_list_one_hot)
val_meta_array_one_hot = np.array(val_meta_list_one_hot)
test_meta_array_one_hot = np.array(test_meta_list_one_hot)

meta_total_feature_one_hot = train_meta_array_one_hot
val_meta_total_feature_one_hot = val_meta_array_one_hot
test_meta_total_feature_one_hot = test_meta_array_one_hot
meta_total_feature = train_meta_array
val_meta_total_feature = val_meta_array
test_meta_total_feature = test_meta_array

train_diag_label = np.array(train_label_dict['diag'])
val_diag_label = np.array(val_label_dict['diag'])
test_diag_label = np.array(test_label_dict['diag'])

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

val_img_pn_label= np.array(val_label_dict['pn'])
val_img_str_label= np.array(val_label_dict['str'])
val_img_pig_label= np.array(val_label_dict['pig'])
val_img_rs_label= np.array(val_label_dict['rs'])
val_img_dag_label= np.array(val_label_dict['dag'])
val_img_bwv_label= np.array(val_label_dict['bwv'])
val_img_vs_label= np.array(val_label_dict['vs'])
val_img_diag_label= np.array(val_label_dict['diag'])

img_pn_label_one_hot = to_categorical(np.array(train_label_dict['pn']))
img_str_label_one_hot= to_categorical(np.array(train_label_dict['str']))
img_pig_label_one_hot= to_categorical(np.array(train_label_dict['pig']))
img_rs_label_one_hot= to_categorical(np.array(train_label_dict['rs']))
img_dag_label_one_hot= to_categorical(np.array(train_label_dict['dag']))
img_bwv_label_one_hot= to_categorical(np.array(train_label_dict['bwv']))
img_vs_label_one_hot= to_categorical(np.array(train_label_dict['vs']))
img_diag_label_one_hot = to_categorical(np.array(train_label_dict['diag']))

val_img_pn_label_one_hot = to_categorical(np.array(val_label_dict['pn']))
val_img_str_label_one_hot= to_categorical(np.array(val_label_dict['str']))
val_img_pig_label_one_hot= to_categorical(np.array(val_label_dict['pig']))
val_img_rs_label_one_hot= to_categorical(np.array(val_label_dict['rs']))
val_img_dag_label_one_hot= to_categorical(np.array(val_label_dict['dag']))
val_img_bwv_label_one_hot= to_categorical(np.array(val_label_dict['bwv']))
val_img_vs_label_one_hot= to_categorical(np.array(val_label_dict['vs']))
val_img_diag_label_one_hot = to_categorical(np.array(val_label_dict['diag']))

test_img_pn_label_one_hot = to_categorical(np.array(test_label_dict['pn']))
test_img_str_label_one_hot= to_categorical(np.array(test_label_dict['str']))
test_img_pig_label_one_hot= to_categorical(np.array(test_label_dict['pig']))
test_img_rs_label_one_hot= to_categorical(np.array(test_label_dict['rs']))
test_img_dag_label_one_hot= to_categorical(np.array(test_label_dict['dag']))
test_img_bwv_label_one_hot= to_categorical(np.array(test_label_dict['bwv']))
test_img_vs_label_one_hot= to_categorical(np.array(test_label_dict['vs']))
test_img_diag_label_one_hot = to_categorical(np.array(test_label_dict['diag']))


def multi_classifier_train():
    forest = RandomForestClassifier(random_state=1)
    clf = MultiOutputClassifier(forest, n_jobs=2)
    meta_label = np.array(
        [img_diag_label, img_pn_label, img_str_label, img_pig_label, img_rs_label, img_dag_label, img_bwv_label,
         img_vs_label]).T
    clf.fit(meta_total_feature_one_hot, meta_label)
    train_preds_all = clf.predict(meta_total_feature_one_hot)
    train_preds_prob_all = clf.predict_proba(meta_total_feature_one_hot)

    # preds labels
    diag_preds_train = train_preds_all[:, 0]
    pn_preds_train = train_preds_all[:, 1]
    str_preds_train = train_preds_all[:, 2]
    pig_preds_train = train_preds_all[:, 3]
    rs_preds_train = train_preds_all[:, 4]
    dag_preds_train = train_preds_all[:, 5]
    bwv_preds_train = train_preds_all[:, 6]
    vs_preds_train = train_preds_all[:, 7]
    diag_preds_train = torch.tensor(diag_preds_train)
    pn_preds_train = torch.tensor(pn_preds_train)
    str_preds_train = torch.tensor(str_preds_train)
    pig_preds_train = torch.tensor(pig_preds_train)
    rs_preds_train = torch.tensor(rs_preds_train)
    dag_preds_train = torch.tensor(dag_preds_train)
    bwv_preds_train = torch.tensor(bwv_preds_train)
    vs_preds_train = torch.tensor(vs_preds_train)

    diag_preds_train = diag_preds_train.unsqueeze(0)
    pn_preds_train = pn_preds_train.unsqueeze(0)
    str_preds_train = str_preds_train.unsqueeze(0)
    pig_preds_train = pig_preds_train.unsqueeze(0)
    rs_preds_train = rs_preds_train.unsqueeze(0)
    dag_preds_train = dag_preds_train.unsqueeze(0)
    bwv_preds_train = bwv_preds_train.unsqueeze(0)
    vs_preds_train = vs_preds_train.unsqueeze(0)

    # true labels
    img_diag_label_t = torch.tensor(img_diag_label).unsqueeze(0)
    img_pn_label_t = torch.tensor(img_pn_label).unsqueeze(0)
    img_str_label_t = torch.tensor(img_str_label).unsqueeze(0)
    img_pig_label_t = torch.tensor(img_pig_label).unsqueeze(0)
    img_rs_label_t = torch.tensor(img_rs_label).unsqueeze(0)
    img_dag_label_t = torch.tensor(img_dag_label).unsqueeze(0)
    img_bwv_label_t = torch.tensor(img_bwv_label).unsqueeze(0)
    img_vs_label_t = torch.tensor(img_vs_label).unsqueeze(0)

    img_diag_label_t = np.array(img_diag_label_t.cpu(), dtype="int")
    img_pn_label_t = np.array(img_pn_label_t.cpu(), dtype="int")
    img_str_label_t = np.array(img_str_label_t.cpu(), dtype="int")
    img_pig_label_t = np.array(img_pig_label_t.cpu(), dtype="int")
    img_rs_label_t = np.array(img_rs_label_t.cpu(), dtype="int")
    img_dag_label_t = np.array(img_dag_label_t.cpu(), dtype="int")
    img_bwv_label_t = np.array(img_bwv_label_t.cpu(), dtype="int")
    img_vs_label_t = np.array(img_vs_label_t.cpu(), dtype="int")

    diag_preds_train = np.array(diag_preds_train.cpu(), dtype="int")
    pn_preds_train = np.array(pn_preds_train.cpu(), dtype="int")
    str_preds_train = np.array(str_preds_train.cpu(), dtype="int")
    pig_preds_train = np.array(pig_preds_train.cpu(), dtype="int")
    rs_preds_train = np.array(rs_preds_train.cpu(), dtype="int")
    dag_preds_train = np.array(dag_preds_train.cpu(), dtype="int")
    bwv_preds_train = np.array(bwv_preds_train.cpu(), dtype="int")
    vs_preds_train = np.array(vs_preds_train.cpu(), dtype="int")

    t = np.hstack((img_diag_label_t.T, img_pn_label_t.T))
    t = np.hstack((t, img_str_label_t.T))
    t = np.hstack((t, img_pig_label_t.T))
    t = np.hstack((t, img_rs_label_t.T))
    t = np.hstack((t, img_dag_label_t.T))
    t = np.hstack((t, img_bwv_label_t.T))
    t = np.hstack((t, img_vs_label_t.T))
    t = torch.tensor(t)
    print("t.size:", t.size())
    p = np.hstack((diag_preds_train.T, pn_preds_train.T))
    p = np.hstack((p, str_preds_train.T))
    p = np.hstack((p, pig_preds_train.T))
    p = np.hstack((p, rs_preds_train.T))
    p = np.hstack((p, dag_preds_train.T))
    p = np.hstack((p, bwv_preds_train.T))
    p = np.hstack((p, vs_preds_train.T))
    p = torch.tensor(p)
    print("p.size:", p.size())
    certainty_b = (t == p).long()
    certainty_ones = torch.tensor(np.ones(certainty_b.size()))
    uncertainty_label_meta = []
    for i in range(certainty_ones.size(0)):
        mean_squared_error_tmp = mean_squared_error(certainty_ones[i, :], certainty_b[i, :])
        if mean_squared_error_tmp >= 0.6:
            mean_squared_error_tmp = 1
        else:
            mean_squared_error_tmp = 0
        uncertainty_label_meta.append([mean_squared_error_tmp])
    uncertainty_label_meta = torch.tensor(uncertainty_label_meta)
    uncertainty_label_meta = uncertainty_label_meta.float().cuda()
    uncertainty_label_meta = torch.squeeze(uncertainty_label_meta)
    print("uncertainty_label_meta.size:", uncertainty_label_meta.size())

    # Regression
    # train the regression model
    meta_total_feature_2 = np.array(torch.cat((torch.tensor(meta_total_feature), p), 1))
    n_folds = 6
    model_br = BayesianRidge()
    model_lr = LinearRegression()
    model_etc = ElasticNet()
    model_svr = SVR()
    model_gbr = GradientBoostingRegressor()
    model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']
    model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr]
    cos_score_list = []
    pred_y_list = []
    for model in model_dic:
        scores = cross_val_score(model, meta_total_feature_2, uncertainty_label_meta.cpu(), cv=n_folds)
        cos_score_list.append(scores)
        pred_y_list.append(model.fit(meta_total_feature_2, uncertainty_label_meta.cpu()).predict(meta_total_feature_2))
    n_samples, n_features = meta_total_feature_2.shape
    print("n_samples:", n_samples)
    print("n_features:", n_features)
    model_metrics_names = [explained_variance_score, mean_squared_error, r2_score]
    model_metrics_list = []
    for i in range(len(model_names)):
        tmp_list = []
        for m in model_metrics_names:
            tmp_score = m(uncertainty_label_meta.cpu(), pred_y_list[i])
            tmp_list.append(tmp_score)
        model_metrics_list.append(tmp_list)
    df1 = pd.DataFrame(cos_score_list, index=model_names)
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mse', 'r2'])
    print(70 * '-')
    print('cross validation result:')
    print(df1)
    print(70 * '-')
    print('regression metrics:')
    print(df2)
    print(70 * '-')
    #
    plt.figure()
    plt.plot(np.arange(meta_total_feature_2.shape[0]), uncertainty_label_meta.cpu(), color='k', label='true y')
    color_list = ['r', 'b', 'g', 'y', 'c']
    linestyle_list = ['-', '.', 'o', 'v', '*']
    for i, pre_y in enumerate(pred_y_list):
        plt.plot(np.arange(meta_total_feature_2.shape[0]), pred_y_list[i], color_list[i], label=model_names[i])
    plt.title('regression result comparison')
    plt.legend(loc='upper right')
    plt.ylabel('real and predicted value')
    plt.show()

    return clf,model_gbr

def multi_classifier_predict(clf,model_gbr,test_meta_total_feature,test_meta_total_feature_one_hot):
    test_preds_all = clf.predict(test_meta_total_feature_one_hot)
    test_probs_all = clf.predict_proba(test_meta_total_feature_one_hot)
    diag_preds_test = test_preds_all[:, 0]
    pn_preds_test = test_preds_all[:, 1]
    str_preds_test = test_preds_all[:, 2]
    pig_preds_test = test_preds_all[:, 3]
    rs_preds_test = test_preds_all[:, 4]
    dag_preds_test = test_preds_all[:, 5]
    bwv_preds_test = test_preds_all[:, 6]
    vs_preds_test = test_preds_all[:, 7]
    diag_probs_test = np.array(test_probs_all[0])
    pn_probs_test = np.array(test_probs_all[1])
    str_probs_test = np.array(test_probs_all[2])
    pig_probs_test = np.array(test_probs_all[3])
    rs_probs_test = np.array(test_probs_all[4])
    dag_probs_test = np.array(test_probs_all[5])
    bwv_probs_test = np.array(test_probs_all[6])
    vs_probs_test = np.array(test_probs_all[7])

    diag_preds_test = torch.tensor(diag_preds_test)
    pn_preds_test = torch.tensor(pn_preds_test)
    str_preds_test = torch.tensor(str_preds_test)
    pig_preds_test = torch.tensor(pig_preds_test)
    rs_preds_test = torch.tensor(rs_preds_test)
    dag_preds_test = torch.tensor(dag_preds_test)
    bwv_preds_test = torch.tensor(bwv_preds_test)
    vs_preds_test = torch.tensor(vs_preds_test)

    diag_preds_test = diag_preds_test.unsqueeze(0)
    pn_preds_test = pn_preds_test.unsqueeze(0)
    str_preds_test = str_preds_test.unsqueeze(0)
    pig_preds_test = pig_preds_test.unsqueeze(0)
    rs_preds_test = rs_preds_test.unsqueeze(0)
    dag_preds_test = dag_preds_test.unsqueeze(0)
    bwv_preds_test = bwv_preds_test.unsqueeze(0)
    vs_preds_test = vs_preds_test.unsqueeze(0)

    diag_preds_test = np.array(diag_preds_test.cpu(), dtype="int")
    pn_preds_test = np.array(pn_preds_test.cpu(), dtype="int")
    str_preds_test = np.array(str_preds_test.cpu(), dtype="int")
    pig_preds_test = np.array(pig_preds_test.cpu(), dtype="int")
    rs_preds_test = np.array(rs_preds_test.cpu(), dtype="int")
    dag_preds_test = np.array(dag_preds_test.cpu(), dtype="int")
    bwv_preds_test = np.array(bwv_preds_test.cpu(), dtype="int")
    vs_preds_test = np.array(vs_preds_test.cpu(), dtype="int")

    p = np.hstack((diag_preds_test.T, pn_preds_test.T))
    p = np.hstack((p, str_preds_test.T))
    p = np.hstack((p, pig_preds_test.T))
    p = np.hstack((p, rs_preds_test.T))
    p = np.hstack((p, dag_preds_test.T))
    p = np.hstack((p, bwv_preds_test.T))
    p = np.hstack((p, vs_preds_test.T))
    p = torch.tensor(p)
    #print("p.size:", p.size())
    test_meta_total_feature_2 = np.array(torch.cat((torch.tensor(test_meta_total_feature), p), 1))
    uncertainty = model_gbr.predict(test_meta_total_feature_2)  # bigger uncertainty, less accurate the prediction is.

    return [diag_preds_test, pn_preds_test, str_preds_test, pig_preds_test, rs_preds_test, dag_preds_test, bwv_preds_test,
         vs_preds_test],[diag_probs_test, pn_probs_test, str_probs_test, pig_probs_test, rs_probs_test, dag_probs_test, bwv_probs_test,
         vs_probs_test], uncertainty

def predict(net2,net, test_index_list, df, model_name, out_dir, mode, TTA=4, size=229,img_type="clinic",data_mode="img"):
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir + 'log.multi_modality_{}_{}_sinlesion.txt'.format(mode, model_name), mode='w')
    log.write('\n--- [START %s] %s\n\n' % ('IDENTIFIER', '-' * 64))
    net.set_mode('valid')
    net2.set_mode('valid')

    # 7-point score
    # prob #pred

    # 1 pigment_network
    pn_prob_typ_list = [];
    pn_prob_asp_list = [];
    pn_prob_asb_list = [];
    pn_pred_typ_list = [];
    pn_pred_asp_list = [];
    pn_pred_asb_list = [];
    pn_pred_list = []
    pn_prob_list = []
    # 2 streak
    str_prob_asb_list = [];
    str_prob_reg_list = [];
    str_prob_irg_list = [];
    str_pred_asb_list = [];
    str_pred_reg_list = [];
    str_pred_irg_list = [];
    str_pred_list = []
    str_prob_list = []
    # 3 pigmentation
    pig_prob_asb_list = [];
    pig_prob_reg_list = [];
    pig_prob_irg_list = [];
    pig_pred_asb_list = [];
    pig_pred_reg_list = [];
    pig_pred_irg_list = [];
    pig_pred_list = []
    pig_prob_list = []
    # 4 regression structure
    rs_prob_asb_list = [];
    rs_prob_prs_list = [];
    rs_pred_asb_list = [];
    rs_pred_prs_list = [];
    rs_pred_list = []
    rs_prob_list = []
    # 5 dots and globules
    dag_prob_asb_list = [];
    dag_prob_reg_list = [];
    dag_prob_irg_list = [];
    dag_pred_asb_list = [];
    dag_pred_reg_list = [];
    dag_pred_irg_list = [];
    dag_pred_list = []
    dag_prob_list = []
    # 6 blue whitish veil 1
    bwv_prob_asb_list = [];
    bwv_prob_prs_list = [];
    bwv_pred_asb_list = [];
    bwv_pred_prs_list = [];
    bwv_pred_list = []
    bwv_prob_list = []
    # 7 vascular strucuture
    vs_prob_asb_list = [];
    vs_prob_reg_list = [];
    vs_prob_irg_list = [];
    vs_pred_asb_list = [];
    vs_pred_reg_list = [];
    vs_pred_irg_list = [];
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

    pred_uncertainty_list = []

    # diagnositic_prob and diagnositic_label
    nevu_prob_list = [];
    bcc_prob_list = [];
    mel_prob_list = [];
    misc_prob_list = [];
    sk_prob_list = [];
    nevu_pred_list = [];
    bcc_pred_list = [];
    mel_pred_list = [];
    misc_pred_list = [];
    sk_pred_list = [];

    nevu_label_list = [];
    bcc_label_list = [];
    mel_label_list = [];
    misc_label_list = [];
    sk_label_list = []
    seven_point_feature_list = []

    #if data_mode == "metadata":
     #   clf,model_gbr = multi_classifier_train()

    for index_num in tqdm(test_index_list):
        img_info = df[index_num:index_num + 1]
        clinic_path = img_info['clinic']
        dermoscopy_path = img_info['derm']
        source_dir = './release_v0/release_v0/images/'
        clinic_img = cv2.imread(source_dir + clinic_path[index_num])
        dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])

        #metadata chosen here!
        meta_data_one_hot,meta_data = encode_meta_choosed_label(img_info, index_num)
        meta_data_one_hot = np.array([meta_data_one_hot])
        meta_data = np.array([meta_data])

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
            meta_data_one_hot = torch.tensor(np.array([meta_data_one_hot,meta_data_one_hot,meta_data_one_hot,meta_data_one_hot]))
            meta_data = torch.tensor(np.array([meta_data,meta_data,meta_data,meta_data]))

            dermoscopy_img_total = np.array([dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf])
            clinic_img_total = np.array([clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf])
        elif TTA == 6:
            meta_data_one_hot = torch.tensor(
                np.array([meta_data_one_hot, meta_data_one_hot, meta_data_one_hot, meta_data_one_hot,meta_data_one_hot,meta_data_one_hot]))
            meta_data = torch.tensor(np.array([meta_data, meta_data, meta_data, meta_data,meta_data,meta_data]))

            dermoscopy_img_total = np.array(
                [dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf, dermoscopy_img_90,
                 dermoscopy_img_270])
            clinic_img_total = np.array(
                [clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf, clinic_img_90, clinic_img_270])

        dermoscopy_img_tensor = torch.from_numpy(np.transpose(dermoscopy_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255
        clinic_img_tensor = torch.from_numpy(np.transpose(clinic_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255

        if img_type == "clic":
            [(logit_diagnosis11, logit_pn11, logit_str11, logit_pig11, logit_rs11, logit_dag11, logit_bwv11,
            logit_vs11,logit_uncertainty11)] = net((clinic_img_tensor).cuda())
        elif img_type == "derm":
            [(logit_diagnosis22, logit_pn22, logit_str22, logit_pig22, logit_rs22, logit_dag22, logit_bwv22,
            logit_vs22,logit_uncertainty22)] = net((dermoscopy_img_tensor).cuda())
        else:
            [(logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic,
              logit_bwv_clic, logit_vs_clic, logit_uncertainty_clic),
             (logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm,
              logit_bwv_derm, logit_vs_derm, logit_uncertainty_derm),
             (logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs,
              logit_uncertainty),
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
            logit_uncertainty = logit_uncertainty11

        elif img_type == "derm":
            logit_diagnosis = logit_diagnosis22
            logit_pn = logit_pn22
            logit_str = logit_str22
            logit_pig = logit_pig22
            logit_rs = logit_rs22
            logit_dag = logit_dag22
            logit_bwv = logit_bwv22
            logit_vs = logit_vs22
            logit_uncertainty = logit_uncertainty22
        else:
            logit_diagnosis = logit_diagnosis
            logit_pn = logit_pn
            logit_str = logit_str
            logit_pig = logit_pig
            logit_rs = logit_rs
            logit_dag = logit_dag
            logit_bwv = logit_bwv
            logit_vs = logit_vs
            logit_uncertainty = logit_uncertainty

        if data_mode == "metadata":
            logit_diagnosis = torch.nn.Softmax(dim=1)(logit_diagnosis)
            logit_pn = torch.nn.Softmax(dim=1)(logit_pn)
            logit_str = torch.nn.Softmax(dim=1)(logit_pn)
            logit_pig = torch.nn.Softmax(dim=1)(logit_pig)
            logit_rs = torch.nn.Softmax(dim=1)(logit_rs)
            logit_dag = torch.nn.Softmax(dim=1)(logit_dag)
            logit_bwv = torch.nn.Softmax(dim=1)(logit_bwv)
            logit_vs = torch.nn.Softmax(dim=1)(logit_vs)

            meta_data_one_hot = meta_data_one_hot.squeeze()
            meta_data_one_hot = torch.cat(
                [meta_data_one_hot.float().cuda(), logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs],
                dim=1)
            meta_data_one_hot = meta_data_one_hot.unsqueeze(dim=1)
            [(logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs,
              logit_uncertainty)] = net2((meta_data_one_hot.float().cuda()))
            logit_diagnosis = logit_diagnosis
            logit_pn = logit_pn
            logit_str = logit_str
            logit_pig = logit_pig
            logit_rs = logit_rs
            logit_dag = logit_dag
            logit_bwv = logit_bwv
            logit_vs = logit_vs
            logit_uncertainty = logit_uncertainty

        # diagnostic_pred
        #print("pred1:", logit_diagnosis.shape)
        pred = softmax(logit_diagnosis.detach().cpu().numpy());
        pred = np.mean(pred, 0);
        #print("pred1_2:", pred.shape)
        pred_ = np.argmax(pred)
        pred_one_hot = to_categorical(pred_,5)
        #print("pred_:",pred_)
        nevu_prob = pred[0];
        bcc_prob = pred[1];
        mel_prob = pred[2];
        misc_prob = pred[3];
        sk_prob = pred[4];
        nevu_pred = pred_one_hot[0];
        bcc_pred = pred_one_hot[1];
        mel_pred = pred_one_hot[2];
        misc_pred = pred_one_hot[3];
        sk_pred = pred_one_hot[4];

        # pn_prob
        pn_pred = softmax(logit_pn.detach().cpu().numpy());
        pn_pred = np.mean(pn_pred, 0);
        pn_pred_ = np.argmax(pn_pred)
        pn_pred_one_hot = to_categorical(pn_pred_,3)
        pn_prob_asb = pn_pred[0];
        pn_prob_typ = pn_pred[1];
        pn_prob_asp = pn_pred[2];
        pn_pred_asb = pn_pred_one_hot[0];
        pn_pred_typ = pn_pred_one_hot[1];
        pn_pred_asp = pn_pred_one_hot[2];

        # str_prob
        str_pred = softmax(logit_str.detach().cpu().numpy())
        str_pred = np.mean(str_pred, 0);
        str_pred_ = np.argmax(str_pred)
        str_pred_one_hot = to_categorical(str_pred_,3)
        str_prob_asb = str_pred[0];
        str_prob_reg = str_pred[1];
        str_prob_irg = str_pred[2];
        str_pred_asb = str_pred_one_hot[0];
        str_pred_reg = str_pred_one_hot[1];
        str_pred_irg = str_pred_one_hot[2];

        # pig_prob
        pig_pred = softmax(logit_pig.detach().cpu().numpy())
        pig_pred = np.mean(pig_pred, 0);
        pig_pred_ = np.argmax(pig_pred)
        pig_pred_one_hot = to_categorical(pig_pred_,3)
        pig_prob_asb = pig_pred[0];
        pig_prob_reg = pig_pred[1];
        pig_prob_irg = pig_pred[2];
        pig_pred_asb = pig_pred_one_hot[0];
        pig_pred_reg = pig_pred_one_hot[1];
        pig_pred_irg = pig_pred_one_hot[2];

        # rs_prob
        rs_pred = softmax(logit_rs.detach().cpu().numpy())
        rs_pred = np.mean(rs_pred, 0);
        rs_pred_ = np.argmax(rs_pred)
        rs_pred_one_hot = to_categorical(rs_pred_,2)
        rs_prob_asb = rs_pred[0];
        rs_prob_prs = rs_pred[1];
        rs_pred_asb = rs_pred_one_hot[0];
        rs_pred_prs = rs_pred_one_hot[1];

        # dag_prob
        dag_pred = softmax(logit_dag.detach().cpu().numpy());
        dag_pred = np.mean(dag_pred, 0);
        dag_pred_ = np.argmax(dag_pred)
        dag_pred_one_hot = to_categorical(dag_pred_,3)
        dag_prob_asb = dag_pred[0];
        dag_prob_reg = dag_pred[1];
        dag_prob_irg = dag_pred[2];
        dag_pred_asb = dag_pred_one_hot[0];
        dag_pred_reg = dag_pred_one_hot[1];
        dag_pred_irg = dag_pred_one_hot[2];

        # bwv_prob
        bwv_pred = softmax(logit_bwv.detach().cpu().numpy());
        bwv_pred = np.mean(bwv_pred, 0);
        bwv_pred_ = np.argmax(bwv_pred)
        bwv_pred_one_hot = to_categorical(bwv_pred_,2)
        bwv_prob_asb = bwv_pred[0];
        bwv_prob_prs = bwv_pred[1];
        bwv_pred_asb = bwv_pred_one_hot[0];
        bwv_pred_prs = bwv_pred_one_hot[1];

        # vs_prob
        vs_pred = softmax(logit_vs.detach().cpu().numpy());
        #print("vs_pred.shape:",vs_pred.shape)
        vs_pred = np.mean(vs_pred, 0);
        vs_pred_ = np.argmax(vs_pred)
        vs_pred_one_hot = to_categorical(vs_pred_,3)
        vs_prob_asb = vs_pred[0];
        vs_prob_reg = vs_pred[1];
        vs_prob_irg = vs_pred[2];
        vs_pred_asb = vs_pred_one_hot[0];
        vs_pred_reg = vs_pred_one_hot[1];
        vs_pred_irg = vs_pred_one_hot[2];

        #uncertainty
        #print("logit_uncertainty:",logit_uncertainty)
        if data_mode == "metadata":
            uncertainty_pred =nn.Sigmoid()(logit_uncertainty);
            uncertainty_pred = uncertainty_pred.squeeze(0);
            uncertainty_pred = np.array(uncertainty_pred.detach().cpu().numpy());
            uncertainty_pred = np.mean(uncertainty_pred, 0);

        else:
            uncertainty_pred = nn.Sigmoid()(logit_uncertainty);
            uncertainty_pred = np.array(uncertainty_pred.detach().cpu().numpy());
            uncertainty_pred = np.mean(uncertainty_pred,0);
        #print("uncertainty_pred.shape:",uncertainty_pred.shape)

        seven_point_feature_list.append(
            np.concatenate([pred, pn_pred, str_pred, pig_pred, rs_pred, dag_pred, bwv_pred, vs_pred,uncertainty_pred], 0))
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

        #uncertainty
        pred_uncertainty_list.append(uncertainty_pred)
        # diagnostic_label
        pred_list.append(pred_);
        prob_list.append(pred);
        gt_list.append(diagnosis_label);
        nevu_prob_list.append(nevu_prob);
        bcc_prob_list.append(bcc_prob);
        mel_prob_list.append(mel_prob);
        misc_prob_list.append(misc_prob);
        sk_prob_list.append(sk_prob);

        nevu_pred_list.append(nevu_pred);
        bcc_pred_list.append(bcc_pred);
        mel_pred_list.append(mel_pred);
        misc_pred_list.append(misc_pred);
        sk_pred_list.append(sk_pred);

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

        pn_pred_typ_list.append(pn_pred_typ);
        pn_pred_asp_list.append(pn_pred_asp);
        pn_pred_asb_list.append(pn_pred_asb);

        pn_label_asb_list.append(pigment_network_label_one_hot[0]);
        pn_label_typ_list.append(pigment_network_label_one_hot[1]);
        pn_label_asp_list.append(pigment_network_label_one_hot[2]);

        # str_label
        str_pred_list.append(str_pred_);
        str_prob_list.append(str_pred);

        str_label_list.append(streaks_label)
        str_prob_reg_list.append(str_prob_reg);
        str_prob_irg_list.append(str_prob_irg);
        str_prob_asb_list.append(str_prob_asb);

        str_pred_reg_list.append(str_pred_reg);
        str_pred_irg_list.append(str_pred_irg);
        str_pred_asb_list.append(str_pred_asb);

        str_label_asb_list.append(streaks_label_one_hot[0]);
        str_label_reg_list.append(streaks_label_one_hot[1]);
        str_label_irg_list.append(streaks_label_one_hot[2])

        # pig_label
        pig_pred_list.append(pig_pred_);
        pig_prob_list.append(pig_pred);
        pig_label_list.append(pigmentation_label)

        pig_prob_reg_list.append(pig_prob_reg);
        pig_prob_irg_list.append(pig_prob_irg);
        pig_prob_asb_list.append(pig_prob_asb);

        pig_pred_reg_list.append(pig_pred_reg);
        pig_pred_irg_list.append(pig_pred_irg);
        pig_pred_asb_list.append(pig_pred_asb);

        pig_label_asb_list.append(pigmentation_label_one_hot[0]);
        pig_label_reg_list.append(pigmentation_label_one_hot[1]);
        pig_label_irg_list.append(pigmentation_label_one_hot[2])

        # rs_label
        rs_pred_list.append(rs_pred_);
        rs_prob_list.append(rs_pred);
        rs_label_list.append(regression_structures_label)

        rs_prob_asb_list.append(rs_prob_asb);
        rs_prob_prs_list.append(rs_prob_prs);

        rs_pred_asb_list.append(rs_pred_asb);
        rs_pred_prs_list.append(rs_pred_prs);

        rs_label_asb_list.append(regression_structures_label_one_hot[0]);
        rs_label_prs_list.append(regression_structures_label_one_hot[1])

        # dag_label
        dag_pred_list.append(dag_pred_);
        dag_prob_list.append(dag_pred);
        dag_label_list.append(dots_and_globules_label)

        dag_prob_reg_list.append(dag_prob_reg);
        dag_prob_irg_list.append(dag_prob_irg);
        dag_prob_asb_list.append(dag_prob_asb);

        dag_pred_reg_list.append(dag_pred_reg);
        dag_pred_irg_list.append(dag_pred_irg);
        dag_pred_asb_list.append(dag_pred_asb);

        dag_label_asb_list.append(dots_and_globules_label_one_hot[0]);
        dag_label_reg_list.append(dots_and_globules_label_one_hot[1]);
        dag_label_irg_list.append(dots_and_globules_label_one_hot[2])

        # bwv_label
        bwv_pred_list.append(bwv_pred_);
        bwv_prob_list.append(bwv_pred);
        bwv_label_list.append(blue_whitish_veil_label)

        bwv_prob_asb_list.append(bwv_prob_asb);
        bwv_prob_prs_list.append((bwv_prob_prs));

        bwv_pred_asb_list.append(bwv_pred_asb);
        bwv_pred_prs_list.append(bwv_pred_prs);

        bwv_label_asb_list.append(blue_whitish_veil_label_one_hot[0]);
        bwv_label_prs_list.append(blue_whitish_veil_label_one_hot[1])

        # vs_label
        vs_pred_list.append(vs_pred_);
        vs_prob_list.append(vs_pred);
        vs_label_list.append(vascular_structures_label)

        vs_prob_reg_list.append(vs_prob_reg);
        vs_prob_irg_list.append(vs_prob_irg);
        vs_prob_asb_list.append(vs_prob_asb);

        vs_pred_reg_list.append(vs_pred_reg);
        vs_pred_irg_list.append(vs_pred_irg);
        vs_pred_asb_list.append(vs_pred_asb);

        vs_label_asb_list.append(vascular_structures_label_one_hot[0]);
        vs_label_reg_list.append(vascular_structures_label_one_hot[1]);
        vs_label_irg_list.append(vascular_structures_label_one_hot[2])

    uncertainty_pred = np.array(pred_uncertainty_list).squeeze();

    pred = np.array(pred_list).squeeze();
    prob = np.array(prob_list).squeeze();

    gt = np.array(gt_list)
    nevu_prob = np.array(nevu_prob_list);
    bcc_prob = np.array(bcc_prob_list);
    mel_prob = np.array(mel_prob_list);
    misc_prob = np.array(misc_prob_list);
    sk_prob = np.array(sk_prob_list);

    nevu_pred = np.array(nevu_pred_list);
    bcc_pred = np.array(bcc_pred_list);
    mel_pred = np.array(mel_pred_list);
    misc_pred = np.array(misc_pred_list);
    sk_pred = np.array(sk_pred_list);

    nevu_label = np.array(nevu_label_list);
    bcc_label = np.array(bcc_label_list);
    mel_label = np.array(mel_label_list);
    misc_label = np.array(misc_label_list);
    sk_label = np.array(sk_label_list)

    #pn
    pn_pred = np.array(pn_pred_list).squeeze();
    pn_prob = np.array(pn_prob_list).squeeze();

    pn_gt = np.array(pn_label_list)
    pn_prob_typ = np.array(pn_prob_typ_list);
    pn_prob_asp = np.array(pn_prob_asp_list);
    pn_prob_asb = np.array(pn_prob_asb_list)

    pn_pred_typ = np.array(pn_pred_typ_list);
    pn_pred_asp = np.array(pn_pred_asp_list);
    pn_pred_asb = np.array(pn_pred_asb_list);

    pn_label_typ = np.array(pn_label_typ_list);
    pn_label_asp = np.array(pn_label_asp_list);
    pn_label_asb = np.array(pn_label_asb_list)

    #str
    str_pred = np.array(str_pred_list).squeeze();
    str_prob = np.array(str_prob_list).squeeze();

    str_gt = np.array(str_label_list)
    str_prob_asb = np.array(str_prob_asb_list);
    str_prob_reg = np.array(str_prob_reg_list);
    str_prob_irg = np.array(str_prob_irg_list);

    str_pred_asb = np.array(str_pred_asb_list);
    str_pred_reg = np.array(str_pred_reg_list);
    str_pred_irg = np.array(str_pred_irg_list);

    str_label_asb = np.array(str_label_asb_list);
    str_label_reg = np.array(str_label_reg_list);
    str_label_irg = np.array(str_label_irg_list)

    #pig
    pig_pred = np.array(pig_pred_list).squeeze();
    pig_prob = np.array(pig_prob_list).squeeze();

    pig_gt = np.array(pig_label_list)
    pig_prob_asb = np.array(pig_prob_asb_list);
    pig_prob_reg = np.array(pig_prob_reg_list);
    pig_prob_irg = np.array(pig_prob_irg_list);

    pig_pred_asb = np.array(pig_pred_asb_list);
    pig_pred_reg = np.array(pig_pred_reg_list);
    pig_pred_irg = np.array(pig_pred_irg_list);

    pig_label_asb = np.array(pig_label_asb_list);
    pig_label_reg = np.array(pig_label_reg_list);
    pig_label_irg = np.array(pig_label_irg_list)

    #rs
    rs_pred = np.array(rs_pred_list).squeeze();
    rs_prob = np.array(rs_prob_list).squeeze();

    rs_gt = np.array(rs_label_list)
    rs_prob_asb = np.array(rs_prob_asb_list);
    rs_prob_prs = np.array(rs_prob_prs_list);

    rs_pred_asb = np.array(rs_pred_asb_list);
    rs_pred_prs = np.array(rs_pred_prs_list);

    rs_label_asb = np.array(rs_label_asb_list);
    rs_label_prs = np.array(rs_label_prs_list)

    #dag
    dag_pred = np.array(dag_pred_list).squeeze();
    dag_prob = np.array(dag_prob_list).squeeze();

    dag_gt = np.array(dag_label_list)
    dag_prob_asb = np.array(dag_prob_asb_list);
    dag_prob_reg = np.array(dag_prob_reg_list);
    dag_prob_irg = np.array(dag_prob_irg_list);

    dag_pred_asb = np.array(dag_pred_asb_list);
    dag_pred_reg = np.array(dag_pred_reg_list);
    dag_pred_irg = np.array(dag_pred_irg_list);

    dag_label_asb = np.array(dag_label_asb_list);
    dag_label_reg = np.array(dag_label_reg_list);
    dag_label_irg = np.array(dag_label_irg_list)

    #bwv
    bwv_pred = np.array(bwv_pred_list).squeeze();
    bwv_prob = np.array(bwv_prob_list).squeeze();

    bwv_gt = np.array(bwv_label_list)
    bwv_prob_asb = np.array(bwv_prob_asb_list);
    bwv_prob_prs = np.array(bwv_prob_prs_list);

    bwv_pred_asb = np.array(bwv_pred_asb_list);
    bwv_pred_prs = np.array(bwv_pred_prs_list);

    bwv_label_asb = np.array(bwv_label_asb_list);
    bwv_label_prs = np.array(bwv_label_prs_list)

    #vs
    vs_pred = np.array(vs_pred_list).squeeze();
    vs_prob = np.array(vs_prob_list).squeeze();

    vs_gt = np.array(vs_label_list)
    vs_prob_asb = np.array(vs_prob_asb_list);
    vs_prob_reg = np.array(vs_prob_reg_list);
    vs_prob_irg = np.array(vs_prob_irg_list);

    vs_pred_asb = np.array(vs_pred_asb_list);
    vs_pred_reg = np.array(vs_pred_reg_list);
    vs_pred_irg = np.array(vs_pred_irg_list);

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

    #calculate SEN, SPE, PRE
    cm_Pn_asb = sklearn.metrics.confusion_matrix((np.array(pn_label_asb) * 1).flatten(), pn_pred_asb.flatten())
    Tn_pn_asb, Fp_pn_asb, Fn_pn_asb, Tp_pn_asb = cm_Pn_asb.ravel()
    cm_Pn_typ = sklearn.metrics.confusion_matrix((np.array(pn_label_typ) * 1).flatten(), pn_pred_typ.flatten())
    Tn_pn_typ, Fp_pn_typ, Fn_pn_typ, Tp_pn_typ = cm_Pn_typ.ravel()
    cm_Pn_aty = sklearn.metrics.confusion_matrix((np.array(pn_label_asp) * 1).flatten(), pn_pred_asp.flatten())
    Tn_pn_aty, Fp_pn_aty, Fn_pn_aty, Tp_pn_aty = cm_Pn_aty.ravel()

    cm_Str_asb = sklearn.metrics.confusion_matrix((np.array(str_label_asb) * 1).flatten(), str_pred_asb.flatten())
    Tn_str_asb, Fp_str_asb, Fn_str_asb, Tp_str_asb = cm_Str_asb.ravel()
    cm_Str_reg = sklearn.metrics.confusion_matrix((np.array(str_label_reg) * 1).flatten(), str_pred_reg.flatten())
    Tn_str_reg, Fp_str_reg, Fn_str_reg, Tp_str_reg = cm_Str_reg.ravel()
    cm_Str_irg = sklearn.metrics.confusion_matrix((np.array(str_label_irg) * 1).flatten(), str_pred_irg.flatten())
    Tn_str_irg, Fp_str_irg, Fn_str_irg, Tp_str_irg = cm_Str_irg.ravel()

    cm_Pig_asb = sklearn.metrics.confusion_matrix((np.array(pig_label_asb) * 1).flatten(), pig_pred_asb.flatten())
    Tn_pig_asb, Fp_pig_asb, Fn_pig_asb, Tp_pig_asb = cm_Pig_asb.ravel()
    cm_Pig_reg = sklearn.metrics.confusion_matrix((np.array(pig_label_reg) * 1).flatten(), pig_pred_reg.flatten())
    Tn_pig_reg, Fp_pig_reg, Fn_pig_reg, Tp_pig_reg = cm_Pig_reg.ravel()
    cm_Pig_irg = sklearn.metrics.confusion_matrix((np.array(pig_label_irg) * 1).flatten(), pig_pred_irg.flatten())
    Tn_pig_irg, Fp_pig_irg, Fn_pig_irg, Tp_pig_irg = cm_Pig_irg.ravel()

    cm_Rs_asb = sklearn.metrics.confusion_matrix((np.array(rs_label_asb) * 1).flatten(), rs_pred_asb.flatten())
    Tn_rs_asb, Fp_rs_asb, Fn_rs_asb, Tp_rs_asb = cm_Rs_asb.ravel()
    cm_Rs_prs = sklearn.metrics.confusion_matrix((np.array(rs_label_prs) * 1).flatten(), rs_pred_prs.flatten())
    Tn_rs_prs, Fp_rs_prs, Fn_rs_prs, Tp_rs_prs = cm_Rs_prs.ravel()

    cm_Vs_asb = sklearn.metrics.confusion_matrix((np.array(vs_label_asb) * 1).flatten(), vs_pred_asb.flatten())
    Tn_vs_asb, Fp_vs_asb, Fn_vs_asb, Tp_vs_asb = cm_Vs_asb.ravel()
    cm_Vs_reg = sklearn.metrics.confusion_matrix((np.array(vs_label_reg) * 1).flatten(), vs_pred_reg.flatten())
    Tn_vs_reg, Fp_vs_reg, Fn_vs_reg, Tp_vs_reg = cm_Vs_reg.ravel()
    cm_Vs_irg = sklearn.metrics.confusion_matrix((np.array(vs_label_irg) * 1).flatten(), vs_pred_irg.flatten())
    Tn_vs_irg, Fp_vs_irg, Fn_vs_irg, Tp_vs_irg = cm_Vs_irg.ravel()

    cm_Bwv_asb = sklearn.metrics.confusion_matrix((np.array(bwv_label_asb) * 1).flatten(), bwv_pred_asb.flatten())
    Tn_bwv_asb, Fp_bwv_asb, Fn_bwv_asb, Tp_bwv_asb = cm_Bwv_asb.ravel()
    cm_Bwv_prs = sklearn.metrics.confusion_matrix((np.array(bwv_label_prs) * 1).flatten(), bwv_pred_prs.flatten())
    Tn_bwv_prs, Fp_bwv_prs, Fn_bwv_prs, Tp_bwv_prs = cm_Bwv_prs.ravel()

    cm_Dag_asb = sklearn.metrics.confusion_matrix((np.array(dag_label_asb) * 1).flatten(), dag_pred_asb.flatten())
    Tn_dag_asb, Fp_dag_asb, Fn_dag_asb, Tp_dag_asb = cm_Dag_asb.ravel()
    cm_Dag_reg = sklearn.metrics.confusion_matrix((np.array(dag_label_reg) * 1).flatten(), dag_pred_reg.flatten())
    Tn_dag_reg, Fp_dag_reg, Fn_dag_reg, Tp_dag_reg = cm_Dag_reg.ravel()
    cm_Dag_irg = sklearn.metrics.confusion_matrix((np.array(dag_label_irg) * 1).flatten(), dag_pred_irg.flatten())
    Tn_dag_irg, Fp_dag_irg, Fn_dag_irg, Tp_dag_irg = cm_Dag_irg.ravel()

    cm_Diag_nev = sklearn.metrics.confusion_matrix((np.array(nevu_label) * 1).flatten(), nevu_pred.flatten())
    Tn_diag_nev, Fp_diag_nev, Fn_diag_nev, Tp_diag_nev = cm_Diag_nev.ravel()
    cm_Diag_bcc = sklearn.metrics.confusion_matrix((np.array(bcc_label) * 1).flatten(), bcc_pred.flatten())
    Tn_diag_bcc, Fp_diag_bcc, Fn_diag_bcc, Tp_diag_bcc = cm_Diag_bcc.ravel()
    cm_Diag_mel = sklearn.metrics.confusion_matrix((np.array(mel_label) * 1).flatten(), mel_pred.flatten())
    Tn_diag_mel, Fp_diag_mel, Fn_diag_mel, Tp_diag_mel = cm_Diag_mel.ravel()
    cm_Diag_misc = sklearn.metrics.confusion_matrix((np.array(misc_label) * 1).flatten(), misc_pred.flatten())
    Tn_diag_misc, Fp_diag_misc, Fn_diag_misc, Tp_diag_misc = cm_Diag_misc.ravel()
    cm_Diag_sk = sklearn.metrics.confusion_matrix((np.array(sk_label) * 1).flatten(), sk_pred.flatten())
    Tn_diag_sk, Fp_diag_sk, Fn_diag_sk, Tp_diag_sk = cm_Diag_sk.ravel()


    #SEN
    SEN_pn_asb = Tp_pn_asb / (Tp_pn_asb + Fn_pn_asb)
    SEN_pn_typ = Tp_pn_typ / (Tp_pn_typ + Fn_pn_typ)
    SEN_pn_aty = Tp_pn_aty / (Tp_pn_aty + Fn_pn_aty)

    SEN_str_asb = Tp_str_asb / (Tp_str_asb + Fn_str_asb)
    SEN_str_reg = Tp_str_reg / (Tp_str_reg + Fn_str_reg)
    SEN_str_irg = Tp_str_irg / (Tp_str_irg + Fn_str_irg)

    SEN_pig_asb = Tp_pig_asb / (Tp_pig_asb + Fn_pig_asb)
    SEN_pig_reg = Tp_pig_reg / (Tp_pig_reg + Fn_pig_reg)
    SEN_pig_irg = Tp_pig_irg / (Tp_pig_irg + Fn_pig_irg)

    SEN_rs_asb = Tp_rs_asb / (Tp_rs_asb + Fn_rs_asb)
    SEN_rs_prs = Tp_rs_prs / (Tp_rs_prs + Fn_rs_prs)

    SEN_vs_asb = Tp_vs_asb / (Tp_vs_asb + Fn_vs_asb)
    SEN_vs_reg = Tp_vs_reg / (Tp_vs_reg + Fn_vs_reg)
    SEN_vs_irg = Tp_vs_irg / (Tp_vs_irg + Fn_vs_irg)

    SEN_bwv_asb = Tp_bwv_asb / (Tp_bwv_asb + Fn_bwv_asb)
    SEN_bwv_prs = Tp_bwv_prs / (Tp_bwv_prs + Fn_bwv_prs)

    SEN_dag_asb = Tp_dag_asb / (Tp_dag_asb + Fn_dag_asb)
    SEN_dag_reg = Tp_dag_reg / (Tp_dag_reg + Fn_dag_reg)
    SEN_dag_irg = Tp_dag_irg / (Tp_dag_irg + Fn_dag_irg)

    SEN_diag_nev = Tp_diag_nev / (Tp_diag_nev + Fn_diag_nev)
    SEN_diag_bcc = Tp_diag_bcc / (Tp_diag_bcc + Fn_diag_bcc)
    SEN_diag_mel = Tp_diag_mel / (Tp_diag_mel + Fn_diag_mel)
    SEN_diag_misc = Tp_diag_misc / (Tp_diag_misc + Fn_diag_misc)
    SEN_diag_sk = Tp_diag_sk / (Tp_diag_sk + Fn_diag_sk)


    #SPE
    SPE_pn_asb = Tn_pn_asb / (Tn_pn_asb + Fp_pn_asb)
    SPE_pn_typ = Tn_pn_typ / (Tn_pn_typ + Fp_pn_typ)
    SPE_pn_aty = Tn_pn_aty / (Tn_pn_aty + Fp_pn_aty)

    SPE_str_asb = Tn_str_asb / (Tn_str_asb + Fp_str_asb)
    SPE_str_reg = Tn_str_reg / (Tn_str_reg + Fp_str_reg)
    SPE_str_irg = Tn_str_irg / (Tn_str_irg + Fp_str_irg)

    SPE_pig_asb = Tn_pig_asb / (Tn_pig_asb + Fp_pig_asb)
    SPE_pig_reg = Tn_pig_reg / (Tn_pig_reg + Fp_pig_reg)
    SPE_pig_irg = Tn_pig_irg / (Tn_pig_irg + Fp_pig_irg)

    SPE_rs_asb = Tn_rs_asb / (Tn_rs_asb + Fp_rs_asb)
    SPE_rs_prs = Tn_rs_prs / (Tn_rs_prs + Fp_rs_prs)

    SPE_vs_asb = Tn_vs_asb / (Tn_vs_asb + Fp_vs_asb)
    SPE_vs_reg = Tn_vs_reg / (Tn_vs_reg + Fp_vs_reg)
    SPE_vs_irg = Tn_vs_irg / (Tn_vs_irg + Fp_vs_irg)

    SPE_bwv_asb = Tn_bwv_asb / (Tn_bwv_asb + Fp_bwv_asb)
    SPE_bwv_prs = Tn_bwv_prs / (Tn_bwv_prs + Fp_bwv_prs)

    SPE_dag_asb = Tn_dag_asb / (Tn_dag_asb + Fp_dag_asb)
    SPE_dag_reg = Tn_dag_reg / (Tn_dag_reg + Fp_dag_reg)
    SPE_dag_irg = Tn_dag_irg / (Tn_dag_irg + Fp_dag_irg)

    SPE_diag_nev = Tn_diag_nev / (Tn_diag_nev + Fp_diag_nev)
    SPE_diag_bcc = Tn_diag_bcc / (Tn_diag_bcc + Fp_diag_bcc)
    SPE_diag_mel = Tn_diag_mel / (Tn_diag_mel + Fp_diag_mel)
    SPE_diag_misc = Tn_diag_misc / (Tn_diag_misc + Fp_diag_misc)
    SPE_diag_sk = Tn_diag_sk / (Tn_diag_sk + Fp_diag_sk)

    #PRE
    PRE_pn_asb = Tp_pn_asb / (Tp_pn_asb + Fp_pn_asb)
    PRE_pn_typ = Tp_pn_typ / (Tp_pn_typ + Fp_pn_typ)
    PRE_pn_aty = Tp_pn_aty / (Tp_pn_aty + Fp_pn_aty)

    PRE_str_asb = Tp_str_asb / (Tp_str_asb + Fp_str_asb)
    PRE_str_reg = Tp_str_reg / (Tp_str_reg + Fp_str_reg)
    PRE_str_irg = Tp_str_irg / (Tp_str_irg + Fp_str_irg)

    PRE_pig_asb = Tp_pig_asb / (Tp_pig_asb + Fp_pig_asb)
    PRE_pig_reg = Tp_pig_reg / (Tp_pig_reg + Fp_pig_reg)
    PRE_pig_irg = Tp_pig_irg / (Tp_pig_irg + Fp_pig_irg)

    PRE_rs_asb = Tp_rs_asb / (Tp_rs_asb + Fp_rs_asb)
    PRE_rs_prs = Tp_rs_prs / (Tp_rs_prs + Fp_rs_prs)

    PRE_vs_asb = Tp_vs_asb / (Tp_vs_asb + Fp_vs_asb)
    PRE_vs_reg = Tp_vs_reg / (Tp_vs_reg + Fp_vs_reg)
    PRE_vs_irg = Tp_vs_irg / (Tp_vs_irg + Fp_vs_irg)

    PRE_bwv_asb = Tp_bwv_asb / (Tp_bwv_asb + Fp_bwv_asb)
    PRE_bwv_prs = Tp_bwv_prs / (Tp_bwv_prs + Fp_bwv_prs)

    PRE_dag_asb = Tp_dag_asb / (Tp_dag_asb + Fp_dag_asb)
    PRE_dag_reg = Tp_dag_reg / (Tp_dag_reg + Fp_dag_reg)
    PRE_dag_irg = Tp_dag_irg / (Tp_dag_irg + Fp_dag_irg)

    PRE_diag_nev = Tp_diag_nev / (Tp_diag_nev + Fp_diag_nev)
    PRE_diag_bcc = Tp_diag_bcc / (Tp_diag_bcc + Fp_diag_bcc)
    PRE_diag_mel = Tp_diag_mel / (Tp_diag_mel + Fp_diag_mel)
    PRE_diag_misc = Tp_diag_misc / (Tp_diag_misc + Fp_diag_misc)
    PRE_diag_sk = Tp_diag_sk / (Tp_diag_sk + Fp_diag_sk)
    log.write('Diag' + '-' * 25 + "\n")
    log.write('nevu_SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_nev,SPE_diag_nev,PRE_diag_nev))
    log.write('bcc SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_bcc,SPE_diag_bcc,PRE_diag_bcc))
    log.write('mel SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_mel,SPE_diag_mel,PRE_diag_mel))
    log.write('misc SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_misc,SPE_diag_misc,PRE_diag_misc))
    log.write('sk SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_sk,SPE_diag_sk,PRE_diag_sk))
    log.write('-' * 10 + '\n')
    log.write('pn_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_pn_asb,SPE_pn_asb,PRE_pn_asb))
    log.write('pn_typ SEN: {}, SPE: {}, PRE: {}'.format(SEN_pn_typ,SPE_pn_typ,PRE_pn_typ))
    log.write('pn_aty SEN: {}, SPE: {}, PRE: {}'.format(SEN_pn_aty,SPE_pn_aty,PRE_pn_aty))
    log.write('-' * 10 +'\n')
    log.write('str_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_str_asb,SPE_str_asb,PRE_str_asb))
    log.write('str_reg SEN: {}, SPE: {}, PRE: {}'.format(SEN_str_reg,SPE_str_reg,PRE_str_reg))
    log.write('str_irg SEN: {}, SPE: {}, PRE: {}'.format(SEN_str_irg,SPE_str_irg,PRE_str_irg))
    log.write('-' * 10 + '\n')
    log.write('pig_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_pig_asb,SPE_pig_asb,PRE_pig_asb))
    log.write('pig_reg SEN: {}, SPE: {}, PRE: {}'.format(SEN_pig_reg,SPE_pig_reg,PRE_pig_reg))
    log.write('pig_irg SEN: {}, SPE: {}, PRE: {}'.format(SEN_pig_irg,SPE_pig_irg,PRE_pig_irg))
    log.write('-' * 10 + '\n')
    log.write('rs_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_rs_asb,SPE_rs_asb,PRE_rs_asb))
    log.write('rs_prs SEN: {}, SPE: {}, PRE: {}'.format(SEN_rs_prs,SPE_rs_prs,PRE_rs_prs))
    log.write('-' * 10 + '\n')
    log.write('vs_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_vs_asb,SPE_vs_asb,PRE_vs_asb))
    log.write('vs_reg SEN: {}, SPE: {}， PRE: {}'.format(SEN_vs_reg,SPE_vs_reg,PRE_vs_reg))
    log.write('vs_irg SEN: {}, SPE: {}, PRE: {}'.format(SEN_vs_irg,SPE_vs_irg,PRE_vs_irg))
    log.write('-' * 10 + '\n')
    log.write('bwv_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_bwv_asb,SPE_bwv_asb,PRE_bwv_asb))
    log.write('bwv_prs SEN: {}, SPE: {}, PRE: {}'.format(SEN_bwv_prs,SPE_bwv_prs,PRE_bwv_prs))
    log.write('-' * 10 + '\n')
    log.write('dag_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_dag_asb,SPE_dag_asb,PRE_dag_asb))
    log.write('dag_reg SEN: {}, SPE: {}, PRE: {}'.format(SEN_dag_reg,SPE_dag_reg,PRE_dag_reg))
    log.write('dag_irg SEN: {}, SPE: {}, PRE: {}'.format(SEN_dag_irg,SPE_dag_irg,PRE_dag_irg))

    #AUC
    nevu_auc = roc_auc_score((np.array(nevu_label) * 1).flatten(), nevu_prob.flatten())
    bcc_auc = roc_auc_score((np.array(bcc_label) * 1).flatten(), bcc_prob.flatten())
    mel_auc = roc_auc_score((np.array(mel_label) * 1).flatten(), mel_prob.flatten())
    misc_auc = roc_auc_score((np.array(misc_label) * 1).flatten(), misc_prob.flatten())
    sk_auc = roc_auc_score((np.array(sk_label) * 1).flatten(), sk_prob.flatten())
    log.write('-' * 25 + '\n')
    log.write('diag_Bcc: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_bcc, SPE_diag_bcc, PRE_diag_bcc, bcc_auc))
    log.write('diag_Mel: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_mel, SPE_diag_mel, PRE_diag_mel, mel_auc))
    log.write('diag_Nev: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_nev, SPE_diag_nev, PRE_diag_nev, nevu_auc))
    log.write('diag_Misc: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_misc, SPE_diag_misc, PRE_diag_misc, misc_auc))
    log.write('diag_Sk: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_sk, SPE_diag_sk, PRE_diag_sk, sk_auc))
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

    return avg_acc, uncertainty_pred,[prob, pn_prob, str_prob, pig_prob, rs_prob, dag_prob, bwv_prob, vs_prob], [
        np.array(nevu_label), np.array(bcc_label), np.array(mel_label), np.array(misc_label), np.array(sk_label)], [
               nevu_prob, bcc_prob, mel_prob, misc_prob, sk_prob], seven_point_feature_list, [gt, pn_gt,
                                                                                              str_gt, pig_gt,
                                                                                              rs_gt, dag_gt,
                                                                                              bwv_gt, vs_gt]


def predict3(net,net2, test_index_list, df, model_name, out_dir, mode, TTA=4, size=224, img_type="clinic"):
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir + 'log.multi_modality_{}_{}_sinlesion.txt'.format(mode, model_name), mode='w')
    log.write('\n--- [START %s] %s\n\n' % ('IDENTIFIER', '-' * 64))
    net.set_mode('valid')
    net2.set_mode('valid')
    # 7-point score
    # prob #pred

    # 1 pigment_network
    pn_prob_typ_list = [];
    pn_prob_asp_list = [];
    pn_prob_asb_list = [];
    pn_pred_typ_list = [];
    pn_pred_asp_list = [];
    pn_pred_asb_list = [];
    pn_pred_list = []
    pn_prob_list = []
    # 2 streak
    str_prob_asb_list = [];
    str_prob_reg_list = [];
    str_prob_irg_list = [];
    str_pred_asb_list = [];
    str_pred_reg_list = [];
    str_pred_irg_list = [];
    str_pred_list = []
    str_prob_list = []
    # 3 pigmentation
    pig_prob_asb_list = [];
    pig_prob_reg_list = [];
    pig_prob_irg_list = [];
    pig_pred_asb_list = [];
    pig_pred_reg_list = [];
    pig_pred_irg_list = [];
    pig_pred_list = []
    pig_prob_list = []
    # 4 regression structure
    rs_prob_asb_list = [];
    rs_prob_prs_list = [];
    rs_pred_asb_list = [];
    rs_pred_prs_list = [];
    rs_pred_list = []
    rs_prob_list = []
    # 5 dots and globules
    dag_prob_asb_list = [];
    dag_prob_reg_list = [];
    dag_prob_irg_list = [];
    dag_pred_asb_list = [];
    dag_pred_reg_list = [];
    dag_pred_irg_list = [];
    dag_pred_list = []
    dag_prob_list = []
    # 6 blue whitish veil 1
    bwv_prob_asb_list = [];
    bwv_prob_prs_list = [];
    bwv_pred_asb_list = [];
    bwv_pred_prs_list = [];
    bwv_pred_list = []
    bwv_prob_list = []
    # 7 vascular strucuture
    vs_prob_asb_list = [];
    vs_prob_reg_list = [];
    vs_prob_irg_list = [];
    vs_pred_asb_list = [];
    vs_pred_reg_list = [];
    vs_pred_irg_list = [];
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

    pred_uncertainty_list = []

    # diagnositic_prob and diagnositic_label
    nevu_prob_list = [];
    bcc_prob_list = [];
    mel_prob_list = [];
    misc_prob_list = [];
    sk_prob_list = [];
    nevu_pred_list = [];
    bcc_pred_list = [];
    mel_pred_list = [];
    misc_pred_list = [];
    sk_pred_list = [];

    nevu_label_list = [];
    bcc_label_list = [];
    mel_label_list = [];
    misc_label_list = [];
    sk_label_list = []
    seven_point_feature_list = []

    fusion_choice = ['0', '1', '2']
    for i in fusion_choice:
        X = []
        Y = []
        print("testing index:",i)
        count = 0
        C = []
        for index_num in tqdm(test_index_list):
            count= count + 1
            C.append(count)
            img_info = df[index_num:index_num + 1]
            clinic_path = img_info['clinic']
            dermoscopy_path = img_info['derm']
            source_dir = './release_v0/release_v0/images/'
            clinic_img = cv2.imread(source_dir + clinic_path[index_num])
            dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])

            #metadata chosen here!
            meta_data_one_hot, meta_data = encode_meta_choosed_label(img_info, index_num)
            meta_data_one_hot = np.array([meta_data_one_hot])
            meta_data = np.array([meta_data])

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
                meta_data_one_hot = torch.tensor(np.array([meta_data_one_hot,meta_data_one_hot,meta_data_one_hot,meta_data_one_hot]))
                meta_data = torch.tensor(np.array([meta_data,meta_data,meta_data,meta_data]))

                dermoscopy_img_total = np.array([dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf])
                clinic_img_total = np.array([clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf])
            elif TTA == 6:
                meta_data_one_hot = torch.tensor(
                    np.array([meta_data_one_hot, meta_data_one_hot, meta_data_one_hot, meta_data_one_hot,meta_data_one_hot,meta_data_one_hot]))
                meta_data = torch.tensor(np.array([meta_data, meta_data, meta_data, meta_data,meta_data,meta_data]))

                dermoscopy_img_total = np.array(
                    [dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf, dermoscopy_img_90,
                     dermoscopy_img_270])
                clinic_img_total = np.array(
                    [clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf, clinic_img_90, clinic_img_270])

            dermoscopy_img_tensor = torch.from_numpy(
                np.transpose(dermoscopy_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255
            clinic_img_tensor = torch.from_numpy(np.transpose(clinic_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255

            if img_type == "clic":
                [(logit_diagnosis11, logit_pn11, logit_str11, logit_pig11, logit_rs11, logit_dag11, logit_bwv11,
                 logit_vs11,logit_uncertainty11)] = net((clinic_img_tensor).cuda())
            elif img_type == "derm":
                [(logit_diagnosis22, logit_pn22, logit_str22, logit_pig22, logit_rs22, logit_dag22, logit_bwv22,
                 logit_vs22,logit_uncertainty22)] = net(((dermoscopy_img_tensor).cuda()))
            else:
                [(logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic,
                  logit_bwv_clic, logit_vs_clic, logit_uncertainty_clic),
                 (logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm,
                  logit_bwv_derm, logit_vs_derm, logit_uncertainty_derm),
                 (logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs,
                  logit_uncertainty),
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
                logit_uncertainty3_1 = logit_uncertainty11

            elif img_type == "derm":
                logit_diagnosis3_1 = logit_diagnosis22
                logit_pn3_1 = logit_pn22
                logit_str3_1 = logit_str22
                logit_pig3_1 = logit_pig22
                logit_rs3_1 = logit_rs22
                logit_dag3_1 = logit_dag22
                logit_bwv3_1 = logit_bwv22
                logit_vs3_1 = logit_vs22
                logit_uncertainty3_1 = logit_uncertainty22

            else:
                logit_diagnosis3_1 = logit_diagnosis
                logit_pn3_1 = logit_pn
                logit_str3_1 = logit_str
                logit_pig3_1 = logit_pig
                logit_rs3_1 = logit_rs
                logit_dag3_1 = logit_dag
                logit_bwv3_1 = logit_bwv
                logit_vs3_1 = logit_vs
                logit_uncertainty3_1 = logit_uncertainty

            logit_diagnosis = torch.nn.Softmax(dim=1)(logit_diagnosis3_1)
            logit_pn = torch.nn.Softmax(dim=1)(logit_pn3_1)
            logit_str = torch.nn.Softmax(dim=1)(logit_pn3_1)
            logit_pig = torch.nn.Softmax(dim=1)(logit_pig3_1)
            logit_rs = torch.nn.Softmax(dim=1)(logit_rs3_1)
            logit_dag = torch.nn.Softmax(dim=1)(logit_dag3_1)
            logit_bwv = torch.nn.Softmax(dim=1)(logit_bwv3_1)
            logit_vs = torch.nn.Softmax(dim=1)(logit_vs3_1)

            meta_data_one_hot = meta_data_one_hot.squeeze()
            meta_data_one_hot = torch.cat(
                [meta_data_one_hot.float().cuda(), logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv,
                 logit_vs],
                dim=1)
            meta_data_one_hot = meta_data_one_hot.unsqueeze(dim=1)
            [(logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs,
              logit_uncertainty)] = net2((meta_data_one_hot.float().cuda()))
            logit_diagnosis3_2 = logit_diagnosis
            logit_pn3_2 = logit_pn
            logit_str3_2 = logit_str
            logit_pig3_2 = logit_pig
            logit_rs3_2 = logit_rs
            logit_dag3_2 = logit_dag
            logit_bwv3_2 = logit_bwv
            logit_vs3_2 = logit_vs
            logit_uncertainty3_2 = logit_uncertainty

            logit_diagnosis3_1 = logit_diagnosis3_1.cpu()
            logit_pn3_1 = logit_pn3_1.cpu()
            logit_str3_1 = logit_str3_1.cpu()
            logit_pig3_1 = logit_pig3_1.cpu()
            logit_rs3_1 = logit_rs3_1.cpu()
            logit_dag3_1 = logit_dag3_1.cpu()
            logit_bwv3_1 = logit_bwv3_1.cpu()
            logit_vs3_1 = logit_vs3_1.cpu()
            logit_uncertainty3_1 = logit_uncertainty3_1.cpu()

            logit_diagnosis3_2 = logit_diagnosis3_2.cpu()
            logit_pn3_2 = logit_pn3_2.cpu()
            logit_str3_2 = logit_str3_2.cpu()
            logit_pig3_2 = logit_pig3_2.cpu()
            logit_rs3_2 = logit_rs3_2.cpu()
            logit_dag3_2 = logit_dag3_2.cpu()
            logit_bwv3_2 = logit_bwv3_2.cpu()
            logit_vs3_2 = logit_vs3_2.cpu()
            logit_uncertainty3_2 = logit_uncertainty3_2.cpu()

            #uncertainty3_1 = nn.Sigmoid()(logit_uncertainty3_1)
            uncertainty3_1 = logit_uncertainty3_1
            uncertainty3_1 = uncertainty3_1.squeeze()
            uncertainty3_1 = np.array(uncertainty3_1.detach().cpu().numpy())
            uncertainty3_1 = np.mean(uncertainty3_1,0);
            #print("uncertainty3_1.shape:",uncertainty3_1.shape)
            #uncertainty3_2 = nn.Sigmoid()(logit_uncertainty3_2)
            uncertainty3_2 = logit_uncertainty3_2
            uncertainty3_2 = uncertainty3_2.squeeze()
            uncertainty3_2 = np.array(uncertainty3_2.detach().cpu().numpy())
            uncertainty3_2 = np.mean(uncertainty3_2, 0);
            uncertainty = torch.tensor([uncertainty3_1,uncertainty3_2])
            uncertainty = nn.Softmax(dim=0)(uncertainty)
            uncertainty3_1 = np.array(uncertainty[0])
            uncertainty3_2 = np.array(uncertainty[1])
            X.append(uncertainty3_1)
            Y.append(uncertainty3_2)
            #print("uncertainty:",uncertainty3_1,uncertainty3_2)
        # diagnostic_pred
            if i == '0':
                pred3_3_1 = softmax(logit_diagnosis3_1.detach().cpu().numpy());
                pred3_3_2 = softmax(logit_diagnosis3_2.detach().cpu().numpy());
                pred3_3 = 0.5 * pred3_3_1 + 0.5 * pred3_3_2
                pred = np.mean(pred3_3, 0);
                pred_ = np.argmax(pred)
                pred_one_hot = to_categorical(pred_, 5)
                nevu_prob = pred[0];
                bcc_prob = pred[1];
                mel_prob = pred[2];
                misc_prob = pred[3];
                sk_prob = pred[4];
                nevu_pred = pred_one_hot[0];
                bcc_pred = pred_one_hot[1];
                mel_pred = pred_one_hot[2];
                misc_pred = pred_one_hot[3];
                sk_pred = pred_one_hot[4];

                # pn_prob
                pn_pred3_3_1 = softmax(logit_pn3_1.detach().cpu().numpy());
                pn_pred3_3_2 = softmax(logit_pn3_2.detach().cpu().numpy());
                pn_pred3_3 = 0.5 * pn_pred3_3_1 + 0.5 * pn_pred3_3_2
                pn_pred = np.mean(pn_pred3_3, 0);
                pn_pred_ = np.argmax(pn_pred)
                pn_pred_one_hot = to_categorical(pn_pred_, 3)
                pn_prob_asb = pn_pred[0];
                pn_prob_typ = pn_pred[1];
                pn_prob_asp = pn_pred[2];
                pn_pred_asb = pn_pred_one_hot[0];
                pn_pred_typ = pn_pred_one_hot[1];
                pn_pred_asp = pn_pred_one_hot[2];

                # str_prob
                str_pred3_3_1 = softmax(logit_str3_1.detach().cpu().numpy());
                str_pred3_3_2 = softmax(logit_str3_2.detach().cpu().numpy());
                str_pred3_3 = 0.5 * str_pred3_3_1 + 0.5 * str_pred3_3_2
                str_pred = np.mean(str_pred3_3, 0);
                str_pred_ = np.argmax(str_pred)
                str_pred_one_hot = to_categorical(str_pred_, 3)
                str_prob_asb = str_pred[0];
                str_prob_reg = str_pred[1];
                str_prob_irg = str_pred[2];
                str_pred_asb = str_pred_one_hot[0];
                str_pred_reg = str_pred_one_hot[1];
                str_pred_irg = str_pred_one_hot[2];

                # pig_prob
                pig_pred3_3_1 = softmax(logit_pig3_1.detach().cpu().numpy());
                pig_pred3_3_2 = softmax(logit_pig3_2.detach().cpu().numpy());
                pig_pred3_3 = 0.5 * pig_pred3_3_1 + 0.5 * pig_pred3_3_2
                pig_pred = np.mean(pig_pred3_3, 0);
                pig_pred_ = np.argmax(pig_pred)
                pig_pred_one_hot = to_categorical(pig_pred_, 3)
                pig_prob_asb = pig_pred[0];
                pig_prob_reg = pig_pred[1];
                pig_prob_irg = pig_pred[2];
                pig_pred_asb = pig_pred_one_hot[0];
                pig_pred_reg = pig_pred_one_hot[1];
                pig_pred_irg = pig_pred_one_hot[2];

                # rs_prob
                rs_pred3_3_1 = softmax(logit_rs3_1.detach().cpu().numpy());
                rs_pred3_3_2 = softmax(logit_rs3_2.detach().cpu().numpy());
                rs_pred3_3 = 0.5 * rs_pred3_3_1 + 0.5 * rs_pred3_3_2
                rs_pred = np.mean(rs_pred3_3, 0);
                rs_pred_ = np.argmax(rs_pred)
                rs_pred_one_hot = to_categorical(rs_pred_, 2)
                rs_prob_asb = rs_pred[0];
                rs_prob_prs = rs_pred[1];
                rs_pred_asb = rs_pred_one_hot[0];
                rs_pred_prs = rs_pred_one_hot[1];

                # dag_prob
                dag_pred3_3_1 = softmax(logit_dag3_1.detach().cpu().numpy());
                dag_pred3_3_2 = softmax(logit_dag3_2.detach().cpu().numpy());
                dag_pred3_3 = 0.5 * dag_pred3_3_1 + 0.5 * dag_pred3_3_2
                dag_pred = np.mean(dag_pred3_3, 0);
                dag_pred_ = np.argmax(dag_pred)
                dag_pred_one_hot = to_categorical(dag_pred_, 3)
                dag_prob_asb = dag_pred[0];
                dag_prob_reg = dag_pred[1];
                dag_prob_irg = dag_pred[2];
                dag_pred_asb = dag_pred_one_hot[0];
                dag_pred_reg = dag_pred_one_hot[1];
                dag_pred_irg = dag_pred_one_hot[2];

                # bwv_prob
                bwv_pred3_3_1 = softmax(logit_bwv3_1.detach().cpu().numpy());
                bwv_pred3_3_2 = softmax(logit_bwv3_2.detach().cpu().numpy());
                bwv_pred3_3 = 0.5 * bwv_pred3_3_1 + 0.5 * bwv_pred3_3_2
                bwv_pred = np.mean(bwv_pred3_3, 0);
                bwv_pred_ = np.argmax(bwv_pred)
                bwv_pred_one_hot = to_categorical(bwv_pred_, 2)
                bwv_prob_asb = bwv_pred[0];
                bwv_prob_prs = bwv_pred[1];
                bwv_pred_asb = bwv_pred_one_hot[0];
                bwv_pred_prs = bwv_pred_one_hot[1];

                # vs_prob
                vs_pred3_3_1 = softmax(logit_vs3_1.detach().cpu().numpy());
                vs_pred3_3_2 = softmax(logit_vs3_2.detach().cpu().numpy());
                vs_pred3_3 = 0.5 * vs_pred3_3_1 + 0.5 * vs_pred3_3_2
                vs_pred = np.mean(vs_pred3_3, 0);
                vs_pred_ = np.argmax(vs_pred)
                vs_pred_one_hot = to_categorical(vs_pred_, 3)
                vs_prob_asb = vs_pred[0];
                vs_prob_reg = vs_pred[1];
                vs_prob_irg = vs_pred[2];
                vs_pred_asb = vs_pred_one_hot[0];
                vs_pred_reg = vs_pred_one_hot[1];
                vs_pred_irg = vs_pred_one_hot[2];

            if i == '1':
                if uncertainty3_1 <= uncertainty3_2:
                    pred3_3 = logit_diagnosis3_1
                    pred3_3 = softmax(pred3_3.detach().cpu().numpy());
                    pred = np.mean(pred3_3, 0);
                    pred_ = np.argmax(pred)
                    pred_one_hot = to_categorical(pred_,5)
                    nevu_prob = pred[0];
                    bcc_prob = pred[1];
                    mel_prob = pred[2];
                    misc_prob = pred[3];
                    sk_prob = pred[4];
                    nevu_pred = pred_one_hot[0];
                    bcc_pred = pred_one_hot[1];
                    mel_pred = pred_one_hot[2];
                    misc_pred = pred_one_hot[3];
                    sk_pred = pred_one_hot[4];

                    # pn_prob
                    pn_pred3_3 =  logit_pn3_1
                    pn_pred3_3 = softmax(pn_pred3_3.detach().cpu().numpy());
                    pn_pred = np.mean(pn_pred3_3, 0);
                    pn_pred_ = np.argmax(pn_pred)
                    pn_pred_one_hot = to_categorical(pn_pred_,3)
                    pn_prob_asb = pn_pred[0];
                    pn_prob_typ = pn_pred[1];
                    pn_prob_asp = pn_pred[2];
                    pn_pred_asb = pn_pred_one_hot[0];
                    pn_pred_typ = pn_pred_one_hot[1];
                    pn_pred_asp = pn_pred_one_hot[2];

                    # str_prob
                    str_pred3_3 = logit_str3_1
                    str_pred3_3 = softmax(str_pred3_3.detach().cpu().numpy());
                    str_pred = np.mean(str_pred3_3, 0);
                    str_pred_ = np.argmax(str_pred)
                    str_pred_one_hot = to_categorical(str_pred_,3)
                    str_prob_asb = str_pred[0];
                    str_prob_reg = str_pred[1];
                    str_prob_irg = str_pred[2];
                    str_pred_asb = str_pred_one_hot[0];
                    str_pred_reg = str_pred_one_hot[1];
                    str_pred_irg = str_pred_one_hot[2];

                    # pig_prob
                    pig_pred3_3 = logit_pig3_1
                    pig_pred3_3 = softmax(pig_pred3_3.detach().cpu().numpy())
                    pig_pred = np.mean(pig_pred3_3, 0);
                    pig_pred_ = np.argmax(pig_pred)
                    pig_pred_one_hot = to_categorical(pig_pred_,3)
                    pig_prob_asb = pig_pred[0];
                    pig_prob_reg = pig_pred[1];
                    pig_prob_irg = pig_pred[2];
                    pig_pred_asb = pig_pred_one_hot[0];
                    pig_pred_reg = pig_pred_one_hot[1];
                    pig_pred_irg = pig_pred_one_hot[2];

                    # rs_prob
                    rs_pred3_3 = logit_rs3_1
                    rs_pred3_3 = softmax(rs_pred3_3.detach().cpu().numpy())
                    rs_pred = np.mean(rs_pred3_3, 0);
                    rs_pred_ = np.argmax(rs_pred)
                    rs_pred_one_hot = to_categorical(rs_pred_,2)
                    rs_prob_asb = rs_pred[0];
                    rs_prob_prs = rs_pred[1];
                    rs_pred_asb = rs_pred_one_hot[0];
                    rs_pred_prs = rs_pred_one_hot[1];

                    # dag_prob
                    dag_pred3_3 = logit_dag3_1
                    dag_pred3_3 = softmax(dag_pred3_3.detach().cpu().numpy())
                    dag_pred = np.mean(dag_pred3_3, 0);
                    dag_pred_ = np.argmax(dag_pred)
                    dag_pred_one_hot = to_categorical(dag_pred_,3)
                    dag_prob_asb = dag_pred[0];
                    dag_prob_reg = dag_pred[1];
                    dag_prob_irg = dag_pred[2];
                    dag_pred_asb = dag_pred_one_hot[0];
                    dag_pred_reg = dag_pred_one_hot[1];
                    dag_pred_irg = dag_pred_one_hot[2];

                    # bwv_prob
                    bwv_pred3_3 = logit_bwv3_1
                    bwv_pred3_3 = softmax(bwv_pred3_3.detach().cpu().numpy())
                    bwv_pred = np.mean(bwv_pred3_3, 0);
                    bwv_pred_ = np.argmax(bwv_pred)
                    bwv_pred_one_hot = to_categorical(bwv_pred_,2)
                    bwv_prob_asb = bwv_pred[0];
                    bwv_prob_prs = bwv_pred[1];
                    bwv_pred_asb = bwv_pred_one_hot[0];
                    bwv_pred_prs = bwv_pred_one_hot[1];

                    # vs_prob
                    vs_pred3_3 = logit_vs3_1
                    vs_pred3_3 = softmax(vs_pred3_3.detach().cpu().numpy())
                    vs_pred = np.mean(vs_pred3_3, 0);
                    vs_pred_ = np.argmax(vs_pred)
                    vs_pred_one_hot = to_categorical(vs_pred_,3)
                    vs_prob_asb = vs_pred[0];
                    vs_prob_reg = vs_pred[1];
                    vs_prob_irg = vs_pred[2];
                    vs_pred_asb = vs_pred_one_hot[0];
                    vs_pred_reg = vs_pred_one_hot[1];
                    vs_pred_irg = vs_pred_one_hot[2];

                else:
                    pred3_3 = logit_diagnosis3_2
                    pred3_3 = softmax(pred3_3.detach().cpu().numpy());
                    pred = np.mean(pred3_3, 0);
                    pred_ = np.argmax(pred)
                    pred_one_hot = to_categorical(pred_,5)
                    nevu_prob = pred[0];
                    bcc_prob = pred[1];
                    mel_prob = pred[2];
                    misc_prob = pred[3];
                    sk_prob = pred[4];
                    nevu_pred = pred_one_hot[0];
                    bcc_pred = pred_one_hot[1];
                    mel_pred = pred_one_hot[2];
                    misc_pred = pred_one_hot[3];
                    sk_pred = pred_one_hot[4];

                    # pn_prob
                    pn_pred3_3 = logit_pn3_2
                    pn_pred3_3 = softmax(pn_pred3_3.detach().cpu().numpy());
                    pn_pred = np.mean(pn_pred3_3, 0);
                    pn_pred_ = np.argmax(pn_pred)
                    pn_pred_one_hot = to_categorical(pn_pred_,3)
                    pn_prob_asb = pn_pred[0];
                    pn_prob_typ = pn_pred[1];
                    pn_prob_asp = pn_pred[2];
                    pn_pred_asb = pn_pred_one_hot[0];
                    pn_pred_typ = pn_pred_one_hot[1];
                    pn_pred_asp = pn_pred_one_hot[2];

                    # str_prob
                    str_pred3_3 = logit_str3_2
                    str_pred3_3 = softmax(str_pred3_3.detach().cpu().numpy());
                    str_pred = np.mean(str_pred3_3, 0);
                    str_pred_ = np.argmax(str_pred)
                    str_pred_one_hot = to_categorical(str_pred_,3)
                    str_prob_asb = str_pred[0];
                    str_prob_reg = str_pred[1];
                    str_prob_irg = str_pred[2];
                    str_pred_asb = str_pred_one_hot[0];
                    str_pred_reg = str_pred_one_hot[1];
                    str_pred_irg = str_pred_one_hot[2];

                    # pig_prob
                    pig_pred3_3 = logit_pig3_2
                    pig_pred3_3 = softmax(pig_pred3_3.detach().cpu().numpy())
                    pig_pred = np.mean(pig_pred3_3, 0);
                    pig_pred_ = np.argmax(pig_pred)
                    pig_pred_one_hot = to_categorical(pig_pred_,3)
                    pig_prob_asb = pig_pred[0];
                    pig_prob_reg = pig_pred[1];
                    pig_prob_irg = pig_pred[2];
                    pig_pred_asb = pig_pred_one_hot[0];
                    pig_pred_reg = pig_pred_one_hot[1];
                    pig_pred_irg = pig_pred_one_hot[2];

                    # rs_prob
                    rs_pred3_3 = logit_rs3_2
                    rs_pred3_3 = softmax(rs_pred3_3.detach().cpu().numpy())
                    rs_pred = np.mean(rs_pred3_3, 0);
                    rs_pred_ = np.argmax(rs_pred)
                    rs_pred_one_hot = to_categorical(rs_pred_,2)
                    rs_prob_asb = rs_pred[0];
                    rs_prob_prs = rs_pred[1];
                    rs_pred_asb = rs_pred_one_hot[0];
                    rs_pred_prs = rs_pred_one_hot[1];

                    # dag_prob
                    dag_pred3_3 = logit_dag3_2
                    dag_pred3_3 = softmax(dag_pred3_3.detach().cpu().numpy())
                    dag_pred = np.mean(dag_pred3_3, 0);
                    dag_pred_ = np.argmax(dag_pred)
                    dag_pred_one_hot = to_categorical(dag_pred_,3)
                    dag_prob_asb = dag_pred[0];
                    dag_prob_reg = dag_pred[1];
                    dag_prob_irg = dag_pred[2];
                    dag_pred_asb = dag_pred_one_hot[0];
                    dag_pred_reg = dag_pred_one_hot[1];
                    dag_pred_irg = dag_pred_one_hot[2];

                    # bwv_prob
                    bwv_pred3_3 = logit_bwv3_2
                    bwv_pred3_3 = softmax(bwv_pred3_3.detach().cpu().numpy())
                    bwv_pred = np.mean(bwv_pred3_3, 0);
                    bwv_pred_ = np.argmax(bwv_pred)
                    bwv_pred_one_hot = to_categorical(bwv_pred_,2)
                    bwv_prob_asb = bwv_pred[0];
                    bwv_prob_prs = bwv_pred[1];
                    bwv_pred_asb = bwv_pred_one_hot[0];
                    bwv_pred_prs = bwv_pred_one_hot[1];

                    # vs_prob
                    vs_pred3_3 = logit_vs3_2
                    vs_pred3_3 = softmax(vs_pred3_3.detach().cpu().numpy())
                    vs_pred = np.mean(vs_pred3_3, 0);
                    vs_pred_ = np.argmax(vs_pred)
                    vs_pred_one_hot = to_categorical(vs_pred_,3)
                    vs_prob_asb = vs_pred[0];
                    vs_prob_reg = vs_pred[1];
                    vs_prob_irg = vs_pred[2];
                    vs_pred_asb = vs_pred_one_hot[0];
                    vs_pred_reg = vs_pred_one_hot[1];
                    vs_pred_irg = vs_pred_one_hot[2];

            if i == '2':
                pred3_3_1 = softmax(logit_diagnosis3_1.detach().cpu().numpy());
                pred3_3_2 = softmax(logit_diagnosis3_2.detach().cpu().numpy());
                pred3_3 = ((1 - uncertainty3_1)*pred3_3_1 + (1 - uncertainty3_2)*pred3_3_2)/(2 - uncertainty3_1 - uncertainty3_2)
                pred = np.mean(pred3_3, 0);
                pred_ = np.argmax(pred)
                pred_one_hot = to_categorical(pred_,5)
                nevu_prob = pred[0];
                bcc_prob = pred[1];
                mel_prob = pred[2];
                misc_prob = pred[3];
                sk_prob = pred[4];
                nevu_pred = pred_one_hot[0];
                bcc_pred = pred_one_hot[1];
                mel_pred = pred_one_hot[2];
                misc_pred = pred_one_hot[3];
                sk_pred = pred_one_hot[4];

                # pn_prob
                pn_pred3_3_1 = softmax(logit_pn3_1.detach().cpu().numpy());
                pn_pred3_3_2 = softmax(logit_pn3_2.detach().cpu().numpy());
                pn_pred3_3 = ((1 - uncertainty3_1)*pn_pred3_3_1 + (1 - uncertainty3_2)*pn_pred3_3_2)/(2 - uncertainty3_1 - uncertainty3_2)
                pn_pred = np.mean(pn_pred3_3, 0);
                pn_pred_ = np.argmax(pn_pred)
                pn_pred_one_hot = to_categorical(pn_pred_,3)
                pn_prob_asb = pn_pred[0];
                pn_prob_typ = pn_pred[1];
                pn_prob_asp = pn_pred[2];
                pn_pred_asb = pn_pred_one_hot[0];
                pn_pred_typ = pn_pred_one_hot[1];
                pn_pred_asp = pn_pred_one_hot[2];

                # str_prob
                str_pred3_3_1 = softmax(logit_str3_1.detach().cpu().numpy());
                str_pred3_3_2 = softmax(logit_str3_2.detach().cpu().numpy());
                str_pred3_3 = ((1 - uncertainty3_1)*str_pred3_3_1 + (1 - uncertainty3_2)*str_pred3_3_2)/(2 - uncertainty3_1 - uncertainty3_2)
                str_pred = np.mean(str_pred3_3, 0);
                str_pred_ = np.argmax(str_pred)
                str_pred_one_hot = to_categorical(str_pred_,3)
                str_prob_asb = str_pred[0];
                str_prob_reg = str_pred[1];
                str_prob_irg = str_pred[2];
                str_pred_asb = str_pred_one_hot[0];
                str_pred_reg = str_pred_one_hot[1];
                str_pred_irg = str_pred_one_hot[2];

                # pig_prob
                pig_pred3_3_1 = softmax(logit_pig3_1.detach().cpu().numpy());
                pig_pred3_3_2 = softmax(logit_pig3_2.detach().cpu().numpy());
                pig_pred3_3 = ((1 - uncertainty3_1)*pig_pred3_3_1 + (1 - uncertainty3_2)*pig_pred3_3_2)/(2 - uncertainty3_1 - uncertainty3_2)
                pig_pred = np.mean(pig_pred3_3, 0);
                pig_pred_ = np.argmax(pig_pred)
                pig_pred_one_hot = to_categorical(pig_pred_,3)
                pig_prob_asb = pig_pred[0];
                pig_prob_reg = pig_pred[1];
                pig_prob_irg = pig_pred[2];
                pig_pred_asb = pig_pred_one_hot[0];
                pig_pred_reg = pig_pred_one_hot[1];
                pig_pred_irg = pig_pred_one_hot[2];

                # rs_prob
                rs_pred3_3_1 = softmax(logit_rs3_1.detach().cpu().numpy());
                rs_pred3_3_2 = softmax(logit_rs3_2.detach().cpu().numpy());
                rs_pred3_3 = ((1 - uncertainty3_1)*rs_pred3_3_1 + (1 - uncertainty3_2)*rs_pred3_3_2)/(2 - uncertainty3_1 - uncertainty3_2)
                rs_pred = np.mean(rs_pred3_3, 0);
                rs_pred_ = np.argmax(rs_pred)
                rs_pred_one_hot = to_categorical(rs_pred_,2)
                rs_prob_asb = rs_pred[0];
                rs_prob_prs = rs_pred[1];
                rs_pred_asb = rs_pred_one_hot[0];
                rs_pred_prs = rs_pred_one_hot[1];

                # dag_prob
                dag_pred3_3_1 = softmax(logit_dag3_1.detach().cpu().numpy());
                dag_pred3_3_2 = softmax(logit_dag3_2.detach().cpu().numpy());
                dag_pred3_3 = ((1 - uncertainty3_1)*dag_pred3_3_1 + (1 - uncertainty3_2)*dag_pred3_3_2)/(2 - uncertainty3_1 - uncertainty3_2)
                dag_pred = np.mean(dag_pred3_3, 0);
                dag_pred_ = np.argmax(dag_pred)
                dag_pred_one_hot = to_categorical(dag_pred_,3)
                dag_prob_asb = dag_pred[0];
                dag_prob_reg = dag_pred[1];
                dag_prob_irg = dag_pred[2];
                dag_pred_asb = dag_pred_one_hot[0];
                dag_pred_reg = dag_pred_one_hot[1];
                dag_pred_irg = dag_pred_one_hot[2];

                # bwv_prob
                bwv_pred3_3_1 = softmax(logit_bwv3_1.detach().cpu().numpy());
                bwv_pred3_3_2 = softmax(logit_bwv3_2.detach().cpu().numpy());
                bwv_pred3_3 = ((1 - uncertainty3_1)*bwv_pred3_3_1 + (1 - uncertainty3_2)*bwv_pred3_3_2)/(2 - uncertainty3_1 - uncertainty3_2)
                bwv_pred = np.mean(bwv_pred3_3, 0);
                bwv_pred_ = np.argmax(bwv_pred)
                bwv_pred_one_hot = to_categorical(bwv_pred_,2)
                bwv_prob_asb = bwv_pred[0];
                bwv_prob_prs = bwv_pred[1];
                bwv_pred_asb = bwv_pred_one_hot[0];
                bwv_pred_prs = bwv_pred_one_hot[1];

                # vs_prob
                vs_pred3_3_1 = softmax(logit_vs3_1.detach().cpu().numpy());
                vs_pred3_3_2 = softmax(logit_vs3_2.detach().cpu().numpy());
                vs_pred3_3 = ((1 - uncertainty3_1)*vs_pred3_3_1 + (1 - uncertainty3_2)*vs_pred3_3_2)/(2 - uncertainty3_1 - uncertainty3_2)
                vs_pred = np.mean(vs_pred3_3, 0);
                vs_pred_ = np.argmax(vs_pred)
                vs_pred_one_hot = to_categorical(vs_pred_,3)
                vs_prob_asb = vs_pred[0];
                vs_prob_reg = vs_pred[1];
                vs_prob_irg = vs_pred[2];
                vs_pred_asb = vs_pred_one_hot[0];
                vs_pred_reg = vs_pred_one_hot[1];
                vs_pred_irg = vs_pred_one_hot[2];

            #encode label
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

            nevu_pred_list.append(nevu_pred);
            bcc_pred_list.append(bcc_pred);
            mel_pred_list.append(mel_pred);
            misc_pred_list.append(misc_pred);
            sk_pred_list.append(sk_pred);

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

            pn_pred_typ_list.append(pn_pred_typ);
            pn_pred_asp_list.append(pn_pred_asp);
            pn_pred_asb_list.append(pn_pred_asb);

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

            str_pred_reg_list.append(str_pred_reg);
            str_pred_irg_list.append(str_pred_irg);
            str_pred_asb_list.append(str_pred_asb);

            str_label_asb_list.append(streaks_label_one_hot[0]);
            str_label_reg_list.append(streaks_label_one_hot[1]);
            str_label_irg_list.append(streaks_label_one_hot[2])

            # pig_label
            pig_pred_list.append(pig_pred_);
            pig_prob_list.append(pig_pred);

            pig_label_list.append(pigmentation_label)
            pig_prob_reg_list.append(pig_prob_reg);
            pig_prob_irg_list.append(pig_prob_irg);
            pig_prob_asb_list.append(pig_prob_asb);

            pig_pred_reg_list.append(pig_pred_reg);
            pig_pred_irg_list.append(pig_pred_irg);
            pig_pred_asb_list.append(pig_pred_asb);

            pig_label_asb_list.append(pigmentation_label_one_hot[0]);
            pig_label_reg_list.append(pigmentation_label_one_hot[1]);
            pig_label_irg_list.append(pigmentation_label_one_hot[2])

            # rs_label
            rs_pred_list.append(rs_pred_);
            rs_prob_list.append(rs_pred);

            rs_label_list.append(regression_structures_label)
            rs_prob_asb_list.append(rs_prob_asb);
            rs_prob_prs_list.append(rs_prob_prs);

            rs_pred_asb_list.append(rs_pred_asb);
            rs_pred_prs_list.append(rs_pred_prs);

            rs_label_asb_list.append(regression_structures_label_one_hot[0]);
            rs_label_prs_list.append(regression_structures_label_one_hot[1])

            # dag_label
            dag_pred_list.append(dag_pred_);
            dag_prob_list.append(dag_pred);

            dag_label_list.append(dots_and_globules_label)
            dag_prob_reg_list.append(dag_prob_reg);
            dag_prob_irg_list.append(dag_prob_irg);
            dag_prob_asb_list.append(dag_prob_asb);

            dag_pred_reg_list.append(dag_pred_reg);
            dag_pred_irg_list.append(dag_pred_irg);
            dag_pred_asb_list.append(dag_pred_asb);

            dag_label_asb_list.append(dots_and_globules_label_one_hot[0]);
            dag_label_reg_list.append(dots_and_globules_label_one_hot[1]);
            dag_label_irg_list.append(dots_and_globules_label_one_hot[2])

            # bwv_label
            bwv_pred_list.append(bwv_pred_);
            bwv_prob_list.append(bwv_pred);

            bwv_label_list.append(blue_whitish_veil_label)
            bwv_prob_asb_list.append(bwv_prob_asb);
            bwv_prob_prs_list.append(bwv_prob_prs);

            bwv_pred_asb_list.append(bwv_pred_asb);
            bwv_pred_prs_list.append(bwv_pred_prs);

            bwv_label_asb_list.append(blue_whitish_veil_label_one_hot[0]);
            bwv_label_prs_list.append(blue_whitish_veil_label_one_hot[1])

            # vs_label
            vs_pred_list.append(vs_pred_);
            vs_prob_list.append(vs_pred);

            vs_label_list.append(vascular_structures_label)
            vs_prob_reg_list.append(vs_prob_reg);
            vs_prob_irg_list.append(vs_prob_irg);
            vs_prob_asb_list.append(vs_prob_asb);

            vs_pred_reg_list.append(vs_pred_reg);
            vs_pred_irg_list.append(vs_pred_irg);
            vs_pred_asb_list.append(vs_pred_asb);

            vs_label_asb_list.append(vascular_structures_label_one_hot[0]);
            vs_label_reg_list.append(vascular_structures_label_one_hot[1]);
            vs_label_irg_list.append(vascular_structures_label_one_hot[2])

        plt.plot(C,X,'Black')
        pred = np.array(pred_list).squeeze();
        prob = np.array(prob_list).squeeze();

        gt = np.array(gt_list)
        nevu_prob = np.array(nevu_prob_list);
        bcc_prob = np.array(bcc_prob_list);
        mel_prob = np.array(mel_prob_list);
        misc_prob = np.array(misc_prob_list);
        sk_prob = np.array(sk_prob_list);

        nevu_pred = np.array(nevu_pred_list);
        bcc_pred = np.array(bcc_pred_list);
        mel_pred = np.array(mel_pred_list);
        misc_pred = np.array(misc_pred_list);
        sk_pred = np.array(sk_pred_list);

        nevu_label = np.array(nevu_label_list);
        bcc_label = np.array(bcc_label_list);
        mel_label = np.array(mel_label_list);
        misc_label = np.array(misc_label_list);
        sk_label = np.array(sk_label_list)

        #pn
        pn_pred = np.array(pn_pred_list).squeeze();
        pn_prob = np.array(pn_prob_list).squeeze();

        pn_gt = np.array(pn_label_list)
        pn_prob_typ = np.array(pn_prob_typ_list);
        pn_prob_asp = np.array(pn_prob_asp_list);
        pn_prob_asb = np.array(pn_prob_asb_list)

        pn_pred_typ = np.array(pn_pred_typ_list);
        pn_pred_asp = np.array(pn_pred_asp_list);
        pn_pred_asb = np.array(pn_pred_asb_list);

        pn_label_typ = np.array(pn_label_typ_list);
        pn_label_asp = np.array(pn_label_asp_list);
        pn_label_asb = np.array(pn_label_asb_list)

        #str
        str_pred = np.array(str_pred_list).squeeze();
        str_prob = np.array(str_prob_list).squeeze();

        str_gt = np.array(str_label_list)
        str_prob_asb = np.array(str_prob_asb_list);
        str_prob_reg = np.array(str_prob_reg_list);
        str_prob_irg = np.array(str_prob_irg_list);

        str_pred_asb = np.array(str_pred_asb_list);
        str_pred_reg = np.array(str_pred_reg_list);
        str_pred_irg = np.array(str_pred_irg_list);

        str_label_asb = np.array(str_label_asb_list);
        str_label_reg = np.array(str_label_reg_list);
        str_label_irg = np.array(str_label_irg_list)

        #pig
        pig_pred = np.array(pig_pred_list).squeeze();
        pig_prob = np.array(pig_prob_list).squeeze();

        pig_gt = np.array(pig_label_list)
        pig_prob_asb = np.array(pig_prob_asb_list);
        pig_prob_reg = np.array(pig_prob_reg_list);
        pig_prob_irg = np.array(pig_prob_irg_list);

        pig_pred_asb = np.array(pig_pred_asb_list);
        pig_pred_reg = np.array(pig_pred_reg_list);
        pig_pred_irg = np.array(pig_pred_irg_list);

        pig_label_asb = np.array(pig_label_asb_list);
        pig_label_reg = np.array(pig_label_reg_list);
        pig_label_irg = np.array(pig_label_irg_list)

        #rs
        rs_pred = np.array(rs_pred_list).squeeze();
        rs_prob = np.array(rs_prob_list).squeeze();

        rs_gt = np.array(rs_label_list)
        rs_prob_asb = np.array(rs_prob_asb_list);
        rs_prob_prs = np.array(rs_prob_prs_list);

        rs_pred_asb = np.array(rs_pred_asb_list);
        rs_pred_prs = np.array(rs_pred_prs_list);

        rs_label_asb = np.array(rs_label_asb_list);
        rs_label_prs = np.array(rs_label_prs_list)

        #dag
        dag_pred = np.array(dag_pred_list).squeeze();
        dag_prob = np.array(dag_prob_list).squeeze();

        dag_gt = np.array(dag_label_list)
        dag_prob_asb = np.array(dag_prob_asb_list);
        dag_prob_reg = np.array(dag_prob_reg_list);
        dag_prob_irg = np.array(dag_prob_irg_list);

        dag_pred_asb = np.array(dag_pred_asb_list);
        dag_pred_reg = np.array(dag_pred_reg_list);
        dag_pred_irg = np.array(dag_pred_irg_list);

        dag_label_asb = np.array(dag_label_asb_list);
        dag_label_reg = np.array(dag_label_reg_list);
        dag_label_irg = np.array(dag_label_irg_list)

        #bwv
        bwv_pred = np.array(bwv_pred_list).squeeze();
        bwv_prob = np.array(bwv_prob_list).squeeze();

        bwv_gt = np.array(bwv_label_list)
        bwv_prob_asb = np.array(bwv_prob_asb_list);
        bwv_prob_prs = np.array(bwv_prob_prs_list);

        bwv_pred_asb = np.array(bwv_pred_asb_list);
        bwv_pred_prs = np.array(bwv_pred_prs_list);

        bwv_label_asb = np.array(bwv_label_asb_list);
        bwv_label_prs = np.array(bwv_label_prs_list)

        #vs
        vs_pred = np.array(vs_pred_list).squeeze();
        vs_prob = np.array(vs_prob_list).squeeze();

        vs_gt = np.array(vs_label_list)
        vs_prob_asb = np.array(vs_prob_asb_list);
        vs_prob_reg = np.array(vs_prob_reg_list);
        vs_prob_irg = np.array(vs_prob_irg_list);

        vs_pred_asb = np.array(vs_pred_asb_list);
        vs_pred_reg = np.array(vs_pred_reg_list);
        vs_pred_irg = np.array(vs_pred_irg_list);

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

        # calculate SEN, SPE, PRE
        cm_Pn_asb = sklearn.metrics.confusion_matrix((np.array(pn_label_asb) * 1).flatten(), pn_pred_asb.flatten())
        Tn_pn_asb, Fp_pn_asb, Fn_pn_asb, Tp_pn_asb = cm_Pn_asb.ravel()
        cm_Pn_typ = sklearn.metrics.confusion_matrix((np.array(pn_label_typ) * 1).flatten(), pn_pred_typ.flatten())
        Tn_pn_typ, Fp_pn_typ, Fn_pn_typ, Tp_pn_typ = cm_Pn_typ.ravel()
        cm_Pn_aty = sklearn.metrics.confusion_matrix((np.array(pn_label_asp) * 1).flatten(), pn_pred_asp.flatten())
        Tn_pn_aty, Fp_pn_aty, Fn_pn_aty, Tp_pn_aty = cm_Pn_aty.ravel()


        cm_Str_asb = sklearn.metrics.confusion_matrix((np.array(str_label_asb) * 1).flatten(), str_pred_asb.flatten())
        Tn_str_asb, Fp_str_asb, Fn_str_asb, Tp_str_asb = cm_Str_asb.ravel()
        cm_Str_reg = sklearn.metrics.confusion_matrix((np.array(str_label_reg) * 1).flatten(), str_pred_reg.flatten())
        Tn_str_reg, Fp_str_reg, Fn_str_reg, Tp_str_reg = cm_Str_reg.ravel()
        cm_Str_irg = sklearn.metrics.confusion_matrix((np.array(str_label_irg) * 1).flatten(), str_pred_irg.flatten())
        Tn_str_irg, Fp_str_irg, Fn_str_irg, Tp_str_irg = cm_Str_irg.ravel()


        cm_Pig_asb = sklearn.metrics.confusion_matrix((np.array(pig_label_asb) * 1).flatten(), pig_pred_asb.flatten())
        Tn_pig_asb, Fp_pig_asb, Fn_pig_asb, Tp_pig_asb = cm_Pig_asb.ravel()
        cm_Pig_reg = sklearn.metrics.confusion_matrix((np.array(pig_label_reg) * 1).flatten(), pig_pred_reg.flatten())
        Tn_pig_reg, Fp_pig_reg, Fn_pig_reg, Tp_pig_reg = cm_Pig_reg.ravel()
        cm_Pig_irg = sklearn.metrics.confusion_matrix((np.array(pig_label_irg) * 1).flatten(), pig_pred_irg.flatten())
        Tn_pig_irg, Fp_pig_irg, Fn_pig_irg, Tp_pig_irg = cm_Pig_irg.ravel()


        cm_Rs_asb = sklearn.metrics.confusion_matrix((np.array(rs_label_asb) * 1).flatten(), rs_pred_asb.flatten())
        Tn_rs_asb, Fp_rs_asb, Fn_rs_asb, Tp_rs_asb = cm_Rs_asb.ravel()
        cm_Rs_prs = sklearn.metrics.confusion_matrix((np.array(rs_label_prs) * 1).flatten(), rs_pred_prs.flatten())
        Tn_rs_prs, Fp_rs_prs, Fn_rs_prs, Tp_rs_prs = cm_Rs_prs.ravel()


        cm_Vs_asb = sklearn.metrics.confusion_matrix((np.array(vs_label_asb) * 1).flatten(), vs_pred_asb.flatten())
        Tn_vs_asb, Fp_vs_asb, Fn_vs_asb, Tp_vs_asb = cm_Vs_asb.ravel()
        cm_Vs_reg = sklearn.metrics.confusion_matrix((np.array(vs_label_reg) * 1).flatten(), vs_pred_reg.flatten())
        Tn_vs_reg, Fp_vs_reg, Fn_vs_reg, Tp_vs_reg = cm_Vs_reg.ravel()
        cm_Vs_irg = sklearn.metrics.confusion_matrix((np.array(vs_label_irg) * 1).flatten(), vs_pred_irg.flatten())
        Tn_vs_irg, Fp_vs_irg, Fn_vs_irg, Tp_vs_irg = cm_Vs_irg.ravel()


        cm_Bwv_asb = sklearn.metrics.confusion_matrix((np.array(bwv_label_asb) * 1).flatten(), bwv_pred_asb.flatten())
        Tn_bwv_asb, Fp_bwv_asb, Fn_bwv_asb, Tp_bwv_asb = cm_Bwv_asb.ravel()
        cm_Bwv_prs = sklearn.metrics.confusion_matrix((np.array(bwv_label_prs) * 1).flatten(), bwv_pred_prs.flatten())
        Tn_bwv_prs, Fp_bwv_prs, Fn_bwv_prs, Tp_bwv_prs = cm_Bwv_prs.ravel()


        cm_Dag_asb = sklearn.metrics.confusion_matrix((np.array(dag_label_asb) * 1).flatten(), dag_pred_asb.flatten())
        Tn_dag_asb, Fp_dag_asb, Fn_dag_asb, Tp_dag_asb = cm_Dag_asb.ravel()
        cm_Dag_reg = sklearn.metrics.confusion_matrix((np.array(dag_label_reg) * 1).flatten(), dag_pred_reg.flatten())
        Tn_dag_reg, Fp_dag_reg, Fn_dag_reg, Tp_dag_reg = cm_Dag_reg.ravel()
        cm_Dag_irg = sklearn.metrics.confusion_matrix((np.array(dag_label_irg) * 1).flatten(), dag_pred_irg.flatten())
        Tn_dag_irg, Fp_dag_irg, Fn_dag_irg, Tp_dag_irg = cm_Dag_irg.ravel()


        cm_Diag_nev = sklearn.metrics.confusion_matrix((np.array(nevu_label) * 1).flatten(), nevu_pred.flatten())
        Tn_diag_nev, Fp_diag_nev, Fn_diag_nev, Tp_diag_nev = cm_Diag_nev.ravel()
        cm_Diag_bcc = sklearn.metrics.confusion_matrix((np.array(bcc_label) * 1).flatten(), bcc_pred.flatten())
        Tn_diag_bcc, Fp_diag_bcc, Fn_diag_bcc, Tp_diag_bcc = cm_Diag_bcc.ravel()
        cm_Diag_mel = sklearn.metrics.confusion_matrix((np.array(mel_label) * 1).flatten(), mel_pred.flatten())
        Tn_diag_mel, Fp_diag_mel, Fn_diag_mel, Tp_diag_mel = cm_Diag_mel.ravel()
        cm_Diag_misc = sklearn.metrics.confusion_matrix((np.array(misc_label) * 1).flatten(), misc_pred.flatten())
        Tn_diag_misc, Fp_diag_misc, Fn_diag_misc, Tp_diag_misc = cm_Diag_misc.ravel()
        cm_Diag_sk = sklearn.metrics.confusion_matrix((np.array(sk_label) * 1).flatten(), sk_pred.flatten())
        Tn_diag_sk, Fp_diag_sk, Fn_diag_sk, Tp_diag_sk = cm_Diag_sk.ravel()


        # SEN
        SEN_pn_asb = Tp_pn_asb / (Tp_pn_asb + Fn_pn_asb)
        SEN_pn_typ = Tp_pn_typ / (Tp_pn_typ + Fn_pn_typ)
        SEN_pn_aty = Tp_pn_aty / (Tp_pn_aty + Fn_pn_aty)


        SEN_str_asb = Tp_str_asb / (Tp_str_asb + Fn_str_asb)
        SEN_str_reg = Tp_str_reg / (Tp_str_reg + Fn_str_reg)
        SEN_str_irg = Tp_str_irg / (Tp_str_irg + Fn_str_irg)


        SEN_pig_asb = Tp_pig_asb / (Tp_pig_asb + Fn_pig_asb)
        SEN_pig_reg = Tp_pig_reg / (Tp_pig_reg + Fn_pig_reg)
        SEN_pig_irg = Tp_pig_irg / (Tp_pig_irg + Fn_pig_irg)


        SEN_rs_asb = Tp_rs_asb / (Tp_rs_asb + Fn_rs_asb)
        SEN_rs_prs = Tp_rs_prs / (Tp_rs_prs + Fn_rs_prs)


        SEN_vs_asb = Tp_vs_asb / (Tp_vs_asb + Fn_vs_asb)
        SEN_vs_reg = Tp_vs_reg / (Tp_vs_reg + Fn_vs_reg)
        SEN_vs_irg = Tp_vs_irg / (Tp_vs_irg + Fn_vs_irg)


        SEN_bwv_asb = Tp_bwv_asb / (Tp_bwv_asb + Fn_bwv_asb)
        SEN_bwv_prs = Tp_bwv_prs / (Tp_bwv_prs + Fn_bwv_prs)


        SEN_dag_asb = Tp_dag_asb / (Tp_dag_asb + Fn_dag_asb)
        SEN_dag_reg = Tp_dag_reg / (Tp_dag_reg + Fn_dag_reg)
        SEN_dag_irg = Tp_dag_irg / (Tp_dag_irg + Fn_dag_irg)


        SEN_diag_nev = Tp_diag_nev / (Tp_diag_nev + Fn_diag_nev)
        SEN_diag_bcc = Tp_diag_bcc / (Tp_diag_bcc + Fn_diag_bcc)
        SEN_diag_mel = Tp_diag_mel / (Tp_diag_mel + Fn_diag_mel)
        SEN_diag_misc = Tp_diag_misc / (Tp_diag_misc + Fn_diag_misc)
        SEN_diag_sk = Tp_diag_sk / (Tp_diag_sk + Fn_diag_sk)

        # SPE
        SPE_pn_asb = Tn_pn_asb / (Tn_pn_asb + Fp_pn_asb)
        SPE_pn_typ = Tn_pn_typ / (Tn_pn_typ + Fp_pn_typ)
        SPE_pn_aty = Tn_pn_aty / (Tn_pn_aty + Fp_pn_aty)


        SPE_str_asb = Tn_str_asb / (Tn_str_asb + Fp_str_asb)
        SPE_str_reg = Tn_str_reg / (Tn_str_reg + Fp_str_reg)
        SPE_str_irg = Tn_str_irg / (Tn_str_irg + Fp_str_irg)


        SPE_pig_asb = Tn_pig_asb / (Tn_pig_asb + Fp_pig_asb)
        SPE_pig_reg = Tn_pig_reg / (Tn_pig_reg + Fp_pig_reg)
        SPE_pig_irg = Tn_pig_irg / (Tn_pig_irg + Fp_pig_irg)


        SPE_rs_asb = Tn_rs_asb / (Tn_rs_asb + Fp_rs_asb)
        SPE_rs_prs = Tn_rs_prs / (Tn_rs_prs + Fp_rs_prs)


        SPE_vs_asb = Tn_vs_asb / (Tn_vs_asb + Fp_vs_asb)
        SPE_vs_reg = Tn_vs_reg / (Tn_vs_reg + Fp_vs_reg)
        SPE_vs_irg = Tn_vs_irg / (Tn_vs_irg + Fp_vs_irg)


        SPE_bwv_asb = Tn_bwv_asb / (Tn_bwv_asb + Fp_bwv_asb)
        SPE_bwv_prs = Tn_bwv_prs / (Tn_bwv_prs + Fp_bwv_prs)


        SPE_dag_asb = Tn_dag_asb / (Tn_dag_asb + Fp_dag_asb)
        SPE_dag_reg = Tn_dag_reg / (Tn_dag_reg + Fp_dag_reg)
        SPE_dag_irg = Tn_dag_irg / (Tn_dag_irg + Fp_dag_irg)


        SPE_diag_nev = Tn_diag_nev / (Tn_diag_nev + Fp_diag_nev)
        SPE_diag_bcc = Tn_diag_bcc / (Tn_diag_bcc + Fp_diag_bcc)
        SPE_diag_mel = Tn_diag_mel / (Tn_diag_mel + Fp_diag_mel)
        SPE_diag_misc = Tn_diag_misc / (Tn_diag_misc + Fp_diag_misc)
        SPE_diag_sk = Tn_diag_sk / (Tn_diag_sk + Fp_diag_sk)

        # PRE
        PRE_pn_asb = Tp_pn_asb / (Tp_pn_asb + Fp_pn_asb)
        PRE_pn_typ = Tp_pn_typ / (Tp_pn_typ + Fp_pn_typ)
        PRE_pn_aty = Tp_pn_aty / (Tp_pn_aty + Fp_pn_aty)


        PRE_str_asb = Tp_str_asb / (Tp_str_asb + Fp_str_asb)
        PRE_str_reg = Tp_str_reg / (Tp_str_reg + Fp_str_reg)
        PRE_str_irg = Tp_str_irg / (Tp_str_irg + Fp_str_irg)


        PRE_pig_asb = Tp_pig_asb / (Tp_pig_asb + Fp_pig_asb)
        PRE_pig_reg = Tp_pig_reg / (Tp_pig_reg + Fp_pig_reg)
        PRE_pig_irg = Tp_pig_irg / (Tp_pig_irg + Fp_pig_irg)


        PRE_rs_asb = Tp_rs_asb / (Tp_rs_asb + Fp_rs_asb)
        PRE_rs_prs = Tp_rs_prs / (Tp_rs_prs + Fp_rs_prs)


        PRE_vs_asb = Tp_vs_asb / (Tp_vs_asb + Fp_vs_asb)
        PRE_vs_reg = Tp_vs_reg / (Tp_vs_reg + Fp_vs_reg)
        PRE_vs_irg = Tp_vs_irg / (Tp_vs_irg + Fp_vs_irg)


        PRE_bwv_asb = Tp_bwv_asb / (Tp_bwv_asb + Fp_bwv_asb)
        PRE_bwv_prs = Tp_bwv_prs / (Tp_bwv_prs + Fp_bwv_prs)


        PRE_dag_asb = Tp_dag_asb / (Tp_dag_asb + Fp_dag_asb)
        PRE_dag_reg = Tp_dag_reg / (Tp_dag_reg + Fp_dag_reg)
        PRE_dag_irg = Tp_dag_irg / (Tp_dag_irg + Fp_dag_irg)


        PRE_diag_nev = Tp_diag_nev / (Tp_diag_nev + Fp_diag_nev)
        PRE_diag_bcc = Tp_diag_bcc / (Tp_diag_bcc + Fp_diag_bcc)
        PRE_diag_mel = Tp_diag_mel / (Tp_diag_mel + Fp_diag_mel)
        PRE_diag_misc = Tp_diag_misc / (Tp_diag_misc + Fp_diag_misc)
        PRE_diag_sk = Tp_diag_sk / (Tp_diag_sk + Fp_diag_sk)
        log.write('Diag' + '-' * 25 + "\n")
        log.write('nevu_SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_nev, SPE_diag_nev, PRE_diag_nev))
        log.write('bcc SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_bcc, SPE_diag_bcc, PRE_diag_bcc))
        log.write('mel SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_mel, SPE_diag_mel, PRE_diag_mel))
        log.write('misc SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_misc, SPE_diag_misc, PRE_diag_misc))
        log.write('sk SEN: {}, SPE: {}, PRE: {}'.format(SEN_diag_sk, SPE_diag_sk, PRE_diag_sk))
        log.write('-' * 10 + '\n')
        log.write('pn_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_pn_asb, SPE_pn_asb, PRE_pn_asb))
        log.write('pn_typ SEN: {}, SPE: {}, PRE: {}'.format(SEN_pn_typ, SPE_pn_typ, PRE_pn_typ))
        log.write('pn_aty SEN: {}, SPE: {}, PRE: {}'.format(SEN_pn_aty, SPE_pn_aty, PRE_pn_aty))
        log.write('-' * 10 + '\n')
        log.write('str_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_str_asb, SPE_str_asb, PRE_str_asb))
        log.write('str_reg SEN: {}, SPE: {}, PRE: {}'.format(SEN_str_reg, SPE_str_reg, PRE_str_reg))
        log.write('str_irg SEN: {}, SPE: {}, PRE: {}'.format(SEN_str_irg, SPE_str_irg, PRE_str_irg))
        log.write('-' * 10 + '\n')
        log.write('pig_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_pig_asb, SPE_pig_asb, PRE_pig_asb))
        log.write('pig_reg SEN: {}, SPE: {}, PRE: {}'.format(SEN_pig_reg, SPE_pig_reg, PRE_pig_reg))
        log.write('pig_irg SEN: {}, SPE: {}, PRE: {}'.format(SEN_pig_irg, SPE_pig_irg, PRE_pig_irg))
        log.write('-' * 10 + '\n')
        log.write('rs_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_rs_asb, SPE_rs_asb, PRE_rs_asb))
        log.write('rs_prs SEN: {}, SPE: {}, PRE: {}'.format(SEN_rs_prs, SPE_rs_prs, PRE_rs_prs))
        log.write('-' * 10 + '\n')
        log.write('vs_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_vs_asb, SPE_vs_asb, PRE_vs_asb))
        log.write('vs_reg SEN: {}, SPE: {}， PRE: {}'.format(SEN_vs_reg, SPE_vs_reg, PRE_vs_reg))
        log.write('vs_irg SEN: {}, SPE: {}, PRE: {}'.format(SEN_vs_irg, SPE_vs_irg, PRE_vs_irg))
        log.write('-' * 10 + '\n')
        log.write('bwv_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_bwv_asb, SPE_bwv_asb, PRE_bwv_asb))
        log.write('bwv_prs SEN: {}, SPE: {}, PRE: {}'.format(SEN_bwv_prs, SPE_bwv_prs, PRE_bwv_prs))
        log.write('-' * 10 + '\n')
        log.write('dag_asb SEN: {}, SPE: {}, PRE: {}'.format(SEN_dag_asb, SPE_dag_asb, PRE_dag_asb))
        log.write('dag_reg SEN: {}, SPE: {}, PRE: {}'.format(SEN_dag_reg, SPE_dag_reg, PRE_dag_reg))
        log.write('dag_irg SEN: {}, SPE: {}, PRE: {}'.format(SEN_dag_irg, SPE_dag_irg, PRE_dag_irg))

        #AUC
        nevu_auc = roc_auc_score((np.array(nevu_label) * 1).flatten(), nevu_prob.flatten())
        bcc_auc = roc_auc_score((np.array(bcc_label) * 1).flatten(), bcc_prob.flatten())
        mel_auc = roc_auc_score((np.array(mel_label) * 1).flatten(), mel_prob.flatten())
        misc_auc = roc_auc_score((np.array(misc_label) * 1).flatten(), misc_prob.flatten())
        sk_auc = roc_auc_score((np.array(sk_label) * 1).flatten(), sk_prob.flatten())
        log.write('-' * 25 + '\n')
        log.write('diag_Bcc: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_bcc, SPE_diag_bcc, PRE_diag_bcc, bcc_auc))
        log.write('diag_Mel: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_mel, SPE_diag_mel, PRE_diag_mel, mel_auc))
        log.write('diag_Nev: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_nev, SPE_diag_nev, PRE_diag_nev, nevu_auc))
        log.write('diag_Misc: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_misc, SPE_diag_misc, PRE_diag_misc, misc_auc))
        log.write('diag_Sk: SEN: {}, SPE: {}, PRE: {}, AUC: {}'.format(SEN_diag_sk, SPE_diag_sk, PRE_diag_sk, sk_auc))
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
        #log.close()

    return avg_acc, [prob, pn_prob, str_prob, pig_prob, rs_prob, dag_prob, bwv_prob, vs_prob], [
        np.array(nevu_label), np.array(bcc_label), np.array(mel_label), np.array(misc_label), np.array(sk_label)], [
               nevu_prob, bcc_prob, mel_prob, misc_prob, sk_prob], seven_point_feature_list, [gt, pn_gt,
                                                                                              str_gt, pig_gt,
                                                                                              rs_gt, dag_gt,
                                                                                              bwv_gt, vs_gt]
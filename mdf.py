import tqdm
import torch
import pickle

import numpy as np
import pandas as pd

import sklearn
import sklearn.covariance
from sklearn import svm

from metrics import *
from outputs import *

def get_empirical_mean_cov(cov_estimator, layer_embeddings, num_layers=13):
    mean_list = []
    precision_list = []

    for l in tqdm.tqdm(range(num_layers)):
        sample_mean = torch.mean(layer_embeddings[:,l,:], axis=0)
        X = layer_embeddings[:,l,:] - sample_mean
        cov_estimator.fit(X.numpy())
        temp_precision = cov_estimator.precision_
        temp_precision = torch.from_numpy(temp_precision).float()
        mean_list.append(sample_mean)#.to(device))
        precision_list.append(temp_precision)#.to(device))
    return mean_list, precision_list


def get_mdf(test_layer_embeddings, layer_mean, layer_precision, num_layers=13):
    mdf_list = []
    
    for l in tqdm.tqdm(range(num_layers)):
        zero_f = test_layer_embeddings[:,l,:] - layer_mean[l]
        gaussian_score = -0.5 * ((zero_f @ layer_precision[l]) @ zero_f.t()).diag()
        mdf_list.append(gaussian_score)
    
    return torch.stack(mdf_list).transpose(1,0)


def oc_svm_train(trn_mdf, test_mdf, ood_mdf, label):
    
    candidate_list = [1e-15, 1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 0.2, 0.5, 1]

    ood_labels = np.ones(shape=(ood_mdf.shape[0], ))
    test_labels = np.zeros(shape=(test_mdf.shape[0], ))

    print(test_mdf.shape)
    print(ood_mdf.shape)
    
    np.random.shuffle(test_mdf)
    np.random.shuffle(ood_mdf)
    best_ours_results = None
    best_ours_AUROC = 0.0
    best_model = None
    # for k in ['poly', 'linear']:
    for k in ['linear']:
        for nuu in tqdm.tqdm(candidate_list):
            #print ("running ---:", "kernel:", k, "nuu:", nuu)
            
            c_lr = svm.OneClassSVM(nu=nuu, kernel=k, degree=2, tol=1e-5)
            #c_lr = sklearn.linear_model.SGDOneClassSVM(nu=nuu, random_state=42)
            c_lr.fit(trn_mdf)
            
            test_scores = c_lr.score_samples(test_mdf)
            ood_scores = c_lr.score_samples(ood_mdf)
            X_scores = np.concatenate((ood_scores, test_scores))
            Y_test = np.concatenate((ood_labels, test_labels))
            
            results = detection_performance(X_scores, Y_test, 'mah_logs', nuu, tag='TMP')
            neg_resuls = detection_performance(-X_scores, Y_test, 'feats_logs', nuu, tag='TMP')
            if sum(results["TMP"].values()) < sum(neg_resuls["TMP"].values()):
                results = neg_resuls
                
            #print(results)

            if results['TMP']['AUROC'] > best_ours_AUROC:
                best_ours_AUROC = results['TMP']['AUROC']
                best_ours_results = results
                best_hypers = '{}-{}'.format(k, nuu)
                # save data for plotting
                best_model = c_lr
                d = {"X_scores": X_scores, "Y_test": Y_test, "Features": np.concatenate((test_scores, ood_scores))}
    mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*best_ours_results['TMP']['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*best_ours_results['TMP']['DTACC']), end='')
    print(' {val:6.2f}'.format(val=100.*best_ours_results['TMP']['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*best_ours_results['TMP']['AUOUT']), end='')
    print("best hyper %s"%(best_hypers)) 
    print ("saving data for plotting")
    with open("../outputs/{}_{}.pkl".format('NTDB', label), "wb") as f:
        pickle.dump(d, f)
    print('-------------------------------')
    
    return d, best_ours_results, best_model


def oc_svm_detector(ecode, ecode_trn_seq, ecode_gen_seq, trn_frac, percent_print, pcodes=False):
    
    TRN_FRAC = trn_frac
    PRC_PRNT = percent_print
    
    pcodes_path = ""
    if pcodes:
        pcodes_path = "pcode_"
    ecode_trn_all_token_layer_embeddings = np.load(f"../outputs/{ecode}_{pcodes_path}trn_all_em.npy", allow_pickle=True)
    ecode_gen_all_token_layer_embeddings = np.load(f"../outputs/{ecode}_{pcodes_path}gen_all_em.npy", allow_pickle=True)
    
        
    ecode_trn_end_token_layer_embeddings = torch.stack([x[-1,:,:] for x in ecode_trn_all_token_layer_embeddings])
    ecode_gen_end_token_layer_embeddings = torch.stack([x[-1,:,:] for x in ecode_gen_all_token_layer_embeddings])
    
    ecode_trn_end_token_layer_embeddings = ecode_trn_end_token_layer_embeddings[:int(len(ecode_trn_end_token_layer_embeddings)*TRN_FRAC),:,:]
    ecode_tst_end_token_layer_embeddings = ecode_trn_end_token_layer_embeddings[int(len(ecode_trn_end_token_layer_embeddings)*TRN_FRAC):,:,:]
    
    # Make OOD same size as TST and keep same order
    ecode_ood_end_token_layer_embeddings = ecode_gen_end_token_layer_embeddings[:len(ecode_tst_end_token_layer_embeddings),:,:]

    cov_estimator = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    
    ecode_trn_end_token_mean, ecode_trn_end_token_inv_cov = get_empirical_mean_cov(cov_estimator, ecode_trn_end_token_layer_embeddings)
    ecode_trn_mdf = get_mdf(torch.Tensor(ecode_trn_end_token_layer_embeddings), ecode_trn_end_token_mean, ecode_trn_end_token_inv_cov)
    ecode_tst_mdf = get_mdf(torch.Tensor(ecode_tst_end_token_layer_embeddings), ecode_trn_end_token_mean, ecode_trn_end_token_inv_cov)
    ecode_ood_mdf = get_mdf(torch.Tensor(ecode_ood_end_token_layer_embeddings), ecode_trn_end_token_mean, ecode_trn_end_token_inv_cov)
    
    # Train the OC SVM
    ecode_score, ecode_results, ecode_det = oc_svm_train(ecode_trn_mdf, ecode_tst_mdf, ecode_ood_mdf, ecode)
    
    ecode_score_df = pd.DataFrame(ecode_score)
    
    # Get the range and delineator (75% quantile)
    rng, d = plot_scores(ecode_score_df, ecode_det)
    
    ecode_score_samples = ecode_det.score_samples(ecode_ood_mdf)
    
    y = ecode_det.score_samples(ecode_ood_mdf)

    ecode_normal = []
    for i,o in enumerate(y):
        if o < d:
            ecode_normal.append((i,o))

    ecode_anomaly = []
    for i,o in enumerate(y):
        if o >= d:
            ecode_anomaly.append((i,o))

    print('Normal:', len(ecode_normal), 'Anomaly:', len(ecode_anomaly))

    #print('=' * 80)
    #print('In Distribution')
    #print('=' * 80)
    rows = []
    for i in range(PRC_PRNT):
        if '<UNK>' not in ecode_gen_seq[ecode_normal[i][0]]:
            row = {}
            row['label'] = '1'
            row['score'] = ecode_normal[i][1]
            row['output'] = string_seq_dsc(ecode_gen_seq[ecode_normal[i][0]], pcodes)
            
            rows.append(row)
            #print('OC SVM Score:', format(ecode_normal[i][1], '.1e'))
            #print_seq_dsc(ecode_gen_seq[ecode_normal[i][0]])
            #print()

    #print('=' * 80)
    #print('Out of Distribution')
    #print('=' * 80)
    
    for i in range(PRC_PRNT):
        if '<UNK>' not in ecode_gen_seq[ecode_anomaly[i][0]]:
            row = {}
            row['label'] = '-1'
            row['score'] = ecode_anomaly[i][1]
            row['output'] = string_seq_dsc(ecode_gen_seq[ecode_anomaly[i][0]], pcodes)
            
            rows.append(row)
            
            #print('OC SVM Score:', format(ecode_anomaly[i][1], '.1e'))
            #print_seq_dsc(ecode_gen_seq[ecode_anomaly[i][0]])
            #print()
    
    return pd.DataFrame(rows), d
    
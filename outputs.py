import re
import pickle
import matplotlib.pyplot as plt

from icd9cms.icd9 import search

with open("../data/pcode_dict.txt", "rb") as fp: 
    icd9_pcode_dict = pickle.load(fp)

def plot_scores(score_df, model, bins=100, ft='X_scores'):
    
    #rng = [score_df[ft].min(), score_df[ft].mean() + 2*score_df[ft].std()]
    rng = [score_df[ft].quantile(0.05), score_df[ft].quantile(0.95)]
    
    y = score_df[score_df['Y_test']==1][ft]
    d = y.quantile(0.75)
    #d = model.offset_[0]
    
    plt.hist(score_df[score_df['Y_test']==1][ft], bins=bins, range=rng, alpha=0.7, label='Generated')
    plt.hist(score_df[score_df['Y_test']==0][ft], bins=bins, range=rng, alpha=0.7, label='NTDB')
    #plt.axvline(x=d, color='k', linestyle='--')
    plt.axvline(x=d, color='k', linestyle='--')
    plt.legend(loc='upper right')
    plt.show()
    
    return rng, d


def print_seq_dsc(seq, pcodes=False):
    cds = seq.split()
    tp = 'START'
    for c in cds:
        if c == '<START>':
            print('=' * 9 + ' START ' + '=' * 9)
        elif c == '<DSTART>':
            tp = 'DX'
            print('=' * 10 + ' DXS ' + '=' * 10)
        elif c == '<PSTART>':
            if pcodes:
                return
            tp = 'PR'
            print('=' * 10 + ' PRS ' + '=' * 10)
        elif c == '<END>':
            print('=' * 10 + ' END ' + '=' * 10)
            return
        elif c == '<UNK>':
            print(f'{c}:Unknown Code')
        else:
            if tp == 'DX':
                d = search(c)
                if d:
                    print(d)
            elif tp == 'PR':
                pr_cd = re.sub(r'\.', '', c)
                if pr_cd in icd9_pcode_dict:
                    print(f"{pr_cd}:{icd9_pcode_dict[pr_cd]}")
                else:
                    print(f'{pr_cd}:Unknown Code')
            else:
                continue
                

def string_seq_dsc(seq, pcodes=False):
    cds = seq.split()
    tp = 'START'
    
    s = ''
    for c in cds:
        if c == '<START>':
            s += '=' * 9 + ' START ' + '=' * 9 + '\n'
        elif c == '<DSTART>':
            tp = 'DX'
            s += '=' * 10 + ' DXS ' + '=' * 10 + '\n'
        elif c == '<PSTART>':
            if pcodes:
                return s
            tp = 'PR'
            s += '=' * 10 + ' PRS ' + '=' * 10 + '\n'
        elif c == '<END>':
            s += '=' * 10 + ' END ' + '=' * 10 + '\n'
            return s
        elif c == '<UNK>':
            s += f'{c}:Unknown Code' + '\n'
        else:
            if tp == 'DX':
                d = search(c)
                if d:
                    s += str(d) + '\n'
            elif tp == 'PR':
                pr_cd = re.sub(r'\.', '', c)
                if pr_cd in icd9_pcode_dict:
                    s += f"{pr_cd}:{icd9_pcode_dict[pr_cd]}" + '\n'
                else:
                    s += f'{pr_cd}:Unknown Code' + '\n'
            elif tp == 'START':
                if c == 'E812.0':
                    s += "E812.0:Other motor vehicle traffic accident involving collision with motor vehicle injuring driver of motor vehicle other than motorcycle\n"
                if c == 'E885.9':
                    s += "E885.9:Accidental fall from other slipping tripping or stumbling\n"
                if c == 'E966.0':
                    s += "E966.0:Assault by cutting and piercing instrument\n"
                if c == 'E965.4':
                    s += "E965.4:Assault by firearms and explosives, Other and unspecified firearm\n"
                if c == 'E924.0':
                    s += "E924.0:Accident caused by hot substance or object, caustic or corrosive material, and steam, Hot liquids and vapors, including steam\n"
            else:
                continue
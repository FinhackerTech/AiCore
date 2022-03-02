
import numpy as np
import pandas as pd
import random
import sys
sys.path.append("/home/xyz/zjj/Django/AIServer/Favourite/AiCore")
from utils import get_k_neighbors

def _predict(record_list):
    '''
     record_list: 列表，每一项是一个int，是id
     return list 格式同上，长度为5
    '''
    features_np=np.load("features.npy")
    pddf=pd.read_csv('data.csv')
    id_list=pddf['ListedCoID'].values
    res=[]
    for uid in record_list:
        index=np.where(id_list==uid)[0][0]
        indexs=get_k_neighbors(features_np,index,5)
        print(pddf['ShortName'].values[indexs])
        res+=id_list[indexs].tolist()
    return random.sample(res, 5)
    
    
if __name__ == '__main__':
    res=_predict([101704,101779,101916])
    print(res)
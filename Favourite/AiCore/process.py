from re import S
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import get_k_neighbors
features_np=np.load("features.npy")
print(features_np.shape)
pddf=pd.read_csv('data.csv')
df =pddf['ShortName']
df_list=pddf['ListedCoID']
scores=np.load('score.npy')
scores=(scores-np.min(scores))/(np.max(scores)-np.min(scores))*100
scores=scores.astype(int)
score_df=pd.DataFrame(scores)
dfs=pd.concat([df_list,score_df],1)
dfs.to_csv('risk.csv')
# times=pddf[[str(1991+i) for i in range(31)]].values

# '''
#     总和，违规次数，近五年违规次数
# '''

# _sum=np.sum(times,1)
# _numbers=np.sum((times>0),1)
# _near_five=np.sum((times[:,-5:]),1)

# scores=[]
# for c in tqdm(range(len(_numbers))):
#     indexs=get_k_neighbors(features_np,c,4)
#     _sum_c=0
#     _numbers_c=0
#     _near_five_c=0
#     for i in indexs:
#         _sum_c+=_sum[i]
#         _numbers_c+=_numbers[i]
#         _near_five_c+=_near_five[i]
#     _sum_self=_sum[c]
#     _numbers_self=_numbers[c]
#     _near_five_self=_near_five[c]
#     scores.append(_near_five_self*10+_numbers_self*6+_sum_self*4+_sum_c*2+_numbers_c*3+_near_five_c*5)
# scores_np=np.array(scores)
# np.save('score.npy',scores_np)
# print(scores_np.shape,scores)

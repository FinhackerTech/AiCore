import numpy as np 

def get_k_neighbors(features,center_indx,k):
    """return a list of k nearest neighbors' index
    Args:
        features (numpy: feature_number * single_feature_size)  
        center_indx (int): the index of the conter point
        k (int): the number of nearest neighbors
    """
    center_feat=features[center_indx]
    center_norm=np.linalg.norm(center_feat)
    cos_similarities=[]
    for feat in features:
        num = float(np.dot(center_feat, feat))  
        denom =center_norm* np.linalg.norm(feat) 
        cos_similarities.append(0.5 + 0.5 * (num / denom) if denom != 0 else 0)
    #cos_similarities.sort(reverse=True)
    indexs=np.array(cos_similarities).argsort()[-(k+1):][::-1].tolist()
    if center_indx in indexs:
        indexs.remove(center_indx)
    else:
        indexs.pop()
    #print(indexs,np.array(cos_similarities)[indexs])
    return indexs
    
    
def get_bert_input(text, tokenizer, max_len=512):
    encoded_pair = tokenizer(text,
                    padding='max_length',  # Pad to max_length
                    truncation=True,       # Truncate to max_length
                    max_length=max_len,  
                    return_tensors='pt')  # Return torch.Tensor objects

    token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
    attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
    token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

    return token_ids, attn_masks, token_type_ids
    
    
if __name__ == '__main__':
    features=np.array([[2,3,4],[-1,-5,3],[1234,23,54]])
    get_k_neighbors(features,0,2)
import numpy as np
import math
from utils.models import Evaluator
#from sklearn.metrics import matthews_corrcoef as MCC
#from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def pred_index(sequences):
    for i, j in enumerate(sequences):
        if j[0] > j[2]:   
            return i-1
    return -1

if __name__ == "__main__":
    new_predict_conv = np.load('test_deepsig/conv5.npy')
    eval_train = np.load('../../data/features/eval.npy')

    count_dict = {}
    pred_cleavage_count, eval_cleavage_count = 0, 0
    for i, (each_predict, cleavage_site) in enumerate(zip(new_predict_conv, eval_train['cleavage_site'])):
        pred_site = pred_index(each_predict)
        
        if -1 != pred_site:
            pred_cleavage_count += 1
        if -1 != cleavage_site:
            eval_cleavage_count += 1

        if (pred_site > 0 and cleavage_site > 0):
            if abs(pred_site - cleavage_site) not in count_dict.keys():
                count_dict[abs(pred_site - cleavage_site)] = 1
            else:
                count_dict[abs(pred_site - cleavage_site)] += 1
    
    if  sum(count_dict.values()) > 0:
        mse = sum([i*i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values())
        rmse = math.sqrt(sum([i*i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values()))
        mae = sum([i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values())
        
        print('{} sequences have been predicted cleavage site by adv-fine-tuned sigUnoNet.'.format(pred_cleavage_count))
        print('{} sequences have cleavage sites in Evaluation Train.'.format(eval_cleavage_count))
        print('--------------------------------------------------------------------------')
        print('{} sequences both have cleavage sites.'.format(sum(count_dict.values())))
        print('Overall MSE: {}'.format(sum([i*i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values())))
        print('Overall RMSE: {}'.format(math.sqrt(sum([i*i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values()))))
        print('Overall MAE: {}'.format(sum([i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values())))

        with open('results/deepsig/result_1.txt', 'a') as f:
            f.write(f"{mse}, {rmse}\n")

    ### MCC
#    eval_data = np.load("../data/features/eval.npy")
#    new_pred = np.load("../test/pred.npy")
#    pred_label_list = []
#
#    for l in new_pred:
#        if np.argmax(l) == 2:
#            pred_label_list.append(2)
#        if np.argmax(l) == 1:
#            pred_label_list.append(1)
#        if np.argmax(l) == 0:
#            pred_label_list.append(0)
#    
#    print()
#    print("MCC:", MCC(eval_data['label'], pred_label_list)) 
#    #print("Precision:", precision_score(eval_data['label'], pred_label_list))
#    #print("Recall:", recall_score(eval_data['label'], pred_label_list))
#    #print("F1-score:", f1_score(eval_data['label'], pred_label_list))
#    print("Accuracy:", accuracy_score(eval_data['label'], pred_label_list))

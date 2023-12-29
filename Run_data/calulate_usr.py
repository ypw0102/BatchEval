import json
import glob
import random
import re
import math
import matplotlib.pyplot as plt
from scipy import stats
import numpy


extime=0

def cal_uni(s):
    count_s = 0
    counts = 0
    for i in range(len(s)):
        for j in range(i+1,len(s)):
            counts+=1
            count_s+=abs(s[i]-s[j])
    return count_s/counts



def get_score(ss,num=10):
    global extime
    new_ss = []
    for s_ in ss:
        s_=s_.lower()
        s=s_.split("scores:")[-1]
        try:
            matches = re.findall(r'\[[^\]]*\]', s)[0]
            s=matches[1:-1].split(",")
            s=[t.split(":")[-1].strip(" ") for t in s]
            for t in range(len(s)):
                try:
                    aa = float(s[t])
                except:
                    s[t] = '0'
            s=[float(t) for t in s]
        except:
            try:
                s=s_.split("scores:")[-1]
                scores = re.findall(r'sample\s*\d+:\s*([\d\.]+)', s)
                s=[float(t) for t in scores]
            except:
                s=[]
        if len(s)!=num and len(s)!=5:
            extime+=1
            s=[-1 for _ in range(num)]
        new_ss.append(s)
    return new_ss

def get_score_single(ss):
    new_ss = []
    # ss=random.sample(ss,15)
    for s in ss:
        # print(s)
        s=s.split(":")[-1].strip(" ")[0]
        try:s=[float(s)]
        except:
            s=[1.0]
        new_ss.append(s)
    return new_ss

def cal_cost(cur,model):
    price={3.5:[0.001,0.002],4:[0.03,0.06],4.5:[0.01,0.03],"llama2":[0,0]}
    cost1,cost2 = 0,0
    try:
        tem =cur['raw_data'][0]
        cost1 += len(tem['prompt'].split(" "))
        for tem in cur['response']:
            cost2 += len(tem.split(" "))
    except:
        print(1)
    return cost1/1000*price[model][0]+cost2/1000*price[model][1]

def get_corrs(a,b):
    pe, p = stats.pearsonr(a,b)
    sp, p = stats.spearmanr(a,b)
    return {"pe":pe,"sp":sp}




temp = 0.2
n = 1
batchsize = 5
sub_name = "batch"

for metrics in ['coherence']:
    data_name = "generate_usr_topical_heterogeneous_{}/".format(metrics[:3].lower())
    data_name_=data_name
    dir_name = data_name

    model = ["llama2",3.5,4][1]
    nums_ = [0]
    further_dirs = ["{}_{}_{}_{}_{}_gpt{}".format(metrics, batchsize, temp, n, sub_name, model)]
    dir_name = "API_Completion/" + dir_name

    for further_dir in further_dirs:
        print(further_dir)
        mappings = []
        batchsize = int(further_dir.split("_")[1])
        n=int(further_dir.split("_")[3])
        if batchsize==1:nums=[0]
        else:
            nums=nums_
        scores_all_set,mappings,ori_scores=[],[],[]
        avg_cost = 0
        sc_ralative={"sp":0,"pe":0,"ke":0}
        avg_sp,avg_pe,uni_count,uni_s,avg_sc_pe,avg_sc_sp=0,0,0,0,0,0
        for i in nums:
            idx_list,human_score_list = [],[]
            cur_sp_,cur_pe_=0,0
            all_data_dirs = glob.glob(dir_name+"{}.json/".format(i)+further_dir+"/raw_data/*")
            all_data = []
            new_all_data = []
            all_response = []
            for idx_ in range(len(all_data_dirs)):
                dir = dir_name+"{}.json/".format(i)+further_dir+"/raw_data/{}.json".format(idx_)
                with open(dir,"r")as f:
                    cur = json.load(f)
                all_response.append(cur)
                if batchsize==1:
                    cur_score = get_score_single(cur["response"])
                else:
                    cur_score = get_score(cur["response"],batchsize)
                idx_list+=[tem['idx'] for tem in cur['raw_data']]
                human_score_list+=[tem['scores'][metrics] for tem in cur['raw_data']]
                avg_cost+=cal_cost(cur,model)
                all_data.append(cur_score)

            for j in range(len(all_data[0])):
                cur_ss = []
                for k in range(len(all_data)):
                        cur_ss+=all_data[k][j]
                new_all_data.append(cur_ss)
            scores = new_all_data[0]
            cur_scores = {idx_list[i]:[idx_list[i],human_score_list[i],scores[i]] for i in range(len(scores))}
            human_score = [cur_scores[i][1] for i in range(len(scores))]
            scores = [cur_scores[i][2] for i in range(len(scores))]
            scores_all_set.append(scores)
        ensemble_score=[numpy.array([scores_all_set[i][j] for i in range(len(scores_all_set))]).mean().item() for j in range(len(scores))]
        results = get_corrs(ensemble_score,human_score)
        print(results)
        avg_cost/=len(scores)
        print("Cost per sample:{}".format(avg_cost))
        print('\n')




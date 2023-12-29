import json
import glob
import re
from scipy import stats
import numpy


def get_score(ss,num=10):
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
        if len(s)!=num and len(s)!=1:
            s=[-1 for _ in range(num)]
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
batchsize = 10
model = 4
iteration_round=5
sub_name = "batch"

for metrics in ["engagingness" ,'understandability','naturalness','coherence','overall']:
    data_name = "generate_usr_topical_heterogeneous_{}/".format(metrics[:3].lower())
    data_name_=data_name
    dir_name = data_name
    nums_ = list(range(iteration_round))
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
            cur_sp_,cur_pe_,cur_ent=0,0,0
            # print(i)


            with open("eval_data_new/{}{}.json".format(data_name_,i),"r")as f:
                raw = json.load(f)
                f.close()
            mapping = raw['maps']
            if 'scores' not in raw.keys():
                raw['scores'] =raw['annotations']
            human = [float(t[metrics]) for t in raw['scores']]
            human_new=[]
            for j in range(len(human)):
                human_new.append(human[mapping[str(j)]])
            all_data_dirs = glob.glob(dir_name+"{}.json/".format(i)+further_dir+"/raw_data/*")
            all_data = []
            new_all_data = []
            all_response = []
            for idx_ in range(len(all_data_dirs)):
                dir = dir_name+"{}.json/".format(i)+further_dir+"/raw_data/{}.json".format(idx_)
                with open(dir,"r")as f:
                    cur = json.load(f)
                all_response.append(cur)

                cur_score =  get_score(cur["response"],batchsize)
                avg_cost+=cal_cost(cur,model)
                all_data.append(cur_score)

            for j in range(len(all_data[0])):
                cur_ss = []
                for k in range(len(all_data)):
                        cur_ss+=all_data[k][j]
                new_all_data.append(cur_ss)

            scores = new_all_data
            score_new = []
            mappings.append(mapping)
            ori_scores.append(scores[0])
            for score in scores:
                score_new_=[]
                for j in range(len(score)):
                        ttt = mapping[str(j)]
                        try:
                            score_new_.append(score[ttt])
                        except:
                            print(1)

                score_new.append(score_new_)
            scores_all_set.append(score_new[0])
            for k in range(len(score_new)):
                r, p = stats.pearsonr(human_new, score_new[k])
                spearmanr = stats.spearmanr(human_new, score_new[k])
                ke,p = stats.kendalltau(human_new, score_new[k])
                cur_pe_+=r
                cur_sp_+=spearmanr.correlation
            cur_pe_ = cur_pe_/len(score_new)
            cur_sp_ = cur_sp_ / len(score_new)
            if len(score_new)>1:
                scores_all = [numpy.array([score_new[i][j] for i in range(len(score_new))]).mean().item() for j in range(len(score_new[0]))]
                r_sc, p = stats.pearsonr(human_new, scores_all)
                ss_sc,p = stats.spearmanr(human_new, scores_all)
                avg_sc_pe+=r_sc
                avg_sc_sp+=ss_sc
            avg_pe+=cur_pe_
            avg_sp+=cur_sp_
            scores.append(score_new)
        avg_pe/=len(nums)
        avg_sp/=len(nums)

        if len(score_new)>1:
            avg_sc_pe /= len(nums)
            avg_sc_sp /= len(nums)
            print("Avg SC")
            print("pearson_score: {}".format(avg_sc_pe))
            print("spearmanr_score: {}".format(avg_sc_sp))

        cross_scores_avg=[]
        for i in range(len(scores_all_set[0])):
            cur_point,cur_num=1e-5,1e-5
            for j in range(len(scores_all_set)):
                if scores_all_set[j][i]!=-1:
                    cur_point+=scores_all_set[j][i]
                    cur_num+=1
            cross_scores_avg.append(cur_point/cur_num)

        r_sc, p = stats.pearsonr(human_new, cross_scores_avg)
        ss_sc, p_ = stats.spearmanr(human_new, cross_scores_avg)
        print("Cross subset SC of {} pearson_score: {}".format(len(nums), r_sc))
        print("Cross subset SC of {} spearmanr_score: {}".format(len(nums), ss_sc))

        avg_cost/=len(raw['scores'])
        print("Cost per sample:{}".format(avg_cost))
    print('\n')

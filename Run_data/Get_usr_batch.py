import glob
import json
import random

import openai
import os
import time
import re
import random
import argparse
class Get:
    def __init__(self):
        self.prompt = ""
    def calc(self,query,temp=0,n=1,model='gpt-4-1106-preview'):
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": query}],
                temperature=temp,
                top_p=0.9,
                n=n
            )
            res = [tem["message"]["content"] for tem in response["choices"]]
            return res,{'prompt':response['usage']['prompt_tokens'],'completion':response['usage']['completion_tokens']}

def get_score(s_,num=10):
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
            s=[-1 for _ in range(num)]
        return s

def generate_new_data(dir_name,data_idx,further_dir,batchsize):
    generate_name = "API_Completion/"+dir_name+"{}.json/{}/raw_data/*".format(data_idx,further_dir)
    dirs = glob.glob(generate_name)
    raw_datas = []
    for dir in dirs:
        with open(dir,"r")as f:
            l=json.load(f)
            f.close()
        raw_datas.append(l)
    datas=[]
    for tem in raw_datas:
        score = get_score(tem['response'][0], num=batchsize)
        for i in range(len(tem['raw_data'])):
            tem['raw_data'][i]['modelscore'] = score[i]
            datas.append(tem['raw_data'][i])
    datas.sort(key=lambda x: x['modelscore'], reverse=True)
    new_datas = []
    bottle = len(datas)//batchsize
    for i in range(bottle):
        cur_batch = []
        for j in range(batchsize):
            cur_batch.append(datas[bottle*j+i])
        random.shuffle(cur_batch)
        new_datas+=cur_batch
    new_batch={}
    for key in new_datas[0].keys():
        new_batch[key] = [t[key] for t in new_datas]
    with open("eval_data_new/"+dir_name+"{}.json".format(data_idx+1),"w")as f:
        json.dump(new_batch,f)


def get_context(tem):
    start_s = "Speaker A: " if len(tem)%2 else "Speaker B: "
    s=""
    for t in tem[:-1]:
        s+=start_s+t.strip()+"\n"
        start_s = "Speaker A: " if start_s=="Speaker B: " else "Speaker B: "
    return s,tem[-1]



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--iteration_round', type=int, default=5)
    argparser.add_argument('--temperature', type=float, default=0.2)
    argparser.add_argument('--n', type=int, default=1)
    argparser.add_argument('--key', type=str, default="")
    argparser.add_argument('--org_key', type=str)
    argparser.add_argument("--batchsize", type = int, default = 10)
    argparser.add_argument('--model', type=str, default="gpt-4-1106-preview")
    argparser.add_argument('--criterion', type=str, default='engagingness')
    args = argparser.parse_args()
    openai.api_key = args.key
    openai.organization = args.org_key

    iteration_round=args.iteration_round
    temp = args.temperature
    n = args.n
    batchsize = args.batchsize
    model = args.model
    metrics = args.criterion
    data_name_ = "generate_usr_topical_heterogeneous_{}/".format(metrics[:3])
    dir_name_ = data_name_
    sub_name="batch"
    prompt_name = "{}_analyze_{}.txt".format(metrics[:3],sub_name)
    prompt_dir = "prompts/topical_chat/{}".format(prompt_name)
    prompt = open(prompt_dir).read()
    data_name_ = "eval_data_new/"+data_name_
    further_dir = "{}_{}_{}_{}_{}_{}".format(metrics,batchsize,temp,n,sub_name,model)


    for data_idx in range(iteration_round):
        data_name=data_name_+"{}.json".format(data_idx)
        dir_name=dir_name_+"{}.json".format(data_idx)
        with open(data_name,"r")as f:
            datas=json.load(f)
            f.close()
        dir_name="API_Completion/"+dir_name+"/"+further_dir
        os.makedirs(dir_name,exist_ok=True)
        os.makedirs(dir_name+"/raw_data",exist_ok=True)
        if not os.path.exists(dir_name+"/current.txt"):
            with open(dir_name+"/current.txt","w")as f:
                f.write("0")
                f.close()
        with open(dir_name+"/current.txt","r")as f:
                start = int(f.read())
                f.close()
        if not os.path.exists(dir_name+"/cost.txt"):
            with open(dir_name+"/cost.txt","w")as f:
                f.write("0 0")
                f.close()
        with open(dir_name+"/cost.txt","r")as f:
                tem = f.read()
                cost_now = {'prompt':int(tem.split(" ")[0]),'completion':int(tem.split(" ")[1])}
                f.close()

        def get_context(tem):
            start_s = "Speaker B: " if len(tem)%2 else "Speaker A: "
            s=""
            for t in tem:
                s+=start_s+t.strip()+"\n"
                start_s = "Speaker A: " if start_s=="Speaker B: " else "Speaker B: "
            return s

        Model = Get()
        for idx in range(start,len(datas["context"])//batchsize):
            tem = [{key:datas[key][idx*batchsize+i] for key in datas.keys()} for i in range(batchsize)]
            source = [get_context(datas["source"][idx*batchsize+i].split("\n")[:-2]) for i in range(batchsize)]
            s=""
            for i in range(batchsize):
                s+="Sample{}:\n\nConversation History:\n".format(i+5 if "icl" in sub_name else i+1)
                s+=source[i]
                s +="\nResponse:\n"
                s+="Speaker A: "+datas["system_output"][idx*batchsize+i]+"\n\n"
            cur_prompt = prompt.replace('{{Data}}', s).replace('{{number}}',str(batchsize))
            for i in range(batchsize):
                tem[i]['prompt'] = cur_prompt
            while True:
                try:
                    response,cost = Model.calc(cur_prompt,n=n,temp=temp,model=model)
                    print(idx)
                    for key in cost.keys():
                        cost_now[key]+=cost[key]
                    with open(dir_name+"/raw_data/{}.json".format(idx),"w")as f:
                        cur_t = {"raw_data":tem,"response":response}
                        json.dump(cur_t,f)
                        f.close()
                    with open(dir_name + "/current.txt", "w") as f:
                        f.write(str(idx+1))
                        f.close()
                    with open(dir_name + "/cost.txt", "w") as f:
                        f.write("{} {}".format(cost_now['prompt'],cost_now['completion']))
                        f.close()
                    break
                except Exception as e:
                    print(e)
                    print("Sleep 10s")
                    time.sleep(10)
        generate_new_data(dir_name_,data_idx,further_dir,batchsize)



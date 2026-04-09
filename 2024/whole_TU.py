# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd

data = pd.read_csv(sys.argv[1], sep="\t")
data = data.drop("target_gene", axis=1)
appris = pd.read_csv(sys.argv[2], sep="\t", header=None)
appris.columns = ["enst","degree"]
appris = appris.T
appris.columns = appris.iloc[0,0:]
appris = appris.iloc[1:,0:]
appris = appris.T

## summation total transcript tpm
sample = data.columns.tolist()
gene = sys.argv[3]
summation = [gene]

for i in sample[1:]:
    per_sum = data[i].sum()
    summation.append(per_sum)
summation = pd.DataFrame(summation).T
summation.columns = sample
trans = data[sample[0]][0:].tolist()
ENST = pd.DataFrame(columns=["ENST"])
for i in trans:
    k = trans.index(i)
    i = i.split(".")[0]
    ENST.loc[k] = i
data = pd.concat([ENST,data], axis=1)


## extract each transcript information & find major or minor form
enst = {}
app1 = []
app2 = []
app3 = []
app4 = []
app5 = []
app6 = []
for i in trans:
    i = i.split(".")[0]
    if i in appris.index.tolist():
        if appris.loc[i].values[0] == 1:
            app1.append(i)
        if appris.loc[i].values[0] == 2:
            app2.append(i)
        if appris.loc[i].values[0] == 3:
            app3.append(i)
        if appris.loc[i].values[0] == 4:
            app4.append(i)
        if appris.loc[i].values[0] == 5:
            app5.append(i)
        if appris.loc[i].values[0] == 6:
            app6.append(i)
    else:
        app6.append(i)

## 1-5 : PRINCIPAL
## less /home/omics/DATA5/hyeongu/DATA2_data/04_hyeongu/APPRIS/appris_data.principal.txt| cut -f 5 | sort -u <- check 할 수 있음
enst[1] = app1
enst[2] = app2
enst[3] = app3
enst[4] = app4
enst[5] = app5
enst[6] = app6  # Not reported & Alt annotation

for i in list(enst.keys()):
    if enst[i] == []:
        del enst[i]

#minor = enst[6] # minor 로 한정
whole = app1+app2+app3+app4+app5+app6
if len(whole) == 0:
    print ("DO NOT have minor forms gene is "+ gene)

## calculate certain transcript usate
for i in set(whole):    # each sig transcript
    if i in data["ENST"].tolist():
        target = data[data["ENST"]==i]
        target = target.drop("ENST", axis=1)
        total = pd.concat([target,summation],axis=0).set_index("gene_ENST",).rename_axis(None)
        for i in range(0,(total.shape[0]-1)):   # a number of
            tmp = total.index[i]
            total.loc[tmp] = (total.iloc[i] / total.loc[gene])
            result = pd.DataFrame(total.loc[tmp]).T
            result = result.replace(np.nan,0).astype(float)
            print ("resultshape: ",result.shape)
#            result.to_csv("whole_TU", sep="\t", mode="a",header=False)
            result.to_csv("/home/jiye/jiye/copycomparison/GC_POLO_tHRD/rawfiles/trim/2nd_align/quantified/processed/"+gene+"_TU", sep="\t", mode="a", header=False)
print ("Successed TU calculation of "+gene)

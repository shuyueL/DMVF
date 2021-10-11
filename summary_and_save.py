import scipy.io as sio
import numpy as np 

# Ss: {i: si # view i }
def interpret_summary_and_save_n_views(Ss, fnum, result_path):
    # view i
    for i in Ss:
        summary = np.zeros(fnum)
        n = len(Ss[i])
        for j in range(n):
            summary[int(Ss[i][j])] = 1
        name = result_path + 'view_' + str(i)
        sio.savemat(name,{'summary': summary})
    return

def interpret_summary_and_save_6_views(s0, s1, s2, s3, s4, s5, fnum, result_path):
    # view 0
    summary = np.zeros(fnum)
    n = len(s0)
    for i in range(n):
        summary[int(s0[i])] = 1
    name = result_path + 'view_0'
    sio.savemat(name,{'summary': summary})
    # view 1
    summary = np.zeros(fnum)
    n = len(s1)
    for i in range(n):
        summary[int(s1[i])] = 1
    name = result_path + 'view_1'
    sio.savemat(name,{'summary': summary})
    # view 2
    summary = np.zeros(fnum)
    n = len(s2)
    for i in range(n):
        summary[int(s2[i])] = 1
    name = result_path + 'view_2'
    sio.savemat(name,{'summary': summary})
    # view 3
    summary = np.zeros(fnum)
    n = len(s3)
    for i in range(n):
        summary[int(s3[i])] = 1
    name = result_path + 'view_3'
    sio.savemat(name,{'summary': summary})
    # view 4
    summary = np.zeros(fnum)
    n = len(s4)
    for i in range(n):
        summary[int(s4[i])] = 1
    name = result_path + 'view_4'
    sio.savemat(name,{'summary': summary})
    # view 5
    summary = np.zeros(fnum)
    n = len(s5)
    for i in range(n):
        summary[int(s5[i])] = 1
    name = result_path + 'view_5'
    sio.savemat(name,{'summary': summary})

    return
    
def interpret_summary_and_save(s0, s1, s2, test_num, fnum, result_path):
    # print('Overall processing rate = ', (len(s0) + len(s1) + len(s2))/(3*fnum*test_num))
    # view 0
    comb = 0
    summary = np.zeros(fnum)
    n = len(s0)
    # print('agent 1 processing rate = ', n/ (fnum*test_num))
    i = 0
    while i < n-1:
        if s0[i] < s0[i+1]:
            summary[int(s0[i])] = 1
        else :
            name = result_path + 'view_0_comb_' + str(comb)
            sio.savemat(name,{'summary': summary})
            summary = np.zeros(fnum)
            comb = comb + 1
        i = i + 1
    if n != 0:
        summary[int(s0[i])] = 1
    name = result_path + 'view_0_comb_' + str(comb)
    sio.savemat(name,{'summary': summary})
    # view 1
    comb = 0
    n = len(s1)
    i = 0
    summary = np.zeros(fnum)
    # print('agent 2 processing rate = ', n/ (fnum* test_num))
    while i < n-1:
        if s1[i] < s1[i+1]:
            summary[int(s1[i])] = 1
        else :
            name = result_path + 'view_1_comb_' + str(comb)
            sio.savemat(name,{'summary': summary})
            summary = np.zeros(fnum)
            comb = comb + 1
        i = i + 1
    if n != 0:
        summary[int(s1[i])] = 1
    name = result_path +'view_1_comb_' + str(comb)
    sio.savemat(name,{'summary': summary})
    # view 2
    comb = 0
    n = len(s2)
    i = 0
    summary = np.zeros(fnum)
    # print('agent 3 processing rate = ', n/ (fnum*test_num))
    while i < n-1:
        if s2[i] < s2[i+1]:
            summary[int(s2[i])] = 1
        else :
            name = result_path + 'view_2_comb_' + str(comb)
            sio.savemat(name,{'summary': summary})
            summary = np.zeros(fnum)
            comb = comb + 1
        i = i + 1
    if n != 0:
        summary[int(s2[i])] = 1
    name = result_path + 'view_2_comb_' + str(comb)
    sio.savemat(name,{'summary': summary})

    return
import pickle as pk
import random
dataset=['train','valid','test']
for tmp in dataset:
    f_train = pk.load(open(tmp+'_criminal_element3_large.pkl', 'rb'))
    dic=f_train
    fact_train = f_train['fact_list']
    law_labels_train = f_train['law_label_lists']
    accu_label_train = f_train['accu_label_lists']
    term_train = f_train['term_lists']
    keti=f_train['keti']
    zhuti=f_train['zhuti']
    zhuguan=f_train['zhuguan']
    a=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 8), (14, 13), (15, 11), (16, 11), (17, 14), (18, 15), (19, 16), (20, 17), (21, 18), (22, 19), (23, 20), (24, 9), (25, 11), (26, 21), (27, 22), (28, 23), (29, 24), (30, 11), (31, 25), (32, 9), (33, 20), (34, 21), (35, 11), (36, 26), (37, 27), (38, 28), (39, 29), (40, 30), (41, 4), (42, 6), (43, 31), (44, 4), (45, 21), (46, 32), (47, 9), (48, 33), (49, 11), (50, 34), (51, 35), (52, 36), (53, 11), (54, 9), (55, 37), (56, 11), (57, 38), (58, 6), (59, 11), (60, 9), (61, 9), (62, 39), (63, 40), (64, 6), (65, 41), (66, 6), (67, 42), (68, 43), (69, 37), (70, 44), (71, 38), (72, 45), (73, 11), (74, 46), (75, 47), (76, 33), (77, 8), (78, 11), (79, 11), (80, 48), (81, 9), (82, 49), (83, 11), (84, 11), (85, 15), (86, 6), (87, 50), (88, 51), (89, 52), (90, 53), (91, 54), (92, 55), (93, 9), (94, 56), (95, 57), (96, 9), (97, 58), (98, 59), (99, 60), (100, 9), (101, 61), (102, 9), (103, 62), (104, 9), (105, 63), (106, 64), (107, 65), (108, 9), (109, 57), (110, 66), (111, 11), (112, 67), (113, 9), (114, 68), (115, 69), (116, 70), (117, 57)]
    keguan=[]
    for i in law_labels_train:
        keguan.append(a[i][1])
    dic['keguan'] = keguan
    with open(tmp + '_criminal_element4_large.pkl', 'wb') as fid:
        pk.dump(dic, fid)
import numpy as np

TRAIN_DATA_PATH = "../data/training.dat"
TEST_DATA_PATH = "../data/testing.dat"
ITEM_BASED_OUTPUT_PATH = "../output/ItemBasedOutput.txt"
USER_BASED_OUTPUT_PATH = "../output/UserBasedOutput.txt"
MEM_BASED_OUTPUT_PATH = "../output/MemoryBasedOutput.txt"

MAX_UID = 2185  # 总用户数
MAX_MID = 74683 # 总电影数
NEAREST = 20    # 采用k近邻

alpha = 0.2     # user based 权重
beta = 1-alpha  # item based 权重

def read_train_data(train_file):
    """
    Args:
        train file handler
    Returns:
        numpy matrix (moive * user)
    """
    rank_matrix = np.matrix(np.zeros((MAX_MID,MAX_UID), dtype=np.int8, order='C'))
    line = train_file.readline()
    while line:
        line = line.split(',')
        uid = int(line[0])
        mid = int(line[1])
        rnk = int(line[2])
        rank_matrix[mid,uid] = rnk
        line = train_file.readline()
    return rank_matrix

def read_test_data(test_file):
    """
    Args:
        test file handler
    Returns:
        dict: {(uid,mid): prediction}
    """
    tasks = {}
    line = test_file.readline()
    while line:
        line = line.split(',')
        uid = int(line[0])
        mid = int(line[1])
        tasks[(uid,mid)] = 0
        line = test_file.readline()
    return tasks

if __name__ == "__main__":
    with open(TRAIN_DATA_PATH, 'r', encoding='UTF-8') as train_file:
        rank_matrix = read_train_data(train_file)
    with open(TEST_DATA_PATH, 'r', encoding='UTF-8') as test_file:
        tasks = read_test_data(test_file)
    # print("different user count:",len(set([item[0] for item in tasks.keys()])))
    # print("different movie count:",len(set([item[1] for item in tasks.keys()])))

    # 预先计算各平均值
    avg_user = np.mean(rank_matrix, axis=0).A.flatten()
    avg_movie = np.mean(rank_matrix, axis=1).A.flatten()

    previous_id = -1
    # user 为列向量，rank matrix 转存为列优先
    rank_matrix = np.asanyarray(rank_matrix, order='F')
    assert(type(rank_matrix) == np.matrix)
    assert(rank_matrix.shape == (MAX_MID,MAX_UID))
    assert(rank_matrix.flags['F_CONTIGUOUS'])
    # user based
    for (test_uid, test_mid) in sorted(tasks.keys(), key=lambda d: d[0]):
        # sorted by user
        # TODO: remove this to test all
        if test_uid != 0:
            break
        if test_uid != previous_id:
            # 重新计算关于 test_uid 的相关性向量
            previous_id = test_uid
            vec0 = rank_matrix[:,test_uid].A.flatten()
            vec0[vec0.nonzero()] = vec0[vec0.nonzero()] - avg_user[test_uid]
            sim = np.zeros(MAX_UID)
            for i in range(MAX_UID):
                # 避免自身相关性排第一
                if i != test_uid:
                    # 取第i列
                    vec1 = rank_matrix[:,i].A.flatten()
                    # 非零项减平均值，去中心化
                    vec1[vec1.nonzero()] = vec1[vec1.nonzero()] - avg_user[i]
                    # 避免 NaN
                    if np.sometrue(vec0) and np.sometrue(vec1):
                        # 计算相似度
                        sim[i] = np.inner(vec0,vec1)/(np.linalg.norm(vec0)*np.linalg.norm(vec1))
            # 排序并获得原索引
            sim_index = np.argsort(-sim)
            sim = np.flip(np.sort(sim))
        # 使用排好序的相似度向量
        # 选取前 k 个有评分且正相关的邻居
        positive_count = 0
        ranks = 0
        weights = 0
        for i in range(MAX_UID):
            # 忽略负相关
            if sim[i] <= 0 or positive_count >= NEAREST:
                break
            # 跳过无评分
            if rank_matrix[test_mid, sim_index[i]] == 0:
                continue
            ranks = ranks + sim[i] * (rank_matrix[test_mid, sim_index[i]] - avg_user[sim_index[i]])
            weights = weights + sim[i]
            positive_count = positive_count + 1
        # 预测
        if positive_count != 0:
            pred = avg_user[test_uid] + ranks / weights
        else:
            pred = avg_user[test_uid]
        # 存储
        # print('(', test_uid, ',', test_mid, '):', pred)
        tasks[(test_uid,test_mid)] = pred
    # 输出文件
    with open(TEST_DATA_PATH, 'r', encoding='UTF-8') as test_file:
        with open(USER_BASED_OUTPUT_PATH, 'w') as output_file:
            line = test_file.readline()
            while line:
                line = line.split(',')
                uid = int(line[0])
                mid = int(line[1])
                output_file.write(str(int(round(tasks[(uid,mid)]))) + '\n')
                line = test_file.readline()

    # item 为行向量，rank matrix 转存为行优先
    rank_matrix = np.asanyarray(rank_matrix, order='C')
    assert(type(rank_matrix) == np.matrix)
    assert(rank_matrix.shape == (MAX_MID,MAX_UID))
    assert(rank_matrix.flags['C_CONTIGUOUS'])

    # item based
    for (test_uid, test_mid) in sorted(tasks.keys(), key=lambda d: d[1]):
        # sorted by movie
        # TODO: remove this to test all
        if test_mid != 0:
            break
        if test_mid != previous_id:
            # 重新计算关于 test_mid 的相关性向量
            previous_id = test_mid
            vec0 = rank_matrix[test_mid, :].A.flatten()
            vec0[vec0.nonzero()] = vec0[vec0.nonzero()] - avg_movie[test_mid]
            sim = np.zeros(MAX_MID)
            for i in range(MAX_MID):
                # 避免自身相关性排第一
                if i != test_mid:
                    # 取第i行
                    vec1 = rank_matrix[i,:].A.flatten()
                    # 非零项减平均值，去中心化
                    vec1[vec1.nonzero()] = vec1[vec1.nonzero()] - avg_movie[i]
                    # 避免 NaN
                    if np.sometrue(vec0) and np.sometrue(vec1):
                        # 计算相似度
                        sim[i] = np.inner(vec0,vec1)/(np.linalg.norm(vec0)*np.linalg.norm(vec1))
            # 排序并获得原索引
            sim_index = np.argsort(-sim)
            sim = np.flip(np.sort(sim))
        # 使用排好序的相似度向量
        # 选取前 k 个有评分且正相关的邻居
        positive_count = 0
        ranks = 0
        weights = 0
        for i in range(MAX_MID):
            # 忽略负相关
            if sim[i] <= 0 or positive_count >= NEAREST:
                break
            # 跳过无评分
            if rank_matrix[sim_index[i], test_uid] == 0:
                continue
            ranks = ranks + sim[i] * rank_matrix[sim_index[i], test_uid]
            weights = weights + sim[i]
            positive_count = positive_count + 1
        # 预测
        if positive_count != 0:
            pred = ranks / weights
        else:
            pred = avg_movie[test_mid]
        # 存储
        # print('(', test_uid, ',', test_mid, '):', pred)
        tasks[(test_uid,test_mid)] = alpha*tasks[(test_uid,test_mid)] + beta*pred
    # 输出文件
    with open(TEST_DATA_PATH, 'r', encoding='UTF-8') as test_file:
        with open(MEM_BASED_OUTPUT_PATH, 'w') as output_file:
            line = test_file.readline()
            while line:
                line = line.split(',')
                uid = int(line[0])
                mid = int(line[1])
                output_file.write(str(int(round(tasks[(uid,mid)]))) + '\n')
                line = test_file.readline()

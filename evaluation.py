import numpy as np
import time
import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist

import utils

'''
@torch.no_grad()
1.节省显存：
在推理阶段，不需要计算梯度，因此可以禁用自动梯度计算来节省显存。
每次处理文本和图像时，不需要记录计算图，从而减少显存占用。
2.提高速度：
禁用梯度计算后，不需要构建计算图，可以提高推理速度。
3.避免意外修改梯度状态：
在推理阶段，不需要累积梯度，禁用梯度计算可以避免梯度状态被意外修改。
'''
@torch.no_grad()
def evaluation(model, data_loader, device, args):
    """
    模型评估函数。

    该函数用于在给定的数据集上评估模型的性能。它主要通过计算和比较图像嵌入和文本嵌入的相似性来完成。

    参数:
    - model: 待评估的模型。
    - data_loader: 数据加载器，用于提供评估数据。
    - device: 模型和数据所在的设备（如CPU或GPU）。
    - args: 包含其他参数（如是否分布式训练）的参数。

    返回:
    - score_matrix_i2t: 图像到文本的相似性评分矩阵。
    - score_matrix_t2i: 文本到图像的相似性评分矩阵。
    """

    # 将模型设置为评估模式
    '''
    设置模型为评估模式的影响
    Batch Normalization 层的行为变化：
        训练模式：在训练模式下，Batch Normalization 层会计算当前批次的均值和方差，并用于归一化当前批次的数据。同时，它还会更新运行均值和运行方差。
        评估模式：在评估模式下，Batch Normalization 层会使用之前训练过程中累积的运行均值和运行方差来进行归一化，而不是当前批次的统计量。这样可以避免评估时的不稳定性。
    Dropout 层的行为变化：
        训练模式：在训练模式下，Dropout 层会随机丢弃一部分神经元，以防止过拟合。
        评估模式：在评估模式下，Dropout 层不会丢弃任何神经元，而是将每个神经元的输出乘以 dropout 概率，以保持期望输出不变。
    其他层的行为变化：
        一些其他层（如某些自定义层）也可能在训练模式和评估模式下有不同的行为。例如，某些自定义的正则化层可能在评估模式下会有不同的处理方式。
    '''

    model.eval()

    # 开始计算评估特征
    print('Computing features for evaluation...')
    start_time = time.time()

    # 提取文本嵌入
    # texts：从数据加载器的数据集中提取文本数据。
    # num_text：计算文本数据的数量。
    texts = data_loader.dataset.text   # 是一个列表
    num_text = len(texts)
    text_bs = 256  # 文本批处理大小 text_bs：设置文本批处理大小为 256。
    # 初始化文本嵌入列表：
    text_embeds = []
    # 遍历文本数据，每次处理 text_bs 个文本
    for i in range(0, num_text, text_bs):
        # 提取当前批次的文本数据。
        text = texts[i: min(num_text, i + text_bs)]
        # 对文本数据进行预处理，并移动到指定设备（如GPU）。
        text_input = data_loader.dataset.preprocess_text(text).to(device)
        # 使用模型计算文本嵌入
        text_embed = model.encode_text(text_input)
        # 将当前批次的文本嵌入添加到列表中
        text_embeds.append(text_embed)
    # 使用torch.cat拼接所有批次的文本嵌入，得到最终的文本嵌入张量。
    text_embeds = torch.cat(text_embeds, dim=0)

    # 提取图像嵌入
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed = model.encode_image(image)
        image_embeds.append(image_embed)

    image_embeds = torch.cat(image_embeds, dim=0)

    # 计算图像和文本间的相似性矩阵
    score_matrix_i2t, score_matrix_t2i = model.get_similarity(
        image_embeds, text_embeds)
    # contiguous(): 将张量重新分配内存，以节省内存。
    score_matrix_i2t = score_matrix_i2t.contiguous()
    score_matrix_t2i = score_matrix_t2i.contiguous()

    # 如果是分布式训练，同步相似性矩阵
    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    # 计算总评估时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    # 返回相似性评分矩阵
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()



@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    """
    Perform Image-Text Matching (ITM) evaluation.

    This function evaluates the ITM task by calculating retrieval metrics for images-to-text and text-to-images.
    It computes the ranks of correct matches and based on these ranks, calculates recall@1, recall@5, recall@10, and mean recall.

    Parameters:
    - scores_i2t: Numpy array, the similarity scores for images-to-text retrieval.
    - scores_t2i: Numpy array, the similarity scores for text-to-images retrieval.
    - txt2img: List, for each text, it contains the index of the matching image in the scores_t2i.
    - img2txt: List, for each image, it contains the indices of the matching texts in the scores_i2t.

    Returns:
    - eval_result: Dict, containing various evaluation metrics for text-to-image and image-to-text retrieval.
    """

    # Images->Text
    #
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        # 使用 np.argsort(score) 得到得分从小到大的索引。
        # [::-1] 反转索引顺序，使得索引按照得分从大到小排列。
        # score = [0.47942554, 0.76821759, 0.52798131, 0.55277106, 0.84622113, 0.6945283, 0.23651333, 0.49212918, 0.56294261, 0.41281757]
        # inds = [4, 1, 8, 3, 2, 5, 7, 0, 9, 6]
        inds = np.argsort(score)[::-1]
        # 初始化 rank 为一个非常大的数（例如 1e20），以便找到最小的匹配排名。
        rank = 1e20
        # Find the rank of the correct match
    #     self.img2txt = {
    #     （imageindex）0: （img-txt）[0, 1, 2, 3, 4],
    #     1: [5, 6, 7, 8, 9]
    #     }
    #     self.txt2img = {
    #     0: 0,
    #     1: 0,
    #     2: 0,
    #     3: 0,
    #     4: 0,
    #     5: 1,
    #     6: 1,
    #     7: 1,
    #     8: 1,
    #     9: 1
    #       }
        # 遍历 img2txt[index] 中的每个正确文本索引。
        for i in img2txt[index]:
            # 查找正确文本在排序后的索引中的位置。
            tmp = np.where(inds == i)[0][0]  # Get the position of the correct match in the sorted indices
            if tmp < rank:
                rank = tmp  # Update the rank if a better match is found
        ranks[index] = rank  # Record the rank for this image

    # Compute metrics for images-to-text retrieval
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # Calculate recall@1
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)  # Calculate recall@5
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)  # Calculate recall@10

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]  # Sort scores in descending order to get indices
        # Find the rank of the correct match
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics for text-to-images retrieval
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # Calculate recall@1
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)  # Calculate recall@5
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)  # Calculate recall@10

    # Calculate mean recall
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2  # Overall mean recall

    # Round the metrics to a specified number of decimal places for easier reading
    reserveNumber = 2

    # Organize the calculated metrics into a dictionary for return
    eval_result = {'txt_r1': round(tr1,reserveNumber),
                   'txt_r5': round(tr5,reserveNumber),
                   'txt_r10': round(tr10,reserveNumber),
                   'txt_r_mean': round(tr_mean,reserveNumber),
                   'img_r1': round(ir1,reserveNumber),
                   'img_r5': round(ir5,reserveNumber),
                   'img_r10': round(ir10,reserveNumber),
                   'img_r_mean': round(ir_mean,reserveNumber),
                   'r_mean': round(r_mean,reserveNumber)}
    return eval_result


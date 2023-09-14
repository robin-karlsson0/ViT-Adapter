# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import pandas as pd
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(suffix='.npy',
                                                     delete=False,
                                                     dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def print_sim_treshs(sim_threshs: list, txts: list, sim_poss: list,
                     sim_negs: list):
    """
    Args:
        sim_threshs: List of similarity thresholds for each category.
        txts: List of category descriptions.
        sim_poss: List of similarity values for true elements.
        sim_negs: List of similarity values for false elements.
    """
    print('\nSimilarity thresholds (idx, txt, thresh, correct ratio pos|neg,'
          'num pos|neg)')
    entries = []
    for idx, (txt, sim) in enumerate(zip(txts, sim_threshs)):
        if len(sim_poss[idx]) == 0 and len(sim_negs[idx]) == 0:
            continue

        sim_pos = np.array(sim_poss[idx])
        sim_neg = np.array(sim_negs[idx])
        num_pos = len(sim_pos)
        num_neg = len(sim_neg)
        ratio_true = np.sum(sim_pos > sim) / num_pos if num_pos > 0 else None
        ratio_false = np.sum(sim_neg < sim) / num_neg if num_neg > 0 else None

        entry = {
            'txt': [txt],
            'sim': [sim],
            'ratio_true': [ratio_true],
            'ratio_false': [ratio_false],
            'num_pos': [num_pos],
            'num_false': [num_neg]
        }
        entries.append(entry)

    # Merge entries into data dictionary
    data = {}
    for d in entries:
        for k, v in d.items():
            if k in data:
                data[k] += v
            else:
                data[k] = v

    df = pd.DataFrame(data)
    print(df)


def single_gpu_test_thresh(model,
                           data_loader,
                           data_loader_thresh,
                           show=False,
                           out_dir=None,
                           efficient_test=False,
                           opacity=0.5,
                           pre_eval=False,
                           format_only=False,
                           format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    dataset_thresh = data_loader_thresh.dataset.dataset  # Subsampled dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    loader_indices_thresh = data_loader_thresh.batch_sampler

    ################################
    #  Compute optimal thresholds
    ################################
    K = len(dataset.CLASSES)
    sim_poss = [[] for _ in range(K)]
    sim_negs = [[] for _ in range(K)]

    prog_bar = mmcv.ProgressBar(len(data_loader_thresh))
    for batch_indices, data in zip(loader_indices_thresh, data_loader_thresh):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        sim_pos, sim_neg = dataset_thresh.comp_sim(result,
                                                   indices=batch_indices)

        for k in range(K):
            sim_poss[k].extend(sim_pos[k])
            sim_negs[k].extend(sim_neg[k])

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    # Compute thresholds as optimal decision boundary points
    sim_threshs = [None] * K
    for k in range(K):
        sim_pos = sim_poss[k]
        sim_neg = sim_negs[k]
        if len(sim_pos) > 0 and len(sim_neg) > 0:
            dec_b = dataset_thresh.comp_logreg_decision_point(sim_pos, sim_neg)
            sim_threshs[k] = dec_b
    # Clip similarity thresholds
    sim_threshs = [
        min(1, max(-1, s)) if s is not None else s for s in sim_threshs
    ]

    txts = dataset_thresh.CLASSES
    print_sim_treshs(sim_threshs, txts, sim_poss, sim_negs)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(img_show,
                                         result,
                                         palette=dataset.PALETTE,
                                         show=show,
                                         out_file=out_file,
                                         opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(result,
                                            indices=batch_indices,
                                            **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval_thresh(result,
                                             sim_threshs,
                                             indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test_thresh(model,
                          data_loader,
                          data_loader_thresh,
                          tmpdir=None,
                          gpu_collect=False,
                          efficient_test=False,
                          pre_eval=False,
                          format_only=False,
                          format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    dataset_thresh = data_loader_thresh.dataset.dataset  # Subsampled dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler
    loader_indices_thresh = data_loader_thresh.batch_sampler

    ################################
    #  Compute optimal thresholds
    ################################
    K = len(dataset.CLASSES)
    sim_poss = [[] for _ in range(K)]
    sim_negs = [[] for _ in range(K)]

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader_thresh))

    for batch_indices, data in zip(loader_indices_thresh, data_loader_thresh):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        sim_pos, sim_neg = dataset_thresh.comp_sim(result,
                                                   indices=batch_indices)

        for k in range(K):
            sim_poss[k].extend(sim_pos[k])
            sim_negs[k].extend(sim_neg[k])

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()

    # Compute thresholds as optimal decision boundary points
    sim_threshs = np.zeros(K)
    for k in range(K):
        sim_pos = sim_poss[k]
        sim_neg = sim_negs[k]
        if len(sim_pos) > 0 and len(sim_neg) > 0:
            dec_b = dataset_thresh.comp_logreg_decision_point(sim_pos, sim_neg)
            sim_threshs[k] = dec_b
    # Clip similarity thresholds
    sim_threshs = [
        min(1, max(-1, s)) if s is not None else s for s in sim_threshs
    ]

    if rank == 0:
        txts = dataset_thresh.CLASSES
        print_sim_treshs(sim_threshs, txts, sim_poss, sim_negs)

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(result,
                                            indices=batch_indices,
                                            **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval_thresh(result,
                                             sim_threshs,
                                             indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

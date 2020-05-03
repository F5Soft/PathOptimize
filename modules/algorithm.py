"""
遗传算法模块
Author: bsy
Date: 2020-03-05
"""

import copy
import time
import random

import numpy as np

from modules.entity.network import Network


def population_init(initial_network: Network, population_size=1):
    """
    初始化网络种群
    :param initial_network: 初始化基于的原始网络（即未生成车辆范围和路径的网络）
    :param population_size: 种群大小
    :return: List[Network] 初始化产生的种群
    """
    population = [copy.copy(initial_network) for i in range(population_size)]
    for p in population:
        p.coverage_init()
        p.path_generate()
    return population


def natural_selection(population):
    """
    根据网络的适应度函数进行自然选择
    :param population: 待选择的种群
    :return: None
    """
    weights = np.array([p.adaptive() for p in population])
    if np.max(weights) < 0:
        weights -= np.min(weights)
    for i in range(len(population)):
        population[i] = random.choices(population, weights, k=1)[0]


def gene_mutation(population, mutation_rate=0.1):
    """
    使网络种群中所有卡车的基因按照给定的概率突变
    :param population: 网络种群
    :param mutation_rate: 每个卡车的基因的突变率
    :return: None
    """
    for p in population:
        p.coverage_mutation(mutation_rate)


def gene_recombination(population, recombination_rate=0.1):
    """
    网络种群中部分个体之间基因相互重组
    :param population: 网络种群
    :param recombination_rate: 单个网络与其他随机一个网络发生部分基因交换的概率
    :return:
    """
    n = len(population)
    for p in population:
        if random.random() < recombination_rate:
            p.coverage_recombination(population[random.randrange(n)])


def genetic_algorithm(population, iteration=10, mutation_rate=0.1, recombination_rate=0.1):
    """
    遗传算法
    :param population: 网络种群
    :param iteration: 迭代次数
    :param mutation_rate: 突变概率
    :param recombination_rate: 重组概率
    :return: Network 整个过程中的适应度最高的网络
    """
    time_start = time.time()
    network_best = population[0]
    while iteration > 0:
        natural_selection(population)
        gene_mutation(population, mutation_rate)
        gene_recombination(population, recombination_rate)
        # 根据突变后的基因，重新计算网络中每辆卡车范围和路径
        for p in population:
            p.coverage_allocate()
            p.path_generate()
        for p in population:
            if p.adaptive() > network_best.adaptive():
                network_best = p
        # todo 优化遗传算法的速度和准确率
        # 如果适应度小于0，说明重量或者距离超过上限，不可取，但是会导致这里跑很久，不知道怎么办
        if network_best.adaptive() > 0:
            iteration -= 1
    return network_best

import cv2
import numpy as np
from numba import jit, njit
import copy


def get_mesh_center(img, step):
    """
    求取初始网格的中心点
    :param img: 输入图像
    :param step: 方格的边长
    :return:
    """
    coordinate = []
    width_num = np.round(img.shape[1] / step)
    height_num = np.round(img.shape[0] / step)
    for i in range(int(width_num)):
        for j in range(int(height_num)):
            width = int(i * step + step / 2)
            height = int(j * step + step / 2)
            if width <= img.shape[1] - step / 3 and height <= img.shape[0] - step / 3:
                coordinate.append([width, height])
    coordinate = np.array(coordinate, dtype=np.int)
    return coordinate


def local_min_gradient(img, coordinates):
    """
    求取初始网格中心点的3*3区域内的梯度
    :param img: 图片
    :param coordinates: 中心点坐标
    :return:
    """
    min_x = 0
    min_y = 0
    calculate_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_coordinate = []
    for coordinate in coordinates:
        min_grad = 1e10
        for x in range(coordinate[0] - 1, coordinate[0] + 2):
            for y in range(coordinate[1] - 1, coordinate[1] + 2):
                origin = calculate_img[y, x]
                origin_x = calculate_img[y, x + 1]
                origin_y = calculate_img[y + 1, x]
                grad_x = int(origin_x) - int(origin)
                grad_y = int(origin_y) - int(origin)
                grad = abs(grad_x) + abs(grad_y)
                if grad < min_grad:
                    min_grad = grad
                    min_x = x
                    min_y = y

        new_coordinate.append([min_x, min_y])
    return np.array(new_coordinate, dtype=np.int32)


@njit()
def calculate_distance(centers, img, step):
    """
    求取每个像素点到中心点的距离，搜索距离为2*s
    :param centers: 方框的中间点
    :param img: 图片
    :param step: 步长
    :return:
    """

    # todo 初始化距离值
    distance_map = np.ones((img.shape[0], img.shape[1]), dtype=np.float64)
    distance_map = distance_map * 100000
    cluster_map = distance_map.astype(np.int32)

    for iter_ in range(10):
        # todo 计算哪些像素属于哪一个簇
        counter = 0

        for center in centers:
            for i in range(center[0] - step, center[0] + step):
                for j in range(center[1] - step, center[1] + step):
                    if 0 <= i < img.shape[1] and 0 <= j < img.shape[1]:

                        distance_spacial = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                        distance_color = np.sqrt(
                            (int(img[j, i, 0]) - int(img[center[1], center[0], 0])) ** 2 +
                            (int(img[j, i, 1]) - int(img[center[1], center[0], 1])) ** 2 +
                            (int(img[j, i, 2]) - int(img[center[1], center[0], 2])) ** 2)

                        distance = np.sqrt((distance_spacial / step) ** 2 + (distance_color / 150) ** 2)
                        if distance_map[j, i] > distance:
                            distance_map[j, i] = distance
                            cluster_map[j, i] = counter
            counter += 1
        # todo 更新center的值
        centers = centers - centers
        counter = np.zeros((centers.shape[0],), dtype=np.int32)
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                no_cluster = cluster_map[row, col]
                centers[no_cluster, 0] += col
                centers[no_cluster, 1] += row
                counter[no_cluster] = counter[no_cluster] + 1

        for i in range(centers.shape[0]):
            centers[i, 0] /= counter[i]
            centers[i, 1] /= counter[i]
        centers = centers.astype(np.int32)

    return centers, cluster_map


@njit()
def get_super_pixel(img, new_center, cluster_map):
    """
    通过簇信息获取最后的超像素图像显示
    :param img: 原始图片
    :param new_center: 中心点位置
    :param cluster_map: 簇图
    :return:
    """
    result = np.zeros_like(img, dtype=np.uint8)
    color = np.zeros((new_center.shape[0], 3), dtype=np.uint8)
    for i in range(new_center.shape[0]):
        color[i, :] = img[new_center[i, 1], new_center[i, 0], :]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j, :] = color[cluster_map[i, j], :]
    return result


@njit()
def get_contours(cluster_map):
    contours = []
    a = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    b = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    for row in range(cluster_map.shape[0]):
        for col in range(cluster_map.shape[1]):
            # todo 与其八领域比较
            for i in range(8):
                row_ = row + a[i]
                col_ = col + b[i]
                if 0 <= row_ < cluster_map.shape[0] and 0 <= col_ < cluster_map.shape[1] \
                        and not cluster_map[row, col] == cluster_map[row_, col_]:
                    contours.append([col, row])
    contours = np.array(contours, dtype=np.int32)
    return contours

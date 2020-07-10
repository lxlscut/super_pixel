import cv2
import numpy as np
from SILC.gernerate_super_pixel import get_mesh_center, local_min_gradient, calculate_distance, get_super_pixel, \
    get_contours
from util.draw import Draw
import copy

if __name__ == '__main__':
    img = cv2.imread("C:\\Users\\26840\\Pictures\\lena.jpg")
    mesh_center = get_mesh_center(img, 30)
    new_mesh_center = local_min_gradient(img, mesh_center)
    print(mesh_center.shape)
    draw = Draw()
    mesh_center_pic = draw.draw(src=img, src_point=mesh_center)
    mesh_center_new_pic = draw.draw(src=mesh_center_pic, src_point=new_mesh_center)
    new_center, cluster_map = calculate_distance(new_mesh_center.astype(np.int32), img, 30)
    print(np.where(cluster_map > 90000))
    result = get_super_pixel(img, new_center, cluster_map)
    contours = get_contours(cluster_map)
    new_center_pic = draw.draw(src=img, src_point=new_center)
    contour = draw.draw_contours(src=new_center_pic, src_point=contours)
    cv2.imshow("contour", contour)
    cv2.imshow("result", result)
    cv2.imshow("new_center", new_center_pic)
    cv2.imshow("mesh_center_pic", mesh_center_pic)
    cv2.imshow("mesh_center_new_pic", mesh_center_new_pic)
    cv2.imshow("img", img)
    cv2.waitKey(0)

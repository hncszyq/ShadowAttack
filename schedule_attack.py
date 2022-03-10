# -*- coding: utf-8 -*-

import cv2
import numpy as np
from datetime import datetime
from utils import draw_shadow
from utils import shadow_edge_blur
from gtsrb import test_single_image


def cal_angle(year, month, day, h, m, s, lon, lat, timezone):

    doy = (datetime(year, month, day) - datetime(year, 1, 1)).days + 1

    n0 = 79.6764 + 0.2422 * (year - 1985) - int((year - 1985) / 4.0)
    sitar = 2 * np.pi * (doy - n0) / 365.2422
    ed1 = 0.3723 + 23.2567 * np.sin(sitar) + 0.1149 * np.sin(2 * sitar) \
        - 0.1712 * np.sin(3 * sitar) - 0.758 * np.cos(sitar) + 0.3656 \
        * np.cos(2 * sitar) + 0.0201 * np.cos(3 * sitar)
    ed = ed1 * np.pi / 180

    if lon >= 0:
        if timezone == -13:
            d_lon = lon - (np.floor((lon * 10 - 75) / 150) + 1) * 15.0
        else:
            d_lon = lon - timezone * 15.0
    else:
        if timezone == -13:
            d_lon = (np.floor((lon * 10 - 75) / 150) + 1) * 15.0 - lon
        else:
            d_lon = timezone * 15.0 - lon

    et = 0.0028 - 1.9857 * np.sin(sitar) + 9.9059 * np.sin(2 * sitar) \
        - 7.0924 * np.cos(sitar) - 0.6882 * np.cos(2 * sitar)
    gtd_t1 = h + m / 60.0 + s / 3600.0 + d_lon / 15
    gtd_t = gtd_t1 + et / 60.0
    d_time_angle1 = 15.0 * (gtd_t - 12)
    d_time_angle = d_time_angle1 * np.pi / 180
    latitude_arc = lat * np.pi / 180

    height_angle_arc = np.arcsin(np.sin(latitude_arc) * np.sin(ed)
                                 + np.cos(latitude_arc) * np.cos(ed)
                                 * np.cos(d_time_angle))

    cos_azimuth_angle = (np.sin(height_angle_arc) * np.sin(
        latitude_arc) - np.sin(ed)) / np.cos(
        height_angle_arc) / np.cos(latitude_arc)
    azimuth_angle_arc = np.arccos(cos_azimuth_angle)
    height_angle = height_angle_arc * 180 / np.pi
    azimuth_angle1 = azimuth_angle_arc * 180 / np.pi

    if d_time_angle < 0:
        azimuth_angle = 180 - azimuth_angle1
    else:
        azimuth_angle = 180 + azimuth_angle1

    return height_angle, azimuth_angle


def cal_delta(height_angle, azimuth_angle, dis):
    x_delta = dis / np.sin((azimuth_angle - 90) * np.pi / 180.) * np.tan(height_angle * np.pi / 180.)
    z_delta = dis / np.tan((azimuth_angle - 90) * np.pi / 180.)
    return x_delta, dis, z_delta


if __name__ == "__main__":

    # load clean image and it's mask
    image = cv2.imread('./tmp/gtsrb_30.png')
    mask = cv2.imread('./tmp/gtsrb_30_mask.png')
    inner_list = np.where(mask[:, :, -1] > 0)

    # calculate the position of the sun at 8:30 am
    HeightAngle, AzimuthAngle = cal_angle(2021, 9, 1, 8, 30, 0, 0, 45, 0)

    # optimized coordinates of P on the traffic sign (XOZ plain).
    shadow_coord = np.array([[37.247183201129175, 0., 54.42535467575114],
                             [201.08244542240166, 0., 160.67371239784262],
                             [21.69656212009891, 0., -44.40775159299855]])

    # calculate the position of the cardboard
    distance = 224 * 1. / 0.6
    delta_x, delta_y, delta_z = cal_delta(HeightAngle, AzimuthAngle, distance)
    board_coord = shadow_coord + np.array([delta_x, delta_y, delta_z])

    # calculate the position of the shadow from 8:25 am to 8:35 am
    H, M, S = 8, 25, 0
    while True:
        print(f'{H}:{M}:{str(S).zfill(2)}', end=' ')
        HeightAngle, AzimuthAngle = cal_angle(2021, 9, 1, H, M, S, 0, 45, 0)
        delta_x, delta_y, delta_z = cal_delta(HeightAngle, AzimuthAngle, distance)
        shadow_coord = board_coord - np.array([delta_x, delta_y, delta_z])
        shadow_image, shadow_area = draw_shadow(
            shadow_coord[:, [0, -1]].flatten(), image, inner_list, 0.43)
        shadow_image = shadow_edge_blur(shadow_image, shadow_area, 5)
        cv2.imwrite('./tmp/temp.bmp', shadow_image)
        test_single_image('./tmp/temp.bmp', 1)
        cv2.imshow('image', shadow_image)
        cv2.waitKey(10)
        S += 1
        if S == 60:
            S = 0
            M += 1
        if M == 35:
            break

#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/06/03
# @author:aiden
# Recognize color and calculate the object's position information(识别颜色以及计算物体的位置信息)
import cv2
import math
import copy
import numpy as np

class ColorDetection:
    def __init__(self, lab_config, detect_info, debug=False):
        '''
        lab_config: lab threshold, in dictionary format(lab阈值，字典形式)
        detect_info: Detection type, in ROS format(检测类型，ros格式)
        '''
        self.debug = debug
        self.lab_data = lab_config
        self.detect_info = detect_info.data
        self.image_process_size = [160, 120]  # Image processing size(图像处理大小)
        
    def update_lab_config(self, lab_data):
        '''
        Update LAB parameter(更新lab参数)
        :param lab_data: lab threshold, in dictionary format(lab阈值，字典形式)
        :return:
        '''
        self.lab_data = lab_data

    def update_detect_info(self, detect_info):
        '''
        Update detection type(更新检测类型)
        :param detect_info: Detection type, in ROS format(检测类型，ros格式)
        '''
        self.detect_info = detect_info
        for i in self.detect_info:
            print('current_ detect type: ', i.detect_type)

    def value_remapped(self, x, in_min, in_max, out_min, out_max, data_type=float):
        '''
        Map a value from the specified range to another(将一个值从给定区间映射到另一个区间)
        param x: The current input value(当前输入值)
        param in_min: The minimum value of the current range(当前区间最小值)
        param in_max: The maximum value of the current range(当前区间最大值)
        param out_min: The minimum value of the mapped range(映射到的区间最小值)
        param out_max: The maximum value of the mapped range(映射到的区间最大值)
        param data_type: The final structure type(最后的结构类型)
        '''
        return data_type((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    def point_remapped(self, point, now, new, data_type=float):
        """
        Map the coordinate of one point from a picture to a new picture of different size(将一个点的坐标从一个图片尺寸映射的新的图片上)
        :param point: Coordinate of point(点的坐标)
        :param now: Size of current picture(现在图片的尺寸)
        :param new: New picture size(新的图片尺寸)
        :return: New point coordinate(新的点坐标)
        """
        x, y = point
        now_w, now_h = now
        new_w, new_h = new
        new_x = self.value_remapped(x, 0, now_w, 0, new_w)
        new_y = self.value_remapped(y, 0, now_h, 0, new_h) 
        
        return data_type(new_x), data_type(new_y)

    def get_area_max_contour(self, contours, min_area=50, max_area=640*480):
        """
        Get the contour whose area is the largest. Filter out those whose area is too small(获取轮廓中面积最重大的一个, 过滤掉面积过小的情况)
        :param contours: Contour list(轮廓列表)
        :param threshold: Area threshold. Contour whose area is less than this value will be filtered out(面积阈值, 小于这个面积的轮廓会被过滤)
        :return: If the maximum contour area is greater than this threshold, return the
        largest contour, otherwise return None(如果最大的轮廓面积大于阈值则返回最大的轮廓, 否则返回None)
        """
        contour_area_max = 0
        area_max_contour = None

        for c in contours:  # Traverse through all contours(历遍所有轮廓)
            contour_area_temp = math.fabs(cv2.contourArea(c))  # Calculate the area of the contour(计算轮廓面积)
            if contour_area_temp > contour_area_max:
                contour_area_max = contour_area_temp
                if max_area >= contour_area_temp >= min_area:  # Filter interference(过滤干扰)
                    area_max_contour = c

        return area_max_contour,  contour_area_max  # Return the contour with the maximum area(返回最大的轮廓)

    def get_lab_binary(self, lab_image, lower, upper, kernel=3):
        '''
        Binarize Lab and perform erosion and dilation(将lab二值化，并进行腐蚀膨胀)
        param lab_image: Input the image converted to Lab color space(输入转换为lab后的图像)
        param lower: The first range of the threshold(阈值的第一个区间)
        param upper: The second range of the threshold(阈值的第二个区间)
        param kernel: (The size of the erosion and dilation kernel(腐蚀膨胀的核大小)
        '''
        binary = cv2.inRange(lab_image, lower, upper)  # Binarization(二值化)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        eroded = cv2.erode(binary, element)  # Erosion(腐蚀)
        dilated = cv2.dilate(eroded, element)  # Dilation(膨胀)
        
        return dilated

    def get_binary_contour(self, binary, min_area, max_area):
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # Find all contours(找出所有轮廓)
        max_contour, contour_area = self.get_area_max_contour(contours, min_area, max_area)

        return max_contour, contour_area

    def get_lab_contour(self, lab_image, lower, upper, min_area, max_area):
        '''
        Convert LAB to binary image and obtain the contour with the maximum area(将lab转换为二值图并获取最大轮廓)
        param lab_image: Input the image converted to Lab color space(输入转换为lab后的图像)
        param lower: The first range of the threshold(阈值的第一个区间)
        param upper: The second range of the threshold(阈值的第二个区间)
        param min_area: The minimum area of the contour(轮廓的最小面积)
        param max_area: (The maximum area of the contour(轮廓的最大面积)
        '''
        dilated = self.get_lab_binary(lab_image, lower, upper)
        max_contour, contour_area = self.get_binary_contour(dilated, min_area, max_area) 
        return max_contour, contour_area

    def cal_line_angle(self, point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        return int(math.degrees(math.atan2(dy, dx)))

    def cal_two_lines_angle(self, line1_point1, line1_point2, line2_point1, line2_point2):
        if line1_point2[0] == line1_point1[0]:
             slope1 = float('inf')
        else:
            # Calculate the slope of line A(计算直线A的斜率)
            slope1 = (line1_point2[1] - line1_point1[1]) / (line1_point2[0] - line1_point1[0])

        if line2_point2[0] == line2_point1[0]:
             slope2 = float('inf')
        else:
            # Calculate the slope of line B(计算直线B的斜率)
            slope2 = (line2_point2[1] - line2_point1[1]) / (line2_point2[0] - line2_point1[0])

        # If the slopes of lines A and B  are equal, the angle between them is 0 or 180 degrees, depending on their positions(如果直线A和直线B的斜率相等，则夹角为0或180度，具体取决于它们的位置)
        if slope1 == slope2:
            if line2_point1[0] < line1_point1[0] < line2_point2[0] or line2_point2[0] < line1_point1[0] < line2_point1[0]:
                angle = 0
            else:
                angle = 180
        else:
            # Otherwise, use atan2 function to calculate the angle between lines A and B(否则，使用atan2函数计算夹角)
            angle = math.degrees(math.atan2(slope2 - slope1, 1 + slope1 * slope2))

        return angle

    def find_intersection(self, k, x1, y1, x2, y2, x3, y3):
        """
        Given the coordinates of two endpoints (x1, y1) and (x2, y2) of line A, and the slope k, and line B passes through point (x3, y3) with the same slope k(已知直线A的两个端点坐标(x1, y1)和(x2, y2)，以及斜率k，直线B过点(x3, y3)，斜率也是k)
        A perpendicular line is drawn from point (x1, y1) that intersects with line B at point (x4, y4), and another perpendicular line is drawn from point (x2, y2) that intersects with line B at point (x5, y5)(从点(x1, y1)做一条垂线，与B相交于点(x4, y4)，从(x2, y2)做一条垂线，与B相交于点(x5, y5))
        """
        # Slope and intercept of line B(直线B的斜率和截距)
        if k != 0:
            b_slope = k
            b_intercept = y3 - k * x3

            # Slope and intercept of perpendicular line from point (x1, y1)(从点(x1, y1)做的垂线斜率和截距)
            v1_slope = -1 / k
            v1_intercept = y1 - v1_slope * x1

            # Slope and intercept of perpendicular line from point (x2, y2)(从点(x2, y2)做的垂线斜率和截距)
            v2_slope = -1 / k
            v2_intercept = y2 - v2_slope * x2

            # Calculate intersection point(求交点)
            x4 = (v1_intercept - b_intercept) / (b_slope - v1_slope)
            y4 = b_slope * x4 + b_intercept

            x5 = (v2_intercept - b_intercept) / (b_slope - v2_slope)
            y5 = b_slope * x5 + b_intercept
        else:
            x4, y4 = x1, y3
            x5, y5 = x2, y3

        return [int(x4), int(y4)], [int(x5), int(y5)]

    def detect_side(self, contour, roi, img_proc_w, img_proc_h, img_w, img_h):
        '''
        Detect the bottom edge of a rectangle(检测矩形最靠下的边)
        param contour: Input contour(输入轮廓)
        param roi: ROI region(roi区域)
        param img_w: Original image width(原图width)
        param img_h: Original image height(原图height)
        '''
        x, y, angle, corners = self.detect_rect(contour, roi, img_proc_w, img_proc_h, img_w, img_h)        
        
        # Use the four corners of the rectangle as fitting points(将矩形的四个角点作为拟合点)
        pts = np.array([corners[0], corners[1], corners[2], corners[3]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        # (Fit the rectangle to a line using cv2.fitLine() function(使用cv2.fitLine()函数将矩形拟合成一条线)
        [cosx, sinx, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        
        k = math.atan2(sinx, cosx)
        # y = k(x - x0) + y0

        # Calculate the two endpoints of the line segment(计算线段的两个端点)
        min_left_x = (contour[contour[:, :, 0].argmin()][0])[0]
        max_right_x = (contour[contour[:, :, 0].argmax()][0])[0]
        
        min_left_x = self.value_remapped(min_left_x + roi.x_min, 0, img_proc_w, 0, img_w)
        max_right_x = self.value_remapped(max_right_x + roi.x_min, 0, img_proc_w, 0, img_w)

        min_left_y = int(k*(min_left_x - x0) + y0)
        max_right_y = int(k*(max_right_x - x0) + y0)
        
        # Calculate the endpoints of the rectangle's bottom edge(计算矩形最靠下的边的端点)
        max_down_x, max_down_y = contour[contour[:, :, 1].argmax()][0]
        max_down_x, max_down_y = self.point_remapped([max_down_x + roi.x_min, max_down_y + roi.y_min], [img_proc_w, img_proc_h], [img_w, img_h], data_type=int)
        
        [x3, y3], [x4, y4] = self.find_intersection(k, min_left_x, min_left_y, max_right_x, max_right_y, max_down_x, max_down_y)
        
        return x3, y3, x4, y4, int(math.degrees(math.atan(k)))

    def detect_intersection(self, binary, contour, roi, min_area, max_area, img_proc_w, img_proc_h, img_w, img_h):
        '''
        Detect horizontal line(检测横线)
        '''
        # Get the leftmost and rightmost points of the contour(获取轮廓最左，最右)
        min_left_x, min_left_y = contour[contour[:, :, 0].argmin()][0]
        max_right_x, max_right_y = contour[contour[:, :, 0].argmax()][0]
        
        # Crop the region that is 2/3 away from the left and right points(截取离左右点2/3的区域)
        center_x, center_y = int((min_left_x + max_right_x) / 2), int((min_left_y + max_right_y) / 2)
        
        half_center_x1, half_center_y1 = int(min_left_x + 2*(center_x - min_left_x) / 3), int((min_left_y + center_y) / 2)

        half_center_x2, half_center_y2 = int(center_x + (max_right_x - center_x) / 3), int((max_right_y + center_y) / 2)
        
        left_y1 = min(min_left_y, half_center_y1) - 10
        right_y1 = max(min_left_y, half_center_y1) + 10
        left_y2 = min(max_right_y, half_center_y2) - 10
        right_y2 = max(max_right_y, half_center_y2) + 10
        if left_y1 < 0:
            left_y1 = 0
        if right_y1 > img_proc_h:
            right_y1 = img_proc_h
        if left_y2 < 0:
            left_y2 = 0
        if right_y2 > img_proc_h:
            right_y2 = img_proc_h
        
        roi1 = binary[left_y1:right_y1, min(min_left_x, half_center_x1):max(min_left_x, half_center_x1)]
        roi2 = binary[left_y2:right_y2, min(max_right_x, half_center_x2):max(max_right_x, half_center_x2)]
    
        max_contour1, contour_area1 = self.get_binary_contour(roi1, min_area, max_area)
        max_contour2, contour_area2 = self.get_binary_contour(roi2, min_area, max_area)
        if max_contour1 is not None and max_contour2 is not None:
            # Identify the bottom line(识别最靠下的线)
            roi_1 = copy.deepcopy(roi)
            roi_1.y_min = left_y1 + roi.y_min
            roi_1.y_max = right_y1 + roi.y_min
            roi_1.x_min = min(min_left_x, half_center_x1) + roi.x_min
            roi_1.x_max = max(min_left_x, half_center_x1) + roi.x_min
            
            x1, y1 = self.detect_side(max_contour1, roi_1, img_proc_w, img_proc_h, img_w, img_h)[:2]
            
            roi_2 = copy.deepcopy(roi)
            roi_2.y_min = left_y2 + roi.y_min
            roi_2.y_max = right_y2 + roi.y_min
            roi_2.x_min = min(max_right_x, half_center_x2) + roi.x_min
            roi_2.x_max = max(max_right_x, half_center_x2) + roi.x_min
            x2, y2 = self.detect_side(max_contour2, roi_2, img_proc_w, img_proc_h, img_w, img_h)[2:4]
           
            # Count the number of white pixels(统计白色像素数量)
            white_pixels = cv2.countNonZero(binary)
            
            # Calculate the proportion of white pixels(计算白色像素所占比例)
            total_pixels = binary.shape[0] * binary.shape[1]
            white_ratio_old = white_pixels / total_pixels
            
            x_1, y_1 = self.point_remapped([x1, y1], [img_w, img_h], [img_proc_w, img_proc_h], data_type=int)
            x_2, y_2 = self.point_remapped([x2, y2], [img_w, img_h], [img_proc_w, img_proc_h], data_type=int)

            cv2.line(binary, (x_1 - roi.x_min, y_1 - roi.y_min), (x_2 - roi.x_min, y_2 - roi.y_min), (0, 0, 0), 5)
            # Count the number of white pixels(统计白色像素数量)
            white_pixels = cv2.countNonZero(binary)

            # Calculate the proportion of white pixels(计算白色像素所占比例)
            white_ratio_new = white_pixels / total_pixels
            # print(white_ratio_old - white_ratio_new)
            if white_ratio_old - white_ratio_new > 0.02:
                return x1, y1, x2, y2, self.cal_line_angle([x1, y1], [x2, y2])
            else:
                return False
        else:
            return False

    def detect_cross(self, binary, contour, roi, min_area, max_area, img_proc_w, img_proc_h, img_w, img_h, angle_threshold=20):
        '''
        Detect intersection(检测十字路口)
        '''

        min_up_x, min_up_y = contour[contour[:, :, 1].argmin()][0]
        result = self.detect_intersection(binary, contour, roi, min_area, max_area, img_proc_w, img_proc_h, img_w, img_h)
        if result:
            # 计anglee = result
            x1, y1, x2, y2, angle = result
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            angle = self.cal_two_lines_angle([x1, y1], [x2, y2], [min_up_x, min_up_y], [center_x, center_y])
            
            if abs(abs(angle) - 89) < angle_threshold:
                return x1, y1, x2, y2, self.cal_line_angle([x1, y1], [x2, y2])
            else:
                return False
        else:
            return False

    def detect_rect(self, contour, roi, img_proc_w, img_proc_h, img_w, img_h):
        '''
        Obtain the center point and the corners of the contour's minimum bounding rectangle(获取轮廓的最小外接矩形的中心和角点)
        param contour: Input contour(输入轮廓)
        param roi: ROI region(roi区域)
        param img_w: Original image width(原图width)
        param img_h: Original image height(原图height)
        '''
        rect = cv2.minAreaRect(contour)  # Obtain the minimum bounding rectangle(获取最小外接矩形)
        corners = np.int64(cv2.boxPoints(rect))  # Obtain the four corner points of the minimum bounding rectangle(获取最小外接矩形的四个角点)
        for j in range(4):
            corners[j, 0], corners[j, 1] = self.point_remapped([corners[j, 0] + roi.x_min, corners[j, 1] + roi.y_min],
                                                               [img_proc_w, img_proc_h], [img_w, img_h],
                                                               data_type=int)  # Map the point to the original image size(点映射到原图大小)
        x, y = self.point_remapped([rect[0][0] + roi.x_min, rect[0][1] + roi.y_min], [img_proc_w, img_proc_h],
                                   [img_w, img_h], data_type=int)  # Map the point to the original image size(点映射到原图大小)
        return x, y, int(rect[2]), corners

    def detect_circle(self, contour, roi, img_proc_w, img_proc_h, img_w, img_h):
        '''
        Obtain the center point and the radius of the minimum enclosing circle(获取最小外接圆的中心和半径)
        param contour: Input contour(输入轮廓)
        param roi: ROI region(roi区域)
        param img_w: Original image width(原图width)
        param img_h: Original image height(原图height)
        '''
        ((x, y), radius) = cv2.minEnclosingCircle(contour)  # Obtain the minimum enclosing circle(获取最小外接圆)
        x, y = self.point_remapped([x + roi.x_min, y + roi.y_min], [img_proc_w, img_proc_h], [img_w, img_h],
                                   data_type=int)  # Map the point to the original image size(点映射到原图大小)
        radius = self.value_remapped(radius, 0, img_proc_w, 0, img_w, data_type=int)
        
        return x, y, radius
    
    def cal_mean_bgr(self, bgr_image, x, y):
        h, w = bgr_image.shape[:2]
        if y == 0:
           y = 2
        if y == h:
            y = h - 2
        if x == 0:
            x = 2
        if x == w:
            x = w - 2
        image = bgr_image[y - 2:y + 2, x - 2:x + 2]
        mean_b = int(np.mean(image[:,:,0]))
        mean_g = int(np.mean(image[:,:,1]))
        mean_r = int(np.mean(image[:,:,2]))

        return (mean_b, mean_g, mean_r)

    def detect(self, bgr_image):
        '''
        Color detection(颜色检测)
        :param image: The original image to perform color detection on, in BGR format which is OpenCV format(要进行颜色检测的原图像，格式为bgr，即opencv格式)
        :return: Return the original image and the information of the detected object(返回原图像和检测的物体的信息)
        '''
        try:
            img_h, img_w = bgr_image.shape[:2]  # Get the size of the original image(获取原图大小)
            image_draw = bgr_image.copy() 
            
            image_gb = cv2.GaussianBlur(bgr_image, (3, 3), 3)  # Gaussian blur to remove noise(高斯模糊去噪点)

            # Object information list: color, position, size, and angle(物体信息列表:颜色, 位置, 大小, 角度)
            object_info_list = []
            center_x = 0
            center_y = 0
            last_process_size = [0, 0]
            for color_info in self.detect_info:  # Traverse through the color list(遍历颜色列表)
                self.image_process_size = color_info.image_process_size
                if self.image_process_size != last_process_size:
                    image_resize = cv2.resize(image_gb, tuple(self.image_process_size), interpolation=cv2.INTER_NEAREST)  # Image scaling to speed up image processing. Cannot be too small, otherwise a lot of details will be lost(图像缩放, 加快图像处理速度, 不能太小，否则会丢失大量细节)
                    image_lab = cv2.cvtColor(image_resize, cv2.COLOR_BGR2LAB)  # Convert from BGR color space to LAB color space for easier color extraction(bgr空间转lab空间，方便提取颜色)
                
                last_process_size = self.image_process_size
                if color_info.use_name:
                    if color_info.color_name in self.lab_data:  # If there is color to be recognized in LAB(如果要识别的颜色在lab里有)
                        lower = tuple(self.lab_data[color_info.color_name]['min'])
                        upper = tuple(self.lab_data[color_info.color_name]['max'])
                    else:
                        continue
                else:
                    lower = tuple(color_info.lab_min)
                    upper = tuple(color_info.lab_max)

                if color_info.detect_type == 'line':  # Line following detection(巡线检测)
                    line_roi = [color_info.line_roi.up, color_info.line_roi.center, color_info.line_roi.down]
                    for roi in line_roi:
                        if self.debug:
                            x_min, y_min = self.point_remapped([roi.x_min, roi.y_min], self.image_process_size, [img_w, img_h],data_type=int)
                            x_max, y_max = self.point_remapped([roi.x_max, roi.y_max], self.image_process_size, [img_w, img_h],data_type=int)
                            cv2.rectangle(image_draw, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
                        blob = image_lab[roi.y_min:roi.y_max, roi.x_min:roi.x_max]  # 截取roi(crop ROI)
                        max_contour, contour_area = self.get_lab_contour(blob, lower, upper, color_info.min_area, color_info.max_area)
                        if max_contour is not None:
                            x, y, angle, corners = self.detect_rect(max_contour, roi, self.image_process_size[0], self.image_process_size[1], img_w, img_h)
                            # draw_color = self.cal_mean_bgr(image_draw, x, y)
                            cv2.circle(image_draw, (x, y), 3, (0, 255, 255), -1)
                            cv2.drawContours(image_draw, [corners], -1, (0, 255, 255), 3, cv2.LINE_AA)  # Draw the contour of the rectangle(绘制矩形轮廓)
                            center_x = x
                            center_y = y
                            break

                    # Color, position, size, and angle(颜色, 位置, 大小, 角度)
                    if center_x != 0:
                        object_info_list.extend([[color_info.color_name, color_info.detect_type, [center_x, center_y], [img_w, img_h], 0, angle, [], []]])
                        draw_color = self.cal_mean_bgr(image_draw, center_x, center_y)
                        cv2.circle(image_draw, (center_x, center_y), 3, draw_color, -1)  # Draw the center point(画出中心点)
                else:
                    roi = color_info.roi
                    if roi.y_min == roi.y_max:
                        blob = image_lab
                    else:
                        blob = image_lab[roi.y_min:roi.y_max, roi.x_min:roi.x_max]
                    if self.debug:
                        x_min, y_min = self.point_remapped([roi.x_min, roi.y_min], self.image_process_size, [img_w, img_h], data_type=int)
                        x_max, y_max = self.point_remapped([roi.x_max, roi.y_max], self.image_process_size, [img_w, img_h], data_type=int)
                        # draw_color = self.cal_mean_bgr(image_draw, int((x_min+x_max)/2), int((y_min+y_max)/2))
                        cv2.rectangle(image_draw, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
                    if color_info.detect_type not in ['cross', 'intersection']:
                        max_contour, contour_area = self.get_lab_contour(blob, lower, upper, color_info.min_area, color_info.max_area)
                        if max_contour is not None:
                            if color_info.detect_type == 'rect':
                                x, y, angle, corners = self.detect_rect(max_contour, roi, self.image_process_size[0], self.image_process_size[1], img_w, img_h)
                                # draw_color = self.cal_mean_bgr(image_draw, x, y)
                                cv2.drawContours(image_draw, [corners], -1, (0, 255, 255), 3, cv2.LINE_AA)  # Draw the contour of the rectangle(绘制矩形轮廓)
                                object_info_list.extend([[color_info.color_name, color_info.detect_type, [x, y], [img_w, img_h], 0, angle, [], []]])
                            elif color_info.detect_type == 'side':
                                x1, y1, x2, y2, angle = self.detect_side(max_contour, roi, self.image_process_size[0], self.image_process_size[1], img_w, img_h)
                                x, y = int((x1 + x2)/2), int((y1 + y2)/2)
                                # draw_color = self.cal_mean_bgr(image_draw, x, y)
                                cv2.line(image_draw, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                object_info_list.extend([[color_info.color_name, color_info.detect_type, [x, y], [img_w, img_h], 0, angle, [x1, y1], [x2, y2]]])
                            elif color_info.detect_type == 'circle':
                                x, y, radius = self.detect_circle(max_contour, roi, self.image_process_size[0], self.image_process_size[1], img_w, img_h)
                                draw_color = self.cal_mean_bgr(image_draw, x, y)
                                cv2.circle(image_draw, (x, y), radius, draw_color, 3)  # Draw circle(画圆)
                                object_info_list.extend([[color_info.color_name, color_info.detect_type, [x, y], [img_w, img_h], radius, 0, [], []]])
                    else:
                        binary = self.get_lab_binary(blob, lower, upper)
                        max_contour, contour_area = self.get_binary_contour(binary, color_info.min_area, color_info.max_area)
                        if max_contour is not None:
                            if color_info.detect_type == 'cross':
                                result = self.detect_cross(binary, max_contour, roi, color_info.min_area, color_info.max_area, self.image_process_size[0], self.image_process_size[1], img_w, img_h)
                                if result:
                                    x1, y1, x2, y2, angle = result
                                    x, y = int((x1 + x2)/2), int((y1 + y2)/2)
                                    # draw_color = self.cal_mean_bgr(image_draw, x, y) 
                                    cv2.line(image_draw, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                    object_info_list.extend([[color_info.color_name, color_info.detect_type, [x, y], [img_w, img_h], 0, angle, [x1, y1], [x2, y2]]])
                            elif color_info.detect_type == 'intersection':
                                result = self.detect_intersection(binary, max_contour, roi, color_info.min_area, color_info.max_area, self.image_process_size[0], self.image_process_size[1], img_w, img_h)
                                if result:
                                    x1, y1, x2, y2, angle = result
                                    x, y = int((x1 + x2)/2), int((y1 + y2)/2)
                                    # draw_color = self.cal_mean_bgr(image_draw, x, y) 
                                    cv2.line(image_draw, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                    object_info_list.extend([[color_info.color_name, color_info.detect_type, [x, y], [img_w, img_h], 0, angle, [x1, y1], [x2, y2]]])

            return image_draw, object_info_list  # Return the original image and the object information(返回原图像和物体的信息)
        except BaseException as e:
            print('color detect error:', e)
            return image_draw, [] 

if __name__ == '__main__':
    import time
    from rosman_sdk import common
    from rosman_kinematics.motion_manager import MotionManager

    class ROI:
        y_min = 0
        y_max = 120
        x_min = 0
        x_max = 160
    
    class LineROI:
        up = ROI()
        center = ROI()
        down = ROI()
    
    class ColorDetect:
        color_name = ''
        use_name = True
        detect_type = 'rect'
        image_process_size = [160, 120]
        roi = ROI()
        line_roi = LineROI()
        min_area = 10
        max_area = 160*120
        def __init__(self, color_name):
            self.color_name = color_name
    
    class ColorsDetect:
        data = [ColorDetect('red'), ColorDetect('green'), ColorDetect('blue')]
    
    roi = [ # [ROI, weight]
        (20, 30, 40, 120), 
        (40, 50, 40, 120), 
        (60, 70, 40, 120)  
        ]
    line = ColorDetect('red')
    line.detect_type = 'line'
    line.line_roi.up.y_min = roi[0][0]
    line.line_roi.up.y_max = roi[0][1]
    line.line_roi.up.x_min = roi[0][2]
    line.line_roi.up.x_max = roi[0][3]

    line.line_roi.center.y_min = roi[1][0]
    line.line_roi.center.y_max = roi[1][1]
    line.line_roi.center.x_min = roi[1][2]
    line.line_roi.center.x_max = roi[1][3]

    line.line_roi.down.y_min = roi[2][0]
    line.line_roi.down.y_max = roi[2][1]
    line.line_roi.down.x_min = roi[2][2]
    line.line_roi.down.x_max = roi[2][3]
    data = ColorsDetect()
    data.data = [line]

    lab_config = common.get_yaml_data('/home/ubuntu/software/lab_tool/lab_config.yaml')
    # color_detection = ColorDetection(lab_config['lab']['Mono'], data)
    color_detection = ColorDetection(lab_config['lab']['Mono'], ColorsDetect())
    
    # motion_manager = MotionManager()
    # motion_manager.run_action('walk_ready')
    # motion_manager.set_servos_position(200, [[23, 500], [24, 260]])
    cap = cv2.VideoCapture(-1)
    while True:
        ret, frame = cap.read()
        if ret:
            img = color_detection.detect(frame)[0]
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key != -1:
                break
        else:
            time.sleep(0.01)
    cap.release()

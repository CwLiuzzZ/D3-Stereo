import numpy as np
import torch
import cv2

class evaluator_BA_dfm_disp():
    def __init__(self):
        # self.args = args
        self.gt_mask_list = []
        self.generated_mask_list = []
        self.gray_image_list = []

        self.gt_disp_group_list = []
        self.pre_disp_group_list = []
        self.color_divide = [(0,51),(52,102),(103,153),(154,204),(205,255)]

        # area of valid pixels
        # number of valid pixels
    
    # input: (batch_size, C, H, W) or (batch_size, H, W)
    # input and accumulate the gt_mask
    def input_gt_mask(self, gt_mask):
        self.gt_mask_list.append(gt_mask.numpy().flatten())

    # input and accumulate the generated_mask
    def input_data(self, generated_mask):
        self.generated_mask_list.append(generated_mask.numpy().flatten())
    
    def input_ori_image(self, image):
        self.gray_image_list.append(image.flatten())

    # process the list and output the metrics
    def process(self,):
        '''
        process the accumulated list to tensor [number of images, ...] 
        and get number of images
        '''
        
        self.image_number = len(self.gt_mask_list) 
        self.bilateral_average()

        value_dic_img,value_dic_mean = self.get_value()
        return value_dic_img,value_dic_mean
    
    def bilateral_average(self):

        for index in range(self.image_number):
            ind_group = []
            disp_group_gt = []
            disp_group_pre = []
            disp = self.gt_mask_list[index]
            gray = self.gray_image_list[index]
            pre_disp = self.generated_mask_list[index]

            ind = np.arange(gray.shape[0])

            max_disp = np.max(disp)
            min_disp = np.min(disp)
            dis_disp = (max_disp-min_disp+1)/5
            disp_divide = []
            for i in range(5):
                disp_divide.append((min_disp+i*dis_disp,min_disp+(1+i)*dis_disp))
            for gray_zone in self.color_divide:
                mask = np.logical_and(gray>=gray_zone[0],gray<=gray_zone[1])
                disp_ = disp[mask]
                ind_ = ind[mask]
                # print(disp_)
                for disp_zone in disp_divide:
                    mask = np.logical_and(disp_>=disp_zone[0],disp_<disp_zone[1])
                    # print(disp_[mask])
                    disp_group_gt.append(disp_[mask])
                    ind_group.append(ind_[mask])
            for i in ind_group:
                disp_group_pre.append(pre_disp[i])
            
            self.gt_disp_group_list.append(disp_group_gt)
            self.pre_disp_group_list.append(disp_group_pre)

    def get_value(self):
        value_dic_img = {}
        value_dic_mean = {}
        value_dic_img['valid_pixel_ratio'] = self.valid_pixel_ratio()

        value_dic_img['D1_all'] = self.D1_all()
        value_dic_img['1_error'] = self.D1_all_thr(1)
        value_dic_img['2_error'] = self.D1_all_thr(2)
        value_dic_img['3_error'] = self.D1_all_thr(3)
        value_dic_img['epe'] = self.epe()
        
        for key in value_dic_img.keys():
            value_dic_mean[key] = round(np.mean(value_dic_img[key]),6)
        
        return value_dic_img,value_dic_mean
    
    # absolute relative error
    def valid_pixel_ratio(self):
        error_list = []
        for index in range(self.image_number):
            input_pre = self.generated_mask_list[index]
            pixel_all = input_pre.shape[0]
            mask = np.zeros(shape = input_pre.shape)
            mask[input_pre>0]=1
            pixel_valid = np.sum(mask)
            error_list.append(pixel_valid/pixel_all)
        return error_list 
    
    # end point error
    def epe(self):
        error_list = []
        for index in range(self.image_number):
            error_list_group = []
            disp_group_gt = self.gt_disp_group_list[index]
            disp_group_pre = self.pre_disp_group_list[index]
            for i in range(len(disp_group_gt)):
                disp_gt = disp_group_gt[i]
                disp_pre = disp_group_pre[i]
                if disp_gt.shape[0] == 0 or np.sum(disp_pre) == 0:
                    continue
                mask = np.zeros(shape = disp_pre.shape)
                mask[disp_pre>0]=1       
                mask[disp_gt==0]=0
                disp_pre = disp_pre[mask==1]
                disp_gt = disp_gt[mask==1]
                error = np.sum(np.abs(disp_gt - disp_pre))/np.sum(mask)
                error_list_group.append(error)
            error_list.append(round(np.mean(error_list_group),6))
        return error_list   

    def D1_all(self):
        error_list = []
        for index in range(self.image_number):
            error_list_group = []
            disp_group_gt = self.gt_disp_group_list[index]
            disp_group_pre = self.pre_disp_group_list[index]
            for i in range(len(disp_group_gt)):
                input_gt = disp_group_gt[i]
                input_pre = disp_group_pre[i]
                if input_gt.shape[0] == 0 or np.sum(input_pre) == 0:
                    continue
                mask = np.zeros(shape = input_pre.shape)
                mask[input_pre>0]=1
                mask[input_gt==0]=0
                input_pre = input_pre[mask==1]
                input_gt = input_gt[mask==1]
                variable1 = abs(input_gt - input_pre)
                variable2 = np.ones(input_pre.shape)
                variable2[variable1 < input_gt*0.05] = 0
                variable2[variable1 < 3] = 0
                error = np.sum(variable2)/np.sum(mask)
                error_list_group.append(error)
            error_list.append(round(np.mean(error_list_group),6))
        return error_list   

    def D1_all_thr(self,thr=3):
        error_list = []
        for index in range(self.image_number):
            error_list_group = []
            disp_group_gt = self.gt_disp_group_list[index]
            disp_group_pre = self.pre_disp_group_list[index]
            for i in range(len(disp_group_gt)):
                input_gt = disp_group_gt[i]
                input_pre = disp_group_pre[i]
                if input_gt.shape[0] == 0 or np.sum(input_pre) == 0:
                    continue
                mask = np.zeros(shape = input_pre.shape)
                mask[input_pre>0]=1
                mask[input_gt==0]=0
                input_pre = input_pre[mask==1]
                input_gt = input_gt[mask==1]
                variable1 = abs(input_gt - input_pre)
                variable2 = np.zeros(input_pre.shape)
                # variable2[variable1 > input_gt*0.05] = 1
                variable2[variable1 > thr] = 1
                error = np.sum(variable2)/np.sum(mask)
                error_list_group.append(error)
            error_list.append(round(np.mean(error_list_group),6))
        return error_list   

class evaluator_dfm_disp():
    def __init__(self,pre_mask=False):
        # self.args = args
        self.gt_mask_list = []
        self.generated_mask_list = []
        # area of valid pixels
        self.valid_mask_list = []
        # number of valid pixels
        self.valid_pixels_list = []
        self.pre_mask = pre_mask
    
    # input: (batch_size, C, H, W) or (batch_size, H, W)
    # input and accumulate the gt_mask
    def input_gt_mask(self, gt_mask):
        # check_zero = torch.zeros(gt_mask.shape)
        # check_zero[gt_mask==0] = 1
        # if torch.sum(check_zero)>0:
        #     print('zero exists in input_data, such pixel will be ignored')
        self.gt_mask_list.append(gt_mask)

    # input and accumulate the generated_mask
    def input_data(self, generated_mask):
        # check_zero = 1.0/generated_mask
        # if torch.isinf(check_zero).any():
        #     print(('zero exists in input_data'))
        #     raise ZeroDivisionError('zero exists in input_data')
        self.generated_mask_list.append(generated_mask)

    # process the list and output the metrics
    def process(self,if_ignore_zero=True):
        '''
        process the accumulated list to tensor [number of images, ...] 
        and get number of images
        '''
        
        self.image_number = len(self.gt_mask_list)        
        value_dic_img,value_dic_mean = self.get_value()
        return value_dic_img,value_dic_mean
    
    def get_value(self):
        value_dic_img = {}
        value_dic_mean = {}
        value_dic_img['valid_pixel_ratio'] = self.valid_pixel_ratio()

        value_dic_img['D1_all'] = self.D1_all()
        value_dic_img['1_error'] = self.D1_all_thr(1)
        value_dic_img['2_error'] = self.D1_all_thr(2)
        value_dic_img['3_error'] = self.D1_all_thr(3)
        value_dic_img['epe'] = self.epe()
        
        for key in value_dic_img.keys():
            value_dic_mean[key] = round(np.mean(value_dic_img[key]),6)
        
        return value_dic_img,value_dic_mean
    
    # absolute relative error
    def valid_pixel_ratio(self):
        error_list = []
        for index in range(self.image_number):
            input_pre = self.generated_mask_list[index]
            pixel_all = input_pre.shape[0]*input_pre.shape[1]
            mask = np.zeros(shape = input_pre.shape)
            mask[input_pre>0]=1
            pixel_valid = np.sum(mask)
            error_list.append(pixel_valid/pixel_all)
        return error_list 

    # # absolute relative error
    # def abs_rel(self):
    #     error_list = []
    #     for index in range(self.image_number):
    #         input_gt = self.gt_mask_list[index]
    #         input_pre = self.generated_mask_list[index]
    #         variable = self.devide_zero(input_gt - input_pre,input_gt)
    #         error = torch.sum(abs(variable))/self.valid_pixels_list[index]
    #         error_list.append(round(error.item(),6))
    #     return error_list   
    
    # # square relative error
    # def Sq_Rel(self):
    #     error_list = []
    #     for index in range(self.image_number):
    #         input_gt = self.gt_mask_list[index]
    #         input_pre = self.generated_mask_list[index]
    #         variable = self.devide_zero(input_gt - input_pre,input_gt)
    #         error = torch.sum(variable*variable)/self.valid_pixels_list[index]
    #         error_list.append(round(error.item(),6))
    #     return error_list   
    
    # # mean absolute error
    # def MAE(self):
    #     error_list = []
    #     for index in range(self.image_number):
    #         input_gt = self.gt_mask_list[index]
    #         input_pre = self.generated_mask_list[index]
    #         error = torch.sum(abs(input_gt - input_pre))/self.valid_pixels_list[index]
    #         error_list.append(round(error.item(),6))
    #     return error_list   
    
    # # root mean square error
    # def RMSE(self):
    #     error_list = []
    #     for index in range(self.image_number):
    #         input_gt = self.gt_mask_list[index]
    #         input_pre = self.generated_mask_list[index]
    #         variable = input_gt - input_pre
    #         error = torch.sqrt(torch.sum(variable * variable)/self.valid_pixels_list[index])
    #         error_list.append(round(error.item(),6))
    #     return error_list   
    
    # # root mean square logarithmic error
    # def RMSE_log(self):
    #     error_list = []
    #     for index in range(self.image_number):
    #         input_gt = self.gt_mask_list[index]
    #         input_pre = self.generated_mask_list[index]
    #         variable1 = torch.log(input_gt)
    #         variable2= torch.log(input_pre)
    #         # replace inf with zero
    #         variable1 = torch.where(torch.isinf(variable1), torch.full_like(variable1, 0), variable1)
    #         variable2 = torch.where(torch.isinf(variable2), torch.full_like(variable2, 0), variable2)
    #         variable = variable1 - variable2
    #         error = torch.sqrt(torch.sum(variable * variable)/self.valid_pixels_list[index])
    #         error_list.append(round(error.item(),6))
    #     return error_list   
    
    # def pixels_deviation_rate(self, deviation):
    #     error_list = []
    #     for index in range(self.image_number):
    #         input_gt = self.gt_mask_list[index]
    #         input_pre = self.generated_mask_list[index]
    #         variable1 = torch.maximum(input_gt/input_pre,self.devide_zero(input_pre,input_gt))
    #         variable2 = torch.zeros(input_pre.shape)
    #         variable2[variable1<deviation] = 1
    #         # erase invalid area
    #         variable2 = variable2 * self.valid_mask_list[index]
    #         error = torch.sum(variable2)/self.valid_pixels_list[index]
    #         error_list.append(round(error.item(),6))
    #     return error_list   

    # end point error
    def epe(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]

            mask = np.zeros(shape = input_pre.shape)
            mask[input_pre>0]=1
            mask[input_gt==0]=0
            input_pre = input_pre[mask==1]
            input_gt = input_gt[mask==1]

            error = torch.sum(abs(input_gt - input_pre))/np.sum(mask)
            error_list.append(round(error.item(),6))
        return error_list   

    def D1_all(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]

            mask = np.zeros(shape = input_pre.shape)
            mask[input_pre>0]=1
            mask[input_gt==0]=0
            input_pre = input_pre[mask==1]
            input_gt = input_gt[mask==1]

            variable1 = abs(input_gt - input_pre)
            variable2 = torch.ones(input_pre.shape)
            variable2[variable1 < input_gt*0.05] = 0
            variable2[variable1 < 3] = 0

            # cv2.imwrite('valid_mask.png',(self.valid_mask_list[index]*255).numpy())
            # cv2.imwrite('bad_pixels.png',(variable2*255).numpy())
            error = torch.sum(variable2)/np.sum(mask)
            error_list.append(round(error.item(),6))
        return error_list   

    def D1_all_thr(self,thr=3):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]

            mask = np.zeros(shape = input_pre.shape)
            mask[input_pre>0]=1
            mask[input_gt==0]=0
            input_pre = input_pre[mask==1]
            input_gt = input_gt[mask==1]

            error_list.append(round(error.item(),6))
            variable1 = abs(input_gt - input_pre)
            variable2 = torch.zeros(input_pre.shape)
            # variable2[variable1 > input_gt*0.05] = 1
            variable2[variable1 > thr] = 1

            # cv2.imwrite('valid_mask.png',(self.valid_mask_list[index]*255).numpy())
            # cv2.imwrite('bad_pixels.png',(variable2*255).numpy())
            error = torch.sum(variable2)/np.sum(mask)
            error_list.append(round(error.item(),6))
        return error_list   
    
    # # mean absolute logarithmic error
    # def log_MAE(self, input_gt, input_pre):
    #     variable1 = torch.log(input_gt)
    #     variable2= torch.log(input_pre)
    #     # replace inf with zero
    #     variable1 = torch.where(torch.isinf(variable1), torch.full_like(variable1, 0), variable1)
    #     variable2 = torch.where(torch.isinf(variable2), torch.full_like(variable2, 0), variable2)
    #     variable = variable1 - variable2
    #     # replace nan with zero
    #     variable = torch.where(torch.isnan(variable), torch.full_like(variable, 0), variable)
    #     error = torch.sum(abs(variable))/self.valid_pixels
    #     return error  
    
    # # inverse mean absolute error
    # def i_MAE(self, input_gt, input_pre):
    #     variable1 = 1.0/input_gt
    #     variable2 = 1.0/input_pre
    #     # replace inf with zero
    #     variable1 = torch.where(torch.isinf(variable1), torch.full_like(variable1, 0), variable1)
    #     variable2 = torch.where(torch.isinf(variable2), torch.full_like(variable2, 0), variable2)
    #     error = torch.sum(abs(variable1 - variable2))/self.valid_pixels
    #     return error     
   
    
    # # inverse root mean square error
    # def iRMSE(self, input_gt, input_pre):
    #     variable1 = 1.0/input_gt
    #     variable2 = 1.0/input_pre
    #     # replace inf with zero
    #     variable1 = torch.where(torch.isinf(variable1), torch.full_like(variable1, 0), variable1)
    #     variable2 = torch.where(torch.isinf(variable2), torch.full_like(variable2, 0), variable2)
    #     variable = variable1 - variable2
    #     error = torch.sqrt(torch.sum(variable * variable)/self.valid_pixels)
    #     return error  
    
    # devide and process the nan item in the ans
    def devide_zero(self,x,y):
        variable = x/y
        variable = torch.where(torch.isnan(variable), torch.full_like(variable, 0), variable)
        return variable


class evaluator_disp():
    def __init__(self,pre_mask=False):
        # self.args = args
        self.gt_mask_list = []
        self.generated_mask_list = []
        self.additional_mask_list = []
        # area of valid pixels
        self.valid_mask_list = []
        # number of valid pixels
        self.valid_pixels_list = []
        self.pre_mask = pre_mask
    
    # input: (batch_size, C, H, W) or (batch_size, H, W)
    # input and accumulate the gt_mask
    def input_gt_mask(self, gt_mask):
        self.gt_mask_list.append(gt_mask)

    # input and accumulate the generated_mask
    def input_data(self, generated_mask):
        self.generated_mask_list.append(generated_mask)
        
    # process the list and output the metrics
    def process(self,if_ignore_zero=True, if_additional_mask=False):
        '''
        process the accumulated list to tensor [number of images, ...] 
        and get number of images
        '''
        
        self.image_number = len(self.gt_mask_list)
        

        for index in range(len(self.gt_mask_list)):
            gt_mask = self.gt_mask_list[index]
            valid_mask = torch.ones(gt_mask.shape)
            valid_mask[gt_mask==0]=0
            self.generated_mask_list[index] = self.generated_mask_list[index]*valid_mask
            self.valid_pixels_list.append(torch.sum(valid_mask))

        # # imply additional masks
        # if if_additional_mask:
        #     additional_masks = torch.cat(self.additional_mask_list)
        #     assert gt_masks.shape[0] == additional_masks.shape[0], 'numbers of gt tensor and additional masks are not equal'
        #     gt_masks = gt_masks * additional_masks
        #     generated_masks = generated_masks * additional_masks
        
        # # ignore the pixels that have gt values zero
        # if if_ignore_zero:
        #     for index in range(len(self.gt_mask_list)):
        #         gt_mask = self.gt_mask_list[index]
        #         valid_mask = torch.ones(gt_mask.shape)
        #         # print(self.gt_mask_list[index].shape)
        #         # right_disp = disparity_transfer(self.gt_mask_list[index])
        #         # 获取长度宽度
        #         H,W = self.gt_mask_list[index].shape
        #         # 构建基础坐标矩阵base，其每个像素的值为其x轴坐标
        #         _,base = torch.meshgrid(torch.from_numpy(np.arange(H)),torch.from_numpy(np.arange(W)))
        #         # target = base - disparity_map，target中像素的值，为左图中每个像素对应的右图像素在右图坐标系下的x坐标
        #         target = -self.gt_mask_list[index] + base
        #         # target中x坐标小于0，表示左图中的像素的对应像素在右图中不可见，就屏蔽掉
        #         valid_mask[target<0]=0
                
                
        #         self.gt_mask_list[index] = self.gt_mask_list[index] * valid_mask
        #         self.generated_mask_list[index] = self.generated_mask_list[index] * valid_mask
        #         self.valid_mask_list.append(valid_mask)
                
        # # count the number of valid pixels
        # if if_additional_mask or if_ignore_zero:
        #     # count pixels with values that are not zero
        #     for valid_mask in self.valid_mask_list:
        #         self.valid_pixels_list.append(torch.sum(valid_mask))
        # else:
        #     # count all the pixels
        #     for index in range(self.image_number):
        #         shape = self.gt_mask_list[index].shape
        #         self.valid_pixels_list.append(shape[0]*shape[1])
        
        value_dic_img,value_dic_mean = self.get_value()
        return value_dic_img,value_dic_mean
    
    def get_value(self):
        value_dic_img = {}
        value_dic_mean = {}
        
        value_dic_img['D1_all'] = self.D1_all()
        value_dic_img['0.5_error'] = self.D1_all_thr(0.5)
        value_dic_img['1_error'] = self.D1_all_thr(1)
        value_dic_img['1.5_error'] = self.D1_all_thr(1.5)
        value_dic_img['2_error'] = self.D1_all_thr(2)
        value_dic_img['3_error'] = self.D1_all_thr(3)
        value_dic_img['epe'] = self.epe()
        value_dic_img['RMS'] = self.RMSE()
        # value_dic_img['A50'] = self.A(50)
        # value_dic_img['A90'] = self.A(90)
        # value_dic_img['A95'] = self.A(95)
        # value_dic_img['A99'] = self.A(99)
        
        for key in value_dic_img.keys():
            value_dic_mean[key] = round(np.mean(value_dic_img[key]),6)
        
        return value_dic_img,value_dic_mean
    
    # absolute relative error
    def abs_rel(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]
            variable = self.devide_zero(input_gt - input_pre,input_gt)
            error = torch.sum(abs(variable))/self.valid_pixels_list[index]
            error_list.append(round(error.item(),6))
        return error_list   
    
    # square relative error
    def Sq_Rel(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]
            variable = self.devide_zero(input_gt - input_pre,input_gt)
            error = torch.sum(variable*variable)/self.valid_pixels_list[index]
            error_list.append(round(error.item(),6))
        return error_list   
    
    # mean absolute error
    def MAE(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]
            error = torch.sum(abs(input_gt - input_pre))/self.valid_pixels_list[index]
            error_list.append(round(error.item(),6))
        return error_list   
    
    # root mean square error
    def RMSE(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]
            variable = input_gt - input_pre
            error = torch.sqrt(torch.sum(torch.square(variable))/self.valid_pixels_list[index])
            error_list.append(round(error.item(),6))
        return error_list   
    
    def A(self,percentage):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index].view(-1)
            input_pre = self.generated_mask_list[index].view(-1)
            input_pre = input_pre[input_gt>0]
            input_gt = input_gt[input_gt>0]
            error = torch.abs(input_gt-input_pre)
            sorted, indices = torch.sort(error) # 在维度1上按照升序排列
            index = int(self.valid_pixels_list[index]*percentage/100)
            error = sorted[index-1]
            error_list.append(round(error.item(),6))
        return error_list   

    # root mean square logarithmic error
    def RMSE_log(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]
            variable1 = torch.log(input_gt)
            variable2= torch.log(input_pre)
            # replace inf with zero
            variable1 = torch.where(torch.isinf(variable1), torch.full_like(variable1, 0), variable1)
            variable2 = torch.where(torch.isinf(variable2), torch.full_like(variable2, 0), variable2)
            variable = variable1 - variable2
            error = torch.sqrt(torch.sum(variable * variable)/self.valid_pixels_list[index])
            error_list.append(round(error.item(),6))
        return error_list   
    
    def pixels_deviation_rate(self, deviation):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]
            variable1 = torch.maximum(input_gt/input_pre,self.devide_zero(input_pre,input_gt))
            variable2 = torch.zeros(input_pre.shape)
            variable2[variable1<deviation] = 1
            # erase invalid area
            variable2 = variable2 * self.valid_mask_list[index]
            error = torch.sum(variable2)/self.valid_pixels_list[index]
            error_list.append(round(error.item(),6))
        return error_list   

    # end point error
    def epe(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]
            error = torch.sum(abs(input_gt - input_pre))/self.valid_pixels_list[index]
            error_list.append(round(error.item(),6))
        return error_list   

    def D1_all(self):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]
            variable1 = abs(input_gt - input_pre)
            variable2 = torch.ones(input_pre.shape)
            variable2[variable1 < input_gt*0.05] = 0
            variable2[variable1 < 3] = 0
            # variable2 = variable2 * self.valid_mask_list[index]
            # cv2.imwrite('valid_mask.png',(self.valid_mask_list[index]*255).numpy())
            # cv2.imwrite('bad_pixels.png',(variable2*255).numpy())
            error = torch.sum(variable2)/self.valid_pixels_list[index]
            error_list.append(round(error.item(),6))
        return error_list   

    def D1_all_thr(self,thr=3):
        error_list = []
        for index in range(self.image_number):
            input_gt = self.gt_mask_list[index]
            input_pre = self.generated_mask_list[index]

            # mask = np.zeros(shape = input_pre.shape)
            # mask[input_pre>0]=1
            # mask[input_gt==0]=0
            # input_pre = input_pre[mask==1]
            # input_gt = input_gt[mask==1]

            variable1 = abs(input_gt - input_pre)
            variable2 = torch.zeros(input_pre.shape)
            # variable2[variable1 > input_gt*0.05] = 1
            variable2[variable1 > thr] = 1
            # variable2 = variable2 * self.valid_mask_list[index]
            # cv2.imwrite('valid_mask.png',(self.valid_mask_list[index]*255).numpy())
            # cv2.imwrite('bad_pixels.png',(variable2*255).numpy())
            error = torch.sum(variable2)/self.valid_pixels_list[index]
            error_list.append(round(error.item(),6))
        return error_list   
    
    # # mean absolute logarithmic error
    # def log_MAE(self, input_gt, input_pre):
    #     variable1 = torch.log(input_gt)
    #     variable2= torch.log(input_pre)
    #     # replace inf with zero
    #     variable1 = torch.where(torch.isinf(variable1), torch.full_like(variable1, 0), variable1)
    #     variable2 = torch.where(torch.isinf(variable2), torch.full_like(variable2, 0), variable2)
    #     variable = variable1 - variable2
    #     # replace nan with zero
    #     variable = torch.where(torch.isnan(variable), torch.full_like(variable, 0), variable)
    #     error = torch.sum(abs(variable))/self.valid_pixels
    #     return error  
    
    # # inverse mean absolute error
    # def i_MAE(self, input_gt, input_pre):
    #     variable1 = 1.0/input_gt
    #     variable2 = 1.0/input_pre
    #     # replace inf with zero
    #     variable1 = torch.where(torch.isinf(variable1), torch.full_like(variable1, 0), variable1)
    #     variable2 = torch.where(torch.isinf(variable2), torch.full_like(variable2, 0), variable2)
    #     error = torch.sum(abs(variable1 - variable2))/self.valid_pixels
    #     return error     
   
    
    # # inverse root mean square error
    # def iRMSE(self, input_gt, input_pre):
    #     variable1 = 1.0/input_gt
    #     variable2 = 1.0/input_pre
    #     # replace inf with zero
    #     variable1 = torch.where(torch.isinf(variable1), torch.full_like(variable1, 0), variable1)
    #     variable2 = torch.where(torch.isinf(variable2), torch.full_like(variable2, 0), variable2)
    #     variable = variable1 - variable2
    #     error = torch.sqrt(torch.sum(variable * variable)/self.valid_pixels)
    #     return error  
    
    # devide and process the nan item in the ans
    def devide_zero(self,x,y):
        variable = x/y
        variable = torch.where(torch.isnan(variable), torch.full_like(variable, 0), variable)
        return variable


class evaluator_image():
    def __init__(self):
        # for others
        self.image_list = []
        self.image2_list = []
        # for pixel NCC
        self.image_list2 = []
        self.image2_list2 = []
        # for image NCC
        self.image_flatten_list = []
        self.image2_flatten_list = []
        # area of valid pixels
        self.valid_mask_list = []
        # number of valid pixels
        self.valid_pixels_list = []
    
    # input: 
    # (H, W, 3)
    # input and accumulate the gt_mask
    def input_image(self, image):
        # check_zero = torch.zeros(image.shape)
        # # check_zero[image==0] = 1
        # # if torch.sum(check_zero)>0:
        # #     print('zero exists in input_data, such pixel will be ignored')
        image = image.astype(np.float32)
        self.image_list.append(image)
        # flat along pixel level
        self.image_list2.append(np.reshape(image,(-1,3)))

    # input and accumulate the generated_mask
    def input_data(self, image2):
        image2 = image2.astype(np.float32)
        self.image2_list.append(image2)
        # flat along pixel level
        self.image2_list2.append(np.reshape(image2,(-1,3)))
        
    # process the list and output the metrics
    def process(self, if_ignore_zero=True):
        '''
        process the accumulated list to tensor [number of images, ...] 
        and get number of images
        '''
        
        self.image_number = len(self.image_list)
        
        # ignore the pixels that have gt values zero
        if if_ignore_zero:
            for index in range(self.image_number):
                
                H,W,_ = self.image2_list[index].shape
                self.valid_pixels_list.append(H*W)
                
                # # ignore the invalid area
                # image_sum = np.sum(self.image2_list[index],axis=2)
                # valid_mask = np.ones(image_sum.shape)
                # valid_mask[image_sum==0] = 0
                # # ignore the invalid area
                # image_sum = np.sum(self.image_list[index],axis=2)
                # valid_mask[image_sum==0] = 0
                # self.valid_pixels_list.append(np.sum(valid_mask))
                
                # valid_mask = np.expand_dims(valid_mask,2)
                # valid_mask = valid_mask.repeat(3,axis=2)
                # self.image2_list[index] = self.image2_list[index] * valid_mask
                # self.image_list[index] = self.image_list[index] * valid_mask
                # # print('mean disp of file pair {}'.format(index+1),torch.mean(self.gt_mask_list[index]),torch.mean(self.generated_mask_list[index]))

                # image_flatten_sum = np.sum(self.image2_list2[index],axis=1)
                # # delete the invalid area
                # remove_ixs=np.where(image_flatten_sum==0)[0]
                # self.image_list2[index]=np.delete(self.image_list2[index],remove_ixs,0)
                # self.image2_list2[index]=np.delete(self.image2_list2[index],remove_ixs,0)
                
                # image_flatten_sum = np.sum(self.image_list2[index],axis=1)
                # # delete the invalid area
                # remove_ixs=np.where(image_flatten_sum==0)[0]
                # self.image_list2[index]=np.delete(self.image_list2[index],remove_ixs,0)
                # self.image2_list2[index]=np.delete(self.image2_list2[index],remove_ixs,0)
                # flat all
                self.image_flatten_list.append(self.image_list2[index].reshape(-1))
                self.image2_flatten_list.append(self.image2_list2[index].reshape(-1))
        
        value_dic_img,value_dic_mean = self.get_value()
        return value_dic_img,value_dic_mean
    
    def get_value(self):
        value_dic_img = {}
        value_dic_mean = {}
        
        # calculate metrics
        # value_dic_img['SSIM'] = self.SSIM()
        value_dic_img['PSNR'] = self.PSNR()
        value_dic_img['MSE'] = self.MSE()
        # value_dic_img['pixel_NCC'] = self.pixel_NCC()
        # value_dic_img['image_NCC'] = self.image_NCC()
        
        for key in value_dic_img.keys():
            value_dic_mean[key] = float(round(np.mean(value_dic_img[key]),6))
        
        return value_dic_img,value_dic_mean
    
    # # absolute relative error
    # def image_NCC(self):
    #     score_list = []
    #     for index in range(self.image_number):
    #         image1 = self.image_flatten_list[index]
    #         image2 = self.image2_flatten_list[index]
    #         mean1 = np.mean(image1)
    #         mean2 = np.mean(image2)
    #         num = image1.shape[0]
    #         score = (np.sum(image1*image2) - num*mean1*mean2)/np.sqrt((np.sum(image1*image1)-num*mean1*mean1)*(np.sum(image2*image2)-num*mean2*mean2))
    #         score = (score+1)/2
    #         if score > 1:
    #             print('big NCC: ',score)
    #             score = 1
    #         assert (score >= 0 and score <= 1), score
    #         score = 1 - score
    #         score_list.append(float(round(score,6)))
    #     return score_list   
    
    # def L1(self):
    #     error_list = []
    #     for index in range(self.image_number):
    #         image1 = self.image_flatten_list[index]
    #         image2 = self.image2_flatten_list[index]
    #         error = np.mean(abs(image1-image2))/255
    #         error_list.append(float(round(error,6)))
    #     return error_list 
    
    # def SSIM(self):
    #     score_list = []
    #     for index in range(self.image_number):
    #         image1 = torch.tensor(np.transpose(self.image_list[index],(2,0,1))).unsqueeze(0)
    #         image2 = torch.tensor(np.transpose(self.image2_list[index],(2,0,1))).unsqueeze(0)
    #         C1 = 0.01 ** 2
    #         C2 = 0.03 ** 2

    #         mu_x = torch.nn.AvgPool2d(3, 1)(image1)
    #         mu_y = torch.nn.AvgPool2d(3, 1)(image2)
    #         mu_x_mu_y = mu_x * mu_y
    #         mu_x_sq = mu_x.pow(2)
    #         mu_y_sq = mu_y.pow(2)

    #         sigma_x = torch.nn.AvgPool2d(3, 1)(image1 * image1) - mu_x_sq
    #         sigma_y = torch.nn.AvgPool2d(3, 1)(image2 * image2) - mu_y_sq
    #         sigma_xy = torch.nn.AvgPool2d(3, 1)(image1 * image2) - mu_x_mu_y

    #         SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    #         SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    #         SSIM = SSIM_n / SSIM_d
    #         score = torch.mean(torch.clamp((1 + SSIM) / 2, 0, 1))
    #         score = 1-score
    #         score_list.append(float(round(score.item(),6)))
    #     return score_list   
    
    def MSE(self):
        error_list = []
        for index in range(self.image_number):
            image1 = self.image_flatten_list[index]
            image2 = self.image2_flatten_list[index]
            error = np.mean(abs(image1-image2)**2)
            error_list.append(float(round(error,6)))
        return error_list 
    
    def PSNR(self):
        error_list = []
        for index in range(self.image_number):
            image1 = self.image_flatten_list[index]
            image2 = self.image2_flatten_list[index]
            MSE = np.mean(abs(image1-image2)**2)
            error=np.mean(10*np.log(255**2/MSE)/np.log(10))
            error_list.append(float(round(error,6)))
        return error_list 



def calcu_EPE(pre,gt):
    ans = torch.mean(torch.abs(gt - pre))
    return ans

def calcu_PEP(pre,gt,thr=1):
    abs_diff = torch.abs(gt - pre)
    num_base = torch.sum(abs_diff>=0)
    num_error = torch.sum(abs_diff>=thr)
    ans = num_error/num_base
    return ans

def calcu_D1all(pre,gt):
    abs_diff = torch.abs(gt - pre)
    num_base = torch.sum(abs_diff>=0)
    error = torch.ones(abs_diff.shape)
    error[abs_diff < gt*0.05] = 0
    error[abs_diff < 3] = 0
    num_error = torch.sum(error)
    ans = num_error/num_base
    return ans
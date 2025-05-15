import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def save_rgb_array_as_jpg(folder_path, image_name, rgb_array):
    """
    将一个 3D RGB 数组保存为指定文件夹中的 JPG 图像。

    Args:
        folder_path (str): 图像将要保存的文件夹路径。
        image_name (str): 图像文件的名称（不包括扩展名）。
        rgb_array (np.ndarray): 表示 RGB 图像的 3D numpy 数组。

    Returns:
        None
    """
    # 创建文件夹路径（如果不存在）
    os.makedirs(folder_path, exist_ok=True)

    # 创建图像的完整路径
    image_path = os.path.join(folder_path, f"{image_name}.jpg")
    
    rgb_array = np.transpose(rgb_array, (1, 2, 0))

    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(np.uint8(rgb_array))

    # 保存图像为 JPG
    image.save(image_path)

    print(f"图像已保存为 {image_path}")

def model_structure(model):
    '''
    打印模型参数信息
    '''
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

def vasualize_obs(game, gt_obs=None, imagine_obs=None, pre_obs=None, pre2obss = None):

    save_path = '/home/hq/LSTW/MSTORM_base/obs_images' + '/' + game
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gt_obs_list = gt_obs # [n, 1, 64, 64, 3] [nparray]
    if gt_obs is not None:
        for i,obs in enumerate(gt_obs_list):
            image_path = os.path.join(save_path, str(i) + '_gt.jpg')

            if obs.shape[0] == 16: # [16,1,3,64,64]
                obs = obs[0]* 255

                # obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
                obs = np.transpose(obs, (1, 2, 0))
                image = Image.fromarray(obs.astype('uint8'))
            else:
                
                image = Image.fromarray(obs.squeeze(0).astype('uint8'))

            image.save(image_path)

    if imagine_obs is not None:
        imagine_obs_list = imagine_obs  # [n, 1, 1, 3, 64, 64] [tensor]
        for i,obs in enumerate(imagine_obs_list):
            image_path = os.path.join(save_path, str(i) + '_imagine.jpg')

            if obs.ndim == 4:
                obs = obs[0]
                obs_to_pil = transforms.ToPILImage()
                image = obs_to_pil(obs)
            else:
                obs_to_pil = transforms.ToPILImage()
                image = obs_to_pil(obs.squeeze(0).squeeze(0))

            image.save(image_path)

    if pre_obs is not None:
        pre_obs_list = pre_obs  # [n, 1, 1, 3, 64, 64] [tensor]
        for i,obs in enumerate(pre_obs_list):
            image_path = os.path.join(save_path, str(i) + '_pre.jpg')

            obs_to_pil = transforms.ToPILImage()
            image = obs_to_pil(obs.squeeze(0).squeeze(0))

            image.save(image_path)

    if pre2obss is not None:
        pre2obss_list = pre2obss  # [n, 1, 1, 3, 64, 64] [tensor]
        for i,obs in enumerate(pre2obss_list):
            image_path = os.path.join(save_path, str(i) + '_pre2.jpg')

            obs_to_pil = transforms.ToPILImage()
            image = obs_to_pil(obs.squeeze(0).squeeze(0))

            image.save(image_path)

    pass

def vasualize_imagine_obs(game, gt_obs=None, imagine_obs=None, pre_obs=None, pre2obss = None, index=0):

    save_path = '/home/hq/LSTW/MSTORM_base/obs_images' + '/' + game + '/' + str(index)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gt_obs_list = gt_obs # [n, 1, 64, 64, 3] [nparray]
    if gt_obs is not None:
        for i,obs in enumerate(gt_obs_list):
            image_path = os.path.join(save_path, str(i) + '_gt.jpg')

            if obs.shape[0] == 16: # [16,1,3,64,64]
                obs = obs[0]* 255

                # obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
                obs = np.transpose(obs, (1, 2, 0))
                image = Image.fromarray(obs.astype('uint8'))
            else:
                
                image = Image.fromarray(obs.squeeze(0).astype('uint8'))

            image.save(image_path)

    if imagine_obs is not None:
        imagine_obs_list = imagine_obs  # [n, 1, 1, 3, 64, 64] [tensor]
        for i,obs in enumerate(imagine_obs_list):
            image_path = os.path.join(save_path, str(i) + '_imagine.jpg')

            if obs.shape[0] == 16: # [16,1,3,64,64]
                image_path = os.path.join(save_path, str(i) + '_imagine')
                if not os.path.exists(image_path):
                    os.makedirs(image_path)

                for j in range(obs.shape[0]):

                    one_obs = obs[j]* 255

                    # obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
                    one_obs = np.transpose(one_obs, (1, 2, 0))
                    image = Image.fromarray(one_obs.astype('uint8'))

                    obs_path = os.path.join(image_path, str(j) + '_imagine.jpg')

                    image.save(obs_path)

    if pre_obs is not None:
        pre_obs_list = pre_obs  # [n, 1, 1, 3, 64, 64] [tensor]
        for i,obs in enumerate(pre_obs_list):
            image_path = os.path.join(save_path, str(i) + '_pre.jpg')

            obs_to_pil = transforms.ToPILImage()
            image = obs_to_pil(obs.squeeze(0).squeeze(0))

            image.save(image_path)

    if pre2obss is not None:
        pre2obss_list = pre2obss  # [n, 1, 1, 3, 64, 64] [tensor]
        for i,obs in enumerate(pre2obss_list):
            image_path = os.path.join(save_path, str(i) + '_pre2.jpg')

            obs_to_pil = transforms.ToPILImage()
            image = obs_to_pil(obs.squeeze(0).squeeze(0))

            image.save(image_path)

    pass


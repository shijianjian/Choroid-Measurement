import os
import cv2
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CHOROID_CONTOUR_COLOUR = ['yellow']
MACULAR_COLOUR = ['yellow']

SUBFOVEAL_FROM_MACULAR = 500
INNER_SECTOR_FROM_SUBFOVEAL = 1000
OUTER_SECTOR_FROM_INNER_SECTOR = 1000

PIXEL_HEIGHT = 2.621
PIXEL_WIDTH = 8.789


class Helpers(object):

    @staticmethod
    def get_coord(point_image, show_result=False):
        arg_idx = np.argmax(point_image[:, :, 0])
        row = arg_idx % point_image.shape[1]
        col = arg_idx // point_image.shape[1]
        if show_result:
            plt.imshow(cv2.circle(point_image.copy(), (row, col), 1, (255, 0, 255), thickness=5, lineType=8, shift=0))
        return (row, col)

    @staticmethod
    def line_connector(line_image, show_result=False):
        line_image = cv2.morphologyEx(line_image, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
        line_image = cv2.addWeighted(cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY), 4, cv2.blur(cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY), (5, 5)), -1, 20)
        ret, line_image = cv2.threshold(line_image, 50, 255, cv2.THRESH_BINARY)
        if show_result:
            plt.imshow(line_image)
        return line_image

    @staticmethod
    def color_chooser(image, colours=['yellow', 'red']):
        def get_boundry(colour):
            if type(colour) == str:
                if colour == 'red':
                    lower = [200, 0, 0]
                    upper = [255, 50, 50]
                elif colour == 'yellow':
                    lower = [20, 100, 100]
                    upper = [40, 255, 255]
                elif colour == 'white':
                    lower = [0, 0, 100]
                    upper = [255, 155, 255]
                elif colour == 'green':
                    lower = [0, 255, 0]
                    upper = [0, 255, 0]
                elif colour == 'blue':
                    lower = [0, 0, 0]
                    upper = [0, 0, 255]
                else:
                    raise ValueError()
            else:
                raise ValueError()
            return lower, upper
        combined_mask = None
        for colour in colours:
            if colour == 'yellow' or colour == 'white':
                # using HSV colour space
                _image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            else:
                _image = image.copy()
            lower, upper = get_boundry(colour)
            mask = cv2.inRange(_image, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        output = cv2.bitwise_and(image, image, mask=combined_mask)
        output = cv2.GaussianBlur(output, (3, 3), 1)
        return output


class ChoroidSegmentor(object):
    def __init__(self, pixel_to_um_rate=(PIXEL_WIDTH, PIXEL_HEIGHT)):
        '''
        pixel_to_um_rate: one pixel convers how much um.
        '''
        self.pixel_to_um_rate = pixel_to_um_rate

    def get_macular_point(self, img, colours=['red']):
        dot = Helpers.color_chooser(img, colours=colours)
        macular_point = Helpers.get_coord(dot)
        logger.debug("Macular Position, %s" % str(macular_point))
        return macular_point

    def sector_crop(self, image, coord):
        '''
        image: image numpy array.
        coord: cutting point around
        '''
        def get_pixel_range(left, right, position):
            if position == 'right' or position == 'center':
                factor = 1
            elif position == 'left':
                factor = -1
            pixel_a = int(left / self.pixel_to_um_rate[0]) * factor
            pixel_b = int(right / self.pixel_to_um_rate[0]) * factor
            if factor == -1:
                pixel_c = pixel_a
                pixel_a = pixel_b
                pixel_b = pixel_c
            logger.debug("Position %s, %d -> %d" % (position, pixel_a, pixel_b))
            return pixel_a, pixel_b

        res = []
        for position in ['left', 'center', 'right']:
            if position == 'center':
                # (a) Subfoveal (central 1000um centered at the macula)
                pixel_a, pixel_b = get_pixel_range(-SUBFOVEAL_FROM_MACULAR, SUBFOVEAL_FROM_MACULAR, 'center')
                subfoveal = image[:, coord[0] + pixel_a:coord[0] + pixel_b, :]
                logger.debug("%s, Subfoveal, cropping %d->%d" % (position, coord[0] + pixel_a, coord[0] + pixel_b))
                res.append(subfoveal)
                continue
            # (b) Inner Sector (500um to 1500um from the macula)
            pixel_a, pixel_b = get_pixel_range(SUBFOVEAL_FROM_MACULAR, SUBFOVEAL_FROM_MACULAR + INNER_SECTOR_FROM_SUBFOVEAL, position)
            innersector = image[:, coord[0] + pixel_a:coord[0] + pixel_b, :]
            logger.debug("%s, Inner Sector, cropping %d->%d" % (position, coord[0] + pixel_a, coord[0] + pixel_b))
            # (c) Outer Sector (1500 to 2500 from the macula)
            pixel_a, pixel_b = get_pixel_range(SUBFOVEAL_FROM_MACULAR + INNER_SECTOR_FROM_SUBFOVEAL, SUBFOVEAL_FROM_MACULAR + INNER_SECTOR_FROM_SUBFOVEAL + OUTER_SECTOR_FROM_INNER_SECTOR, position)
            outersector = image[:, coord[0] + pixel_a:coord[0] + pixel_b, :]
            logger.debug("%s, Outer Sector, cropping %d->%d" % (position, coord[0] + pixel_a, coord[0] + pixel_b))
            if position == 'right':
                res.append(innersector)
                res.append(outersector)
            elif position == 'left':
                res.append(outersector)
                res.append(innersector)
        return res

    def find_contours(self, img):
        if cv2.__version__.startswith('3'):
            _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        res = []
        for i in range(len(contours)):
            draw = cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).copy(), contours, i, (0, 255, 0), thickness=cv2.FILLED)
            selected = Helpers.color_chooser(draw, ['green'])
            res.append(np.argmax(cv2.cvtColor(selected, cv2.COLOR_RGB2GRAY), axis=0))
        return contours, res

    def get_choroid_contours(self, sector, colours=['yellow']):
        sector = Helpers.color_chooser(sector, colours)
        sector = Helpers.line_connector(sector)
        return self.find_contours(sector)


class ThicknessCalculator(object):

    def __init__(self, pixel_to_um_rate):
        self.pixel_to_um_rate = pixel_to_um_rate
        self.reset()

    def set_outcome_validity(self, valid):
        self.valid = valid

    def reset(self):
        self.valid = True
        self.results = []

    def compute(self, array, idx, metrics=['area', 'avg_thickness', 'single_point_thickness']):
        met = {}
        if 'single_point_thickness' in metrics:
            (upper, bot) = self.cal_single_point_thickness(array, idx) if self.valid else (.0, .0)
            met.update({'single_point_thickness_upper': upper, 'single_point_thickness_bot': bot})
        if 'avg_thickness' in metrics:
            (upper, bot) = self.cal_avg_thickness(array) if self.valid else (.0, .0)
            met.update({'avg_thickness_upper': upper, 'avg_thickness_bot': bot})
        if 'area' in metrics:
            (upper, bot) = self.cal_area(array) if self.valid else (.0, .0)
            met.update({'area_upper': upper, 'area_bot': bot})
        return met

    def calculates(self, sectors_choroid, sectors_dst=None, sectors_dst_binary=None, metrics=['area', 'avg_thickness', 'single_point_thickness']):
        segmentor = ChoroidSegmentor()
        for idx, sector in enumerate(sectors_choroid):
            contours, array = segmentor.get_choroid_contours(sectors_choroid[idx], colours=CHOROID_CONTOUR_COLOUR)
            if len(contours) != 3:
                logger.debug('Found %d contours on sector %d.' % (len(contours), idx))
                self.set_outcome_validity(False)
            res_dict = {'contour_array': array}
            res_dict.update({'contour': contours})
            res_dict.update({'src_sector': sectors_choroid[idx]})
            res_dict.update({'dst_sector': sectors_dst[idx]})
            met = self.compute(array, idx, metrics)
            if 'cvi' in metrics:
                (upper, bot) = self.cal_cvi(array, sectors_dst_binary[idx]) if self.valid else (.0, .0)
                res_dict.update({'bin_sector': sectors_dst_binary[idx]})
                met.update({'cvi_upper': upper, 'cvi_bot': bot})
            res_dict.update({'metrics': met})
            self.results.append(res_dict)
        return res_dict

    def cal_area(self, contour_array):
        areas = sorted(contour_array, key=lambda x: np.sum(x))
        upper = np.sum(areas[1] - areas[0]) * self.pixel_to_um_rate[0] * self.pixel_to_um_rate[1]
        bot = np.sum(areas[2] - areas[1]) * self.pixel_to_um_rate[0] * self.pixel_to_um_rate[1]
        return upper, bot

    def cal_cvi(self, contour_array, binary_img):
        def cal_cvi_one(array_up, array_bot):
            selected = []
            for idx, (a, b) in enumerate(zip(array_up, array_bot)):
                selected.append(np.max(binary_img[b:a, idx], axis=1))
            concat = np.concatenate(selected)
            return (len(concat) - np.count_nonzero(concat)) / len(concat)

        areas = sorted(contour_array, key=lambda x: np.sum(x))
        upper = cal_cvi_one(areas[1], areas[0])
        bot = cal_cvi_one(areas[2], areas[1])
        return upper, bot

    def cal_avg_thickness(self, contour_array):
        areas = sorted(contour_array, key=lambda x: np.sum(x))
        upper = np.sum(areas[1] - areas[0]) * self.pixel_to_um_rate[1]
        bot = np.sum(areas[2] - areas[1]) * self.pixel_to_um_rate[1]
        return upper / len(areas[1]), bot / len(areas[1])

    def cal_single_point_thickness(self, contour_array, idx):
        '''
        Calculate middle when given index of 2, calclulation as below:
        | -> | -> | . | <- | <- |
        '''
        areas = sorted(contour_array, key=lambda x: np.sum(x))
        if idx < 2:
            upper = (areas[1][-1] - areas[0][-1]) * self.pixel_to_um_rate[1]
            bot = (areas[2][-1] - areas[1][-1]) * self.pixel_to_um_rate[1]
        if idx == 2:
            select = len(areas[1]) // 2
            upper = (areas[1][select] - areas[0][select]) * self.pixel_to_um_rate[1]
            bot = (areas[2][select] - areas[1][select]) * self.pixel_to_um_rate[1]
        if idx > 2:
            upper = (areas[1][0] - areas[0][0]) * self.pixel_to_um_rate[1]
            bot = (areas[2][0] - areas[1][0]) * self.pixel_to_um_rate[1]
        return upper, bot


class ChorialResultPostProcessor(object):

    def __init__(self):
        pass

    def get_drawing(self, contours, sector):
        if len(contours) != 3:
            drawing = self.random_colour_contour_drawing(sector.copy(), contours)
        else:
            drawing = cv2.drawContours(sector.copy(), contours, -1, (0, 255, 0), thickness=cv2.FILLED)
        return drawing

    def random_colour_contour_drawing(self, image, contours):
        def random_colour():
            return (np.random.randint(0, 255), np.random.randint(100, 150), np.random.randint(150, 255))
        image = cv2.drawContours(image.copy(), contours, 0, random_colour(), thickness=cv2.FILLED)
        for i in range(1, len(contours)):
            image = cv2.drawContours(image.copy(), contours, i, random_colour(), thickness=cv2.FILLED)
        return image

    def extract_results(self, results):
        ''' Extract the results from calcuate function '''
        imgs = []
        dfs = []
        # (a) Subfoveal (central 1000um centered at the macula)
        # (b) Inner Sector (500um to 1500um from the macula)
        # (c) Outer Sector (1500um to 3000um from the macula)
        # print(len(results))
        prefix = ['left_outer', 'left_inner', 'subfoveal', 'right_inner', 'right_outer']
        for i, res in enumerate(results):
            metrics = res['metrics']
            # original image
            contour_pic = self.get_drawing(res['contour'], res['dst_sector'])
            on_image_info = [
                {'key': 'SP.CT Up', 'value': 'single_point_thickness_upper'},
                {'key': 'SP.CT Bot', 'value': 'single_point_thickness_bot'},
                {'key': 'Avg.CT Up', 'value': 'avg_thickness_upper'},
                {'key': 'Avg.CT Bot', 'value': 'avg_thickness_bot'},
                {'key': 'Area Up', 'value': 'area_upper'},
                {'key': 'Area Bot', 'value': 'area_bot'},
                {'key': 'CVI Up', 'value': 'cvi_upper'},
                {'key': 'CVI Bot', 'value': 'cvi_bot'},
            ]
            for idx, item in enumerate(on_image_info):
                cv2.putText(contour_pic, "%s:" % item['key'], (3, (idx * 2 + 1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(contour_pic, "%.2f" % metrics[item['value']], (3, (idx * 2 + 2) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            imgs.append(np.pad(contour_pic, [(0, 0), (5, 5), (0, 0)], mode='constant'))

            # binary image
            contour_pic = self.get_drawing(res['contour'], res['bin_sector'])
            on_image_info = [
                {'key': 'CVI Up', 'value': 'cvi_upper'},
                {'key': 'CVI Bot', 'value': 'cvi_bot'},
            ]
            for idx, item in enumerate(on_image_info):
                cv2.putText(contour_pic, "%s:" % item['key'], (3, (idx * 2 + 1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(contour_pic, "%.2f" % metrics[item['value']], (3, (idx * 2 + 2) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            imgs.append(np.pad(contour_pic, [(0, 0), (5, 5), (0, 0)], mode='constant'))

            dfs.append(pd.DataFrame([metrics]).add_prefix(prefix[i] + "_").T)
        return dfs, imgs


class ChoroidProcessor(object):

    def __init__(self, macular_position_image_path, choroid_position_image_path, original_image_path, binary_image_path=None, pixel_to_um_rate=(PIXEL_WIDTH, PIXEL_HEIGHT)):
        self.macular_position_img = plt.imread(macular_position_image_path)
        self.choroid_position_img = plt.imread(choroid_position_image_path)
        self.original_img = plt.imread(original_image_path)
        logger.debug('Macular position image read from path %s.' % macular_position_image_path if self.macular_position_img is not None else 'Macular position image not read from path %s.' % macular_position_image_path)
        logger.debug('Macular position image shape %s' % str(self.macular_position_img.shape))
        logger.debug('Choroid position image read from path %s.' % choroid_position_image_path if self.choroid_position_img is not None else 'Choroid position image not read from path %s.' % choroid_position_image_path)
        logger.debug('Choroid position image shape %s' % str(self.choroid_position_img.shape))
        logger.debug('Original image read %s.' % original_image_path if self.original_img is not None else 'Original image not read %s.' % original_image_path)
        logger.debug('Original position image shape %s' % str(self.original_img.shape))
        if self.macular_position_img.shape[-1] == self.choroid_position_img.shape[-1] == 3:
            if self.original_img.shape[-1] == 1:
                self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_GRAY2RGB)
        if self.macular_position_img is None:
            raise ValueError("Macular position image not found.")
        if self.choroid_position_img is None:
            raise ValueError("Choroid position image not found.")
        if self.original_img is None:
            raise ValueError("Original image not found.")
        assert self.macular_position_img.shape == self.original_img.shape, \
            'macular_position_image_path %s and original_image_path %s are not equal.' \
            % (self.macular_position_img.shape, self.original_img.shape)
        assert self.choroid_position_img.shape == self.original_img.shape, \
            'choroid_position_image_path %s and original_image_path %s are not equal.' \
            % (self.choroid_position_img.shape, self.original_img.shape)
        if binary_image_path is not None:
            binary_img = plt.imread(binary_image_path)
            logger.debug('Binary image read %s.' % binary_image_path if binary_img is not None else 'Binary image not read %s.' % binary_image_path)
            if len(binary_img.shape) == 2:
                binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
            if binary_img is None:
                raise ValueError("Binary image not found.")
            self.binary_img = binary_img
            assert self.binary_img.shape == self.original_img.shape, \
                'binary_image_path %s and original_image_path %s are not equal.' \
                % (self.binary_img.shape, self.original_img.shape)

        self.pixel_to_um_rate = pixel_to_um_rate
        self.segmentor = ChoroidSegmentor(pixel_to_um_rate=self.pixel_to_um_rate)

        self.rotate_image(include_bin=True if binary_image_path is not None else False, by_slice=(1, 3), inplace=True)

        self.calculator = ThicknessCalculator(pixel_to_um_rate=self.pixel_to_um_rate)
        self.post_processor = ChorialResultPostProcessor()

    def rotate_image(self, include_bin, by_slice=(1, 3), inplace=True):
        """ by_slice by default will refer to the 2nd and the 4th slice around the macular position to correct the image rotation."""
        import scipy.ndimage
        macular_point = self.segmentor.get_macular_point(self.macular_position_img, colours=MACULAR_COLOUR)
        segs = self.segmentor.sector_crop(self.choroid_position_img, macular_point)

        b = self.segmentor.get_choroid_contours(segs[by_slice[0]], colours=CHOROID_CONTOUR_COLOUR)[1][0][0]
        c = self.segmentor.get_choroid_contours(segs[by_slice[1]], colours=CHOROID_CONTOUR_COLOUR)[1][0][-1]
        delt_y = c - b
        # Rotation using the same scaling strategy.
        delt_x = (SUBFOVEAL_FROM_MACULAR + INNER_SECTOR_FROM_SUBFOVEAL) * 2 / self.pixel_to_um_rate[0]
        degree = np.degrees(np.arctan(delt_y / delt_x))
        rotated_pixel_width = np.cos(np.deg2rad(degree)) * self.pixel_to_um_rate[0]
        rotated_pixel_height = np.cos(np.deg2rad(degree)) * self.pixel_to_um_rate[1]
        self.pixel_to_um_rate = (rotated_pixel_width, rotated_pixel_height)
        logger.debug('Rotation degree: %f' % degree)
        logger.debug('Scaling to: %s' % str(self.pixel_to_um_rate))
        if inplace:
            self.original_img = scipy.ndimage.rotate(self.original_img, degree, order=0, prefilter=False)
            self.macular_position_img = scipy.ndimage.rotate(self.macular_position_img, degree, order=0, prefilter=False)
            self.choroid_position_img = scipy.ndimage.rotate(self.choroid_position_img, degree, order=0, prefilter=False)
            if include_bin:
                self.binary_img = scipy.ndimage.rotate(self.binary_img, degree, order=0, prefilter=False)
        else:
            original_img = scipy.ndimage.rotate(self.original_img, degree, order=0, prefilter=False)
            macular_position_img = scipy.ndimage.rotate(self.macular_position_img, degree, order=0, prefilter=False)
            choroid_position_img = scipy.ndimage.rotate(self.choroid_position_img, degree, order=0, prefilter=False)
            if include_bin:
                binary_img = scipy.ndimage.rotate(self.binary_img, degree, order=0, prefilter=False)
                return original_img, macular_position_img, choroid_position_img, binary_img
            return original_img, macular_position_img, choroid_position_img

    def start(self, metrics=['area', 'avg_thickness', 'single_point_thickness', 'cvi']):

        macular_point = self.segmentor.get_macular_point(self.macular_position_img, colours=MACULAR_COLOUR)
        sectors_choroid = self.segmentor.sector_crop(self.choroid_position_img, macular_point)
        sectors_dst = self.segmentor.sector_crop(self.original_img, macular_point)

        if 'cvi' in metrics:
            sectors_dst_binary = self.segmentor.sector_crop(self.binary_img, macular_point)
        else:
            sectors_dst_binary = None

        self.calculator.calculates(sectors_choroid, sectors_dst, sectors_dst_binary, metrics=metrics)
        dfs, imgs = self.post_processor.extract_results(self.calculator.results)

        self.valid = self.calculator.valid
        return dfs, imgs, self.calculator.results


def run(image_pre_name, dst_path, metrics=['area', 'avg_thickness'], show_results=False):
    '''
    Eg. image_pre_name = STDR059_20151209_103654_OPT_R_001
        cal = ThicknessCalculator(
            'STDR059_20151209_103654_OPT_R_001-0003.tif',
            'STDR059_20151209_103654_OPT_R_001-0002.tif',
            'STDR059_20151209_103654_OPT_R_001-0001.tif',
        )
    '''
    p = ChoroidProcessor(
        '%s-0003.tif' % image_pre_name,
        '%s-0002.tif' % image_pre_name,
        '%s-0001.tif' % image_pre_name,
        '%s-NiBlack.tif' % image_pre_name,
    )
    dfs, imgs, res = p.start(metrics=['area', 'avg_thickness', 'single_point_thickness', 'cvi'])
    if show_results:
        plt.figure(figsize=(20, 20))
        plt.imshow(np.hstack(imgs))
    else:
        pd.concat(dfs).T.reset_index(drop=True).to_csv(dst_path + '.csv', index=None)
        logger.debug('Computations: %s' % str(pd.concat(dfs)))
        cv2.imwrite(dst_path + '_raw.jpg', np.hstack(imgs[::2]))
        cv2.imwrite(dst_path + '_bin.jpg', np.hstack(imgs[1::2]))
    if p.valid:
        return res
    else:
        raise ValueError('Invalid Contour Recognition, please redo the drawing.')


def get_results(run_name, status):
    csv = pd.read_csv('%s_results.csv' % run_name)
    csv['image'] = run_name
    csv['status'] = status
    return csv


if __name__ == '__main__':

    import argparse
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(
        description="Choroid Thickness measurement.\n\n"
        "Please follow the following file name conventions:\n\n"
        "\tfilename-0001.tif    -> Original Picture\n"
        "\tfilename-0002.tif    -> Yellow lines for choroid positions\n"
        "\tfilename-0003.tif    -> Yellow dot for macular position\n"
        "\tfilename-NiBlack.tif -> NiBlack Binary image\n\n"
        "Usage as below:\n"
        "\n\tpython index.py --img filename\n"
        "\n\tFor running in batch mode, please make sure the folder structure as below:\n"
        "\tOne_Folder:\n"
        "\t\tfilename(FOLDER NAME IS THE filename INSIDE)\n"
        "\t\t\tfilename-0001.tif\n"
        "\t\t\tfilename-0002.tif\n"
        "\t\t\tfilename-0003.tif\n"
        "\t\t\tfilename-NiBlack.tif -> NiBlack Binary image\n\n"
        "Usage as below:\n"
        "\n\tpython index.py --folder PATH_TO_One_Folder\n"
        "\nNote there is no file suffixes and extentions involved in the command.", formatter_class=RawTextHelpFormatter)
    # python index.py --img /home/ovs-dl/git/Victor/sample/STDR059_20151209_103654_OPT_R_001
    parser.add_argument("--img", type=str, help='Path to the image folder.')
    parser.add_argument("--folder", type=str, help='Path to the folder conatins all image-subfolders. exclude subfixes.')
    parser.add_argument("--logging_level", type=str, default='INFO', help='Logging verbose level to the console, INFO or DEBUG')
    parser.add_argument("--override", action='store_true', help='Whether to override existing results. Default is False.')
    args = parser.parse_args()

    image_pre_name = args.img
    image_folder_path = args.folder
    logging_level = args.logging_level
    ignore = not args.override

    import logging
    import datetime

    date_string_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    logger_file = os.path.join(image_folder_path if args.folder is not None else image_pre_name, 'Choroid_Measurement_Log-%s.log' % date_string_time)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logger_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()

    if logging_level is not None:
        if logging_level.upper() == 'INFO':
            logger.setLevel(logging.INFO)
        elif logging_level.upper() == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        else:
            raise ValueError('Unexpected logging_level %s' % logging_level)

    run_names = []
    if image_pre_name is not None:
        run_names.append(os.path.join(image_pre_name, image_pre_name[image_pre_name.rindex(os.path.sep) + 1:]))

    if image_folder_path is not None:
        for sub_f in os.listdir(image_folder_path):
            image_pre_name = os.path.join(image_folder_path, sub_f, sub_f)  # Repeat needed for image name prefixs
            if os.path.isdir(os.path.join(image_folder_path, sub_f)):
                run_names.append(image_pre_name)

    logger.info('In total %d found.' % len(run_names))

    dfs = []
    succeed = []
    skipped = []
    failed = []
    for idx, run_name in enumerate(run_names):
        if os.path.exists('%s_results.csv' % run_name) and ignore:
            logger.info('Skipping as existing at %d/%d, %s' % (idx + 1, len(run_names), run_name))
            dfs.append(get_results(run_name, 'skipped'))
            skipped.append(run_name)
            continue
        if args.folder is None and logging_level.upper() == 'DEBUG':
            run(run_name, '%s_results' % run_name, show_results=False)
        else:
            try:
                run(run_name, '%s_results' % run_name, show_results=False)
            except Exception as e:
                logger.info('Failed at %d/%d, %s \n stack trace as %s' % (idx + 1, len(run_names), run_name, e))
                failed.append(run_name)
                continue
        logger.info('Succeed at %d/%d, %s' % (idx + 1, len(run_names), run_name))
        succeed.append(run_name)
        dfs.append(get_results(run_name, 'runned'))
    if args.folder is not None:
        pd.concat(dfs).to_csv(os.path.join(image_folder_path, 'Choroid_Measurement_Results-%s.csv' % date_string_time), index=None)
    logger.info('Programme finished with %d total runs, %d succeed runs, %d skipped runs, %d failed runs.' % (len(run_names), len(succeed), len(skipped), len(failed)))

import os
import json
import pickle
import pprint
import matplotlib.pyplot as plt
import supervision as sv
from tqdm import tqdm


def xywh2ratio(img_w, img_h, bbox):
    # bbox_left, bbox_top, bbox_width, bbox_height = bbox

    # cx = (bbox_left + bbox_width/2.0)
    # cy = (bbox_top + bbox_height/2.0)

    # cx = 0 if cx < 0 else cx
    # cx = img_w if cx > 0 else cx
    # cy = 0 if cy < 0 else cy
    # cy = img_h if cy > 0 else cy

    # x_center_ratio = cx/img_w
    # y_center_ratio = cx/img_h
    # w_ratio = bbox_width/img_w
    # h_ratio = bbox_height/img_h

    x, y, w, h = bbox
    x_center = x + w/2.0
    y_center = y + h/2.0
    x_center_ratio = x_center/img_w * 1.0
    y_center_ratio = y_center/img_h * 1.0
    w_ratio = w/img_w * 1.0
    h_ratio = h/img_h * 1.0


    return [x_center_ratio, y_center_ratio, w_ratio, h_ratio]


nuimage_all = {
    0: 'car', 
    1: 'truck', 
    2: 'trailer', 
    3: 'bus', 
    4: 'construction_vehicle', 
    5: 'bicycle', 
    6: 'motorcycle', 
    7: 'pedestrian', 
    8: 'traffic_cone', 
    9: 'barrier'
}

nuimage_all_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
}

bdd100k_all = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign"
}
bdd100k_all_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
}

# PROMPT TRAFFIC
bdd100k_traffic = {
    0: "traffic light",
    1: "traffic sign"
}
bdd100k_traffic_map = {
    8: 0,
    9: 1
}

# PROMPT VEHICLE
bdd100k_vehicle = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "train"
}
bdd100k_vehicle_map = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
}

# PROMPT PEOPLE
bdd100k_people = {
    0: "pedestrian",
    1: "rider",
    2: "motorcycle",
    3: "bicycle"
}

bdd100k_people_properties = {
    4: "man",
    5: "woman",
    6: "child"
}

bdd100k_people_map = {
    0: 0,
    1: 1,
    6: 2,
    7: 3,
}

subset_name = 'all'
bdd100k_det_subset = nuimage_all
bdd100k_det_map = nuimage_all_map

# subset_name = 'traffic'
# bdd100k_det_subset = bdd100k_traffic
# bdd100k_det_map = bdd100k_traffic_map

# subset_name = 'vehicle'
# bdd100k_det_subset = bdd100k_vehicle
# bdd100k_det_map = bdd100k_vehicle_map

# subset_name = 'people'
# bdd100k_det_subset = bdd100k_people
# bdd100k_det_map = bdd100k_people_map


def main():

    HOME = '/home/belay/Documents/github_others/GroundingDINO'
    pp = pprint.PrettyPrinter(indent=4)
    print(f'HOME={HOME}')

    weight_name = "groundingdino_swint_ogc.pth"
    weight_path = os.path.join(HOME, "notebooks/weights", weight_name)

    conf_path = '/home/belay/Documents/github_others/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'

    print(weight_path, "; exist:", os.path.isfile(weight_path))
    print(conf_path, "; exist:", os.path.isfile(conf_path))

    from groundingdino.util.inference import load_model, load_image, predict, annotate

    saved_objs = []
    model = load_model(conf_path, weight_path)

    ############################################
    #                 NuImage                  #
    ############################################
    json_dirname = 'nuimages-v1.0_coco'
    json_name = 'nuimages_v1.0-val'
    image_dir = '/media/18T/data_thesis/NuImages/nuimages-v1.0-all'
    json_path = f'/media/18T/data_thesis/NuImages/{json_dirname}/{json_name}.json'

    ############################################
    #                 BDD100K                  #
    ############################################
    # json_dirname = 'labels_coco2'
    # json_name = 'train_cocofmt'
    # image_dir = '/media/18T/data_thesis/bdd100k/bdd100k_images_100k/images/100k/train'

    # json_dirname = 'labels_coco2'
    # json_name = 'val_cocofmt'
    # image_dir = '/media/18T/data_thesis/bdd100k/bdd100k_images_100k/images/100k/val'

    # json_dirname = 'ShuffleSplit10'
    # json_name = 'Split-1_val'
    # image_dir = '/media/18T/data_thesis/bdd100k/bdd100k_images_100k/images/100k/train'

    # json_dirname = 'OfficialVal_ShuffleSplit10'
    # json_name = 'Split-0'
    # image_dir = '/media/18T/data_thesis/bdd100k/bdd100k_images_100k/images/100k/val'

    # json_path = f'/media/18T/data_thesis/bdd100k/bdd100k_images_100k/{json_dirname}/{json_name}.json'


    BOX_TRESHOLD = 0.2
    TEXT_TRESHOLD = 0.15
    pkl_dir = '/home/belay/Documents/github_others/GroundingDINO/injects'
    pkl_path = f'{pkl_dir}/{json_dirname}_{json_name}_box{BOX_TRESHOLD:.1f}_{subset_name}_properties.pkl'


    with open(json_path, 'r') as f:
        data_dict = json.load(f)


        print(f'\n========== Process {len(data_dict["images"])} images ==========')


    for idx, coco_img_obj in tqdm(enumerate(data_dict['images'])):

        # Testing
        # if idx == 1:
        #     break

        saved_obj = {
            # 'img': None,
            'img_shape': (coco_img_obj['height'], coco_img_obj['width']),
            # 'img_feature': None,
            'img_path': os.path.join(image_dir, coco_img_obj['file_name']),
            'img_name': coco_img_obj['file_name'],
            'gt_bboxes': [],
            'gt_labels': [],
            'gt_labels_des': [],
            'phrases': None,
            'sub_tokens': None,
            'token_logits': None,
            'token_logits_raw': None,
            'class_logits': None,
            'class_logits_raw': None,
            'pred_bboxes': None,
            'pred_bboxes_unmasked': None
            # 'pred_labels': None,
            # 'pred_labels_des': None,
        }



        image_name = coco_img_obj['file_name']
        #print(f'Process {image_name}')
        # if image_name != 'b38f59d4-8dfeca9f.jpg':
        #     continue

        for anno in data_dict['annotations']:

            if coco_img_obj['id'] == anno['image_id']:

                bbox = xywh2ratio(coco_img_obj['width'], coco_img_obj['height'], anno['bbox'])
                
                # nuimages
                classid = anno["category_id"]

                # bdd100k
                #classid = anno["category_id"]-1

                
                classname = bdd100k_all[classid] # use all to map the original coco format
                

                if classname not in list(bdd100k_det_subset.values()):
                    continue
                
                new_classid = bdd100k_det_map[classid]
                saved_obj['gt_bboxes'].append(bbox)
                saved_obj['gt_labels'].append(new_classid)
                saved_obj['gt_labels_des'].append(classname)

        

        if len(saved_obj['gt_bboxes']) == 0:
            continue
        # image_dir = '/media/18T/data_thesis/bdd100k/bdd100k_images_100k/images/100k/train'
        # image_name = "15569b20-8d825141.jpg"

        image_path = saved_obj['img_path']

        #print(f'Process {image_path} ...')

        # here
        #TEXT_PROMPT = "car,pedestrian,   crosswalk"
        #TEXT_PROMPT = 'car, truck, bus, pedestrian, rider, train, motorcycle, bicycle, traffic light, traffic sign'
        # TEXT_PROMPT = "traffic light, traffic sign,crosswalk,  car"

        TEXT_PROMPT = ','.join(bdd100k_det_subset.values())
        TEXT_PROMPT_ADDON = ','.join(bdd100k_people_properties.values())
        TEXT_PROMPT = f'{TEXT_PROMPT},{TEXT_PROMPT_ADDON}'
        

        image_source, image = load_image(image_path)
        # print(f'image.shape={image.shape}')
        print(f'image_path={image_path}')

        pred_bboxes, pred_bboxes_unmasked, logits, phrases, sub_tokens, token_logits, token_logits_raw, class_logits, class_logits_raw, mask, tokenidx2class, class2tokenidx = predict(
            model=model, 
            bboxes = saved_obj['gt_bboxes'],
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )

        # print(f'Original gt_bboxes = {len(saved_obj["gt_bboxes"])}')
        # print(f'Original gt_labels = {len(saved_obj["gt_labels"])}')
        # print(f'Original pred_bboxes.shape = {pred_bboxes.shape}')
        assert len(phrases) == len(pred_bboxes), f'{image_path}'

        if '' in phrases:
            print(phrases)
        saved_obj['phrases'] = phrases
        saved_obj['token_logits'] = token_logits
        saved_obj['token_logits_raw'] = token_logits_raw
        #saved_obj['img_shape'] = [i for i, m in zip(saved_obj['img_shape'], mask) if m is True]
        #saved_obj['img_path'] = [i for i, m in zip(saved_obj['img_path'], mask) if m is True]
        #saved_obj['img_name'] = [i for i, m in zip(saved_obj['img_name'], mask) if m is True]
        #saved_obj['token_logits'] = [i for i, m in zip(saved_obj['token_logits'], mask) if m is True]
        #saved_obj['phrases'] = [i for i, m in zip(saved_obj['phrases'], mask) if m is True]
        #saved_obj['gt_bboxes'] = [i for i, m in zip(saved_obj['gt_bboxes'], mask) if m is True]
        #saved_obj['gt_labels'] = [i for i, m in zip(saved_obj['gt_labels'], mask) if m is True]
        #saved_obj['gt_labels_des'] = [i for i, m in zip(saved_obj['gt_labels_des'], mask) if m is True]

        # print(f'Filtered gt_bboxes = {len(saved_obj["gt_bboxes"])}, Filtered gt_labels = {len(saved_obj["gt_labels"])}')
        # print(f'Return class_logits = {class_logits.shape}')

        saved_obj['pred_bboxes'] = pred_bboxes
        saved_obj['pred_bboxes_unmasked'] = pred_bboxes_unmasked
        #saved_obj['sub_tokens'] = sub_tokens
        #saved_obj['token_logits'] = token_logits

        # no class logits
        # saved_obj['class_logits'] = class_logits
        # saved_obj['class_logits_raw'] = class_logits_raw
        assert len(saved_obj['phrases']) == pred_bboxes.shape[0]
        assert len(saved_obj['gt_bboxes']) == pred_bboxes_unmasked.shape[0]
        # assert len(saved_obj['class_logits']) == pred_bboxes_unmasked.shape[0]
        # assert len(saved_obj['class_logits_raw']) == pred_bboxes_unmasked.shape[0]

        if len(saved_obj['gt_bboxes']) == 0 or pred_bboxes.shape[0] == 0:
            continue

        # print(f"[Saved] gt_labels[0]={saved_obj['gt_labels'][0]}")
        # print(f"[Saved] gt_labels_des[0]={saved_obj['gt_labels_des'][0]}")
        # print(f"[Saved] gt_bboxes[0]={saved_obj['gt_bboxes'][0]}")
        # print(f"[Saved] class_logits[0]={saved_obj['class_logits'][0]}")
        # print(f"[Saved] pred_bboxes[0]={saved_obj['pred_bboxes'][0]}")

        saved_objs.append(saved_obj)

    print(f'\n----- tokenidx2class -----')
    pp.pprint(tokenidx2class)
    print('')
    print(f'\n----- class2tokenidx -----')
    pp.pprint(class2tokenidx)
    print('')

    with open(f'{pkl_path}', 'wb') as handle:
        pickle.dump(saved_objs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
import os
import argparse
import xml.etree.ElementTree as ET
import cv2
parser = argparse.ArgumentParser(description='Input begin number')
parser.add_argument('--begin', type=int, default=1, help='begin number(default 1)')
parser.add_argument('--end', type=int, default=1501, help='end number(default 1501)')
parser.add_argument('--start', type=int, default=1, help='end number(default 1)')
args = parser.parse_args()
begin_sn = args.begin
end_sn = args.end
index = args.start

basic_path = "E:/teams/U-team/dataset/cj_enm/samsi_gochang_org/episode01"
annos_paths = os.path.join(basic_path, "Annotations")
jpegs_paths = os.path.join(basic_path, "JPEGImages")
shots = ['shot_' + str(i).zfill(4) for i in range(begin_sn, end_sn+1)]

def get_annos(index, shots, annos_paths):
    shot = shots[index-1]
    annos_path = os.path.join(annos_paths, shot)
    annos = [os.path.join(annos_path, anno) for anno in os.listdir(annos_path)]
    return annos

annos = get_annos(index, shots, annos_paths)

def find_next_index(index, end_sn, shots, annos_paths):
    index += 1
    while index <= end_sn:
        if get_annos(index, shots, annos_paths):
            break
        else:
            index += 1
    return index
def find_prev_index(index, begin_sn, shots, annos_paths):
    index -= 1
    while index >= begin_sn:
        if get_annos(index, shots, annos_paths):
            break
        else:
            index -= 1
    return index

if not annos:
    temp_index = index
    index = find_next_index(index, end_sn, shots, annos_paths)
    if(index == end_sn + 1):
        index = find_prev_index(temp_index, begin_sn, shots, annos_paths)
        if (index == begin_sn - 1):
            index = temp_index
    annos = get_annos(index, shots, annos_paths)

if not annos:
    print("No target in this range")
else:
    numbers = [0, 64, 128, 192, 255]
    colors = [(i, j, k) for i in numbers for j in numbers for k in numbers]
    delay = 512
    cond = True
    id_set = set([])
    id_img_dict = {}
    valid_info_dict = {}
    print("annotation_start")
    while cond:
        shot = shots[index-1]
        jpegs_path = os.path.join(jpegs_paths, shot)
        for anno in annos:
            jpeg = os.path.join(jpegs_path, os.path.splitext(os.path.basename(anno))[0] + '.jpg')
            frame = cv2.imread(jpeg, cv2.IMREAD_COLOR)
            tree = ET.parse(anno)
            root = tree.getroot()
            for obj in root.findall('object'):
                id = int(obj.find('id').text)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colors[id], 3)
                cv2.putText(frame, str(id), (int((xmin+xmax)/2)-1, int((ymin+ymax)/2)-1), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[id])
                if not id in id_set:
                    id_set.add(id)
                    area = (xmax - xmin) * (ymax - ymin)
                    id_img_dict[id] = (jpeg, xmin, ymin, xmax, ymax, area)
                else:
                    area = (xmax - xmin) * (ymax - ymin)
                    if(id_img_dict[id][5] < area):
                        id_img_dict[id] = (jpeg, xmin, ymin, xmax, ymax, area)                    
            cv2.imshow(shot, frame)
            interrupt = cv2.waitKey(delay) & 0xFF
            if interrupt == ord('a'):
                cv2.destroyAllWindows()
                temp_index = index
                index = find_prev_index(index, begin_sn, shots, annos_paths)
                if (index == begin_sn - 1):
                    index = temp_index
                else:
                    annos = get_annos(index, shots, annos_paths)
                id_set = set([])
                id_img_dict = {}
                break
            elif interrupt == ord('d'):
                cv2.destroyAllWindows()
                temp_index = index
                index = find_next_index(index, end_sn, shots, annos_paths)
                if (index == end_sn + 1):
                    index = temp_index
                else:
                    annos = get_annos(index, shots, annos_paths)
                id_set = set([])
                id_img_dict = {}
                break
            elif interrupt == ord('w'):
                delay = max(int(delay / 2), 1)
            elif interrupt == ord('s'):
                delay = min(delay * 2, 1024)   
            elif interrupt == ord('c'):
                for key in id_img_dict.keys():
                    value = id_img_dict[key]
                    cv2.imshow(str(key), cv2.imread(value[0], cv2.IMREAD_COLOR)[value[2]:value[4], value[1]:value[3]])                
                cv2.waitKey(0)
                print(len(id_set), id_set)
                valid = True
                invalid = True
                val_id_list = []
                checked_input = input('Input valid ids : ').split('|')
                valid_checked_list = checked_input[0].split(',')
                cand_checked_list = checked_input[1].split(',')
                if(valid_checked_list[0] == ''):
                    valid = False
                    if(cand_checked_list[0] != ''):
                        invalid = False
                else:
                    if(len(valid_checked_list) != len(id_set)):
                        valid = False
                        invalid = False
                    for check_id in valid_checked_list:
                        if (int(check_id) in id_set):
                            val_id_list.append(int(check_id))
                            invalid = False
                        else:
                            valid = False
                if (valid):
                    print(shot, "is valid.")
                    valid_info_dict[shot] = (1)
                elif (invalid):
                    print(shot, "is invalid.")
                    valid_info_dict[shot] = (-1)
                else:
                    print(shot, "need check", val_id_list, "is valid.")
                    valid_info_dict[shot] = (0, val_id_list, cand_checked_list)
                cv2.destroyAllWindows()
            elif interrupt == ord('b'):
                print(len(id_set), id_set)
                cv2.waitKey(0)
            elif interrupt == ord('q'):
                cond = False
                cv2.destroyAllWindows()
                break
    print(valid_info_dict)
    print("annotation_finished")
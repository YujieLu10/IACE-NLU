from PIL import Image
from icecream import ic
import clip
import re
import os
import csv
import torch
import argparse
torch.set_num_threads(4)
clip_model, preprocess = clip.load('ViT-B/32', "cuda:0")
vq_parser = argparse.ArgumentParser(description='extract visual features')
vq_parser.add_argument('-task', default="mnli")
vq_parser.add_argument("-eval",  "--eval", action='store_true', help="eval", dest='eval')
vq_parser.add_argument("-nlu_dataset",  "--nlu_dataset", type=str, default="glue")
vq_parser.add_argument("-extract_visual",  "--extract_visual", action='store_true', help="eval", dest='eval')
vq_parser.add_argument("-extract_textual",  "--extract_textual", action='store_true', help="eval", dest='eval')
args = vq_parser.parse_args()

data_root_path = "..data/ima/{}/output/".format(args.nlu_dataset)
save_root_path = "..data/feature/{}/".format(args.nlu_dataset)

task_to_path = {
    "cola": data_root_path + "glue_cola",
    "mnli": data_root_path + "glue_mnli",
    "mnli_m": data_root_path + "glue_mnli_m",
    "mnli_mm": data_root_path + "glue_mnli_mm",
    "mrpc": data_root_path + "glue_mrpc",
    "qnli": data_root_path + "glue_qnli",
    "qqp": data_root_path + "glue_qqp",
    "rte": data_root_path + "glue_rte",
    "sst": data_root_path + "glue_sst",
    "sts": data_root_path + "glue_sts",
    "wnli": data_root_path + "glue_wnli",
    "swag": data_root_path + "swag",
    "squad1": data_root_path + "squad1",
    "squad2": data_root_path + "squad2",
}

task_to_number = {
    "mnli":392703,
    "qnli":104744,
    "qqp":363847,
    "swag":73547,
    "squad1":87599,
    "squad2":130319,
}
task_to_number_dev = {
    "mnli_m":9816,
    "mnli_mm":9833,
    "qnli":5464,
    "qqp":40431,
    "swag":20007,
    "squad1":10570,
    "squad2":11873,
}

for task_name in args.task.split(':'):
    ic(task_name)

for task_name in args.task.split(':'):
    ic(task_name)
    visual_input_path = task_to_path[task_name] + ("_dev" if args.eval else "_train")
    file_list = os.listdir(visual_input_path)
    if task_name in ["wnli", "rte"]:
        file_list.sort(key = lambda x: (int(x[:x.index('_')]), x[x.index('sentence'):[i for i, n in enumerate(x) if n == '_'][1]]))
        img1_features = []
        img2_features = []
        for idx, file in enumerate(file_list):
            sentence_img = Image.open(os.path.join(visual_input_path, file))
            img_f = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            if (idx % 2) == 0:
                img1_features.append(img_f)
            else:
                img2_features.append(img_f)
        torch.save([torch.tensor(img1_features),torch.tensor(img2_features)], os.path.join(save_root_path, str(task_name) + ("_dev" if args.eval else "_train") + ".pt"))
    elif task_name in ["swag", "sts", "mrpc", "qqp", "qnli", "mnli", "mnli_m", "mnli_mm"]:
        img1_features = []
        img2_features = []
        idx_number = task_to_number_dev[task_name] if args.eval else task_to_number[task_name]
        for idx in range(idx_number):
            file1 = str(idx+1) + "_sentence1.png"
            file2 = str(idx+1) + "_sentence2.png"
            sentence_img1 = Image.open(os.path.join(visual_input_path, file1))
            sentence_img2 = Image.open(os.path.join(visual_input_path, file2))
            img_f1 = clip_model.encode_image(preprocess(sentence_img1).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img_f2 = clip_model.encode_image(preprocess(sentence_img2).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img1_features.append(img_f1)
            img2_features.append(img_f2)
        torch.save([torch.tensor(img1_features),torch.tensor(img2_features)], os.path.join(save_root_path, str(task_name) + ("_dev" if args.eval else "_train") + ".pt"))
    elif task_name in ["squad1", "squad2"]:
        img1_features = []
        idx_number = task_to_number_dev[task_name] if args.eval else task_to_number[task_name]
        for idx in range(idx_number):
            file = str(idx+1) + "_question.png"
            sentence_img = Image.open(os.path.join(visual_input_path, file))
            img_f = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img1_features.append(img_f)
        torch.save([torch.tensor(img1_features)], os.path.join(save_root_path, str(task_name) + ("_dev" if args.eval else "_train") + ".pt"))
    elif task_name in ["sst", "cola"]:
        file_list.sort(key = lambda x: int(x[:x.index('_')]))
        img1_features = []
        for idx, file in enumerate(file_list):
            sentence_img = Image.open(os.path.join(visual_input_path, file))
            img_f = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img1_features.append(img_f)
        torch.save([torch.tensor(img1_features)], os.path.join(save_root_path, str(task_name) + ("_dev" if args.eval else "_train") + ".pt"))
    else:
        img1_features = []
        for idx, file in enumerate(file_list):
            sentence_img = Image.open(os.path.join(visual_input_path, file))
            img_f = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img1_features.append(img_f)

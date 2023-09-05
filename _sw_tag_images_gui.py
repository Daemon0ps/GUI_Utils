from __future__ import annotations
import os
import cv2
import nltk
import codecs
import keyring
import numpy as np
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm
import huggingface_hub
from config import CFG
import PySimpleGUI as sg
import onnxruntime as rt
from iptcinfo3 import IPTCInfo
from dataclasses import dataclass
from nltk.corpus import stopwords
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile, ImageOps, PngImagePlugin
from concurrent.futures import ThreadPoolExecutor, as_completed

nltk.download("stopwords")
sw = set(stopwords.words("english"))
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

@dataclass
class CFG:
    WINDOW: sg.Window = None
    CUDA_CHECK = str(rt.get_device())
    HF_TOKEN: str = keyring.get_password("hf", "hf_key")
    GEN_THRESH: float = 0.25
    FILE_PATH: str = ""
    MODEL_NAME: str = ""
    REPO: str = ""
    LIST_GEN: str = "List"
    THREAD_NUM: int = 8
    MODEL_NAME_LIST = ["MOAT", "SwinV2", "ConvNext", "ConvNextV2", "ViT"]
    REPO_LIST = [
        "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        "SmilingWolf/wd-v1-4-vit-tagger-v2",
    ]
    GENERAL_INDEX = []
    TAG_NAMES = []
    FILE_LIST = []
    LABEL_FILENAME: str = "selected_tags.csv"
    MODEL: rt.InferenceSession = None
    IMG_TYPES = [
        x.lower()
        for x in [
            "BLP",
            "BMP",
            "DIB",
            "BUFR",
            "CUR",
            "PCX",
            "DCX",
            "DDS",
            "PS",
            "EPS",
            "FIT",
            "FITS",
            "FLI",
            "FLC",
            "FTC",
            "FTU",
            "GBR",
            "GIF",
            "GRIB",
            "H5",
            "HDF",
            "PNG",
            "APNG",
            "JP2",
            "J2K",
            "JPC",
            "JPF",
            "JPX",
            "J2C",
            "ICNS",
            "ICO",
            "IM",
            "IIM",
            "TIF",
            "TIFF",
            "JFIF",
            "JPE",
            "JPG",
            "JPEG",
            "MPG",
            "MPEG",
            "MPO",
            "MSP",
            "PALM",
            "PCD",
            "PXR",
            "PBM",
            "PGM",
            "PPM",
            "PNM",
            "PSD",
            "BW",
            "RGB",
            "RGBA",
            "SGI",
            "RAS",
            "TGA",
            "ICB",
            "VDA",
            "VST",
            "WEBP",
            "WMF",
            "EMF",
            "XBM",
            "XPM",
            "NEF",
        ]
    ]
    REPLACE_LIST = {
        r"!": "",
        r"+_+": "",
        r"...": "",
        r"!": "",
        r"///": "",
        r"\\m/": "",
        r"\\||/": "",
        r"^^^": "",
        r":)": "",
        r":/": "",
        r":3": "",
        r":d": "",
        r":o": "",
        r":p": "",
        r":q": "",
        r"?": "",
        r"\\m/": "",
        r":3": "",
        r":<": "",
        r":i": "",
        r":o": "",
        r":p": "",
        r":q": "",
        r":t": "",
        r";)": "",
        r";d": "",
        r";o": "",
        r"?": "",
        r"\m/": "",
        r"\\||/": "",
        r"^^^": "",
        r"^_^": "",
        r"0_0": "",
        r"(o)_(o)": "",
        r"+_+": "",
        r"+_-": "",
        r"._.": "",
        r"<o>_<o>": "",
        r"<|>_<|>": "",
        r"=_=": "",
        r">_<": "",
        r"3_3": "",
        r"6_9": "",
        r">_o": "",
        r"@_@": "",
        r"^_^": "",
        r"o_o": "",
        r"u_u": "",
        r"x_x": "",
        r"|_|": "",
        r"||_||": "",
        r":>=": "",
    }

    def __post_init__(self):
        self.WINDOW = CFG.WINDOW
        self.HF_TOKEN = CFG.HF_TOKEN
        self.GEN_THRESH = CFG.GEN_THRESH
        self.FILE_PATH = CFG.FILE_PATH
        self.MODEL_NAME = CFG.MODEL_NAME
        self.REPO = CFG.REPO
        self.LIST_GEN = CFG.LIST_GEN
        self.THREAD_NUM = CFG.THREAD_NUM
        self.MODEL_NAME_LIST = CFG.MODEL_NAME_LIST
        self.REPO_LIST = CFG.REPO_LIST
        self.FILE_LIST = CFG.FILE_LIST
        self.TAG_NAMES = CFG.TAG_NAMES
        self.GENERAL_INDEX = CFG.GENERAL_INDEX
        self.LABEL_FILENAME = CFG.LABEL_FILENAME
        self.MODEL = CFG.MODEL
        super().__setattr__("attr_name", self)


def wfr() -> sg.Window:
    return CFG.WINDOW.refresh()


def infer(x):
    img = img_proc(x)
    t_name = str(f"{f_split(x)[0]}{f_split(x)[1]}.txt")
    input_name = CFG.MODEL.get_inputs()[0].name
    label_name = CFG.MODEL.get_outputs()[0].name
    probs = CFG.MODEL.run([label_name], {input_name: img})[0]
    labels = list(zip(CFG.TAG_NAMES, probs[0].astype(float)))
    general_names = [labels[i] for i in CFG.GENERAL_INDEX]
    tag_list = dict([[x[0], x[1]] for x in general_names if x[1] > CFG.GEN_THRESH])
    txt_write(t_name, tag_list)


def tag_proc():
    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
        status_bar = statbar(len(CFG.FILE_LIST), r"Image Tagging")
        wfr()
        futures = [executor.submit(infer, i) for i in CFG.FILE_LIST]
        for _ in as_completed(futures):
            status_bar.update(n=1)
            wfr()
        status_bar.close()
        CFG.MODEL = None


def iptc_proc():
    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
        status_bar = statbar(len(CFG.FILE_LIST), r"Image Keywords to EXIF/IPTC ")
        futures = [executor.submit(iptc_tag_proc, i) for i in CFG.FILE_LIST]
        for _ in as_completed(futures):
            status_bar.update(n=1)
            wfr()
        status_bar.close()
    return int(1)


def load_labels():
    tag_path = hf_hub_download(
        CFG.REPO, CFG.LABEL_FILENAME, use_auth_token=CFG.HF_TOKEN
    )
    wfr()
    df = pd.read_csv(tag_path)
    CFG.TAG_NAMES = df["name"].tolist()
    CFG.GENERAL_INDEX = list(np.where(df["category"] == 0)[0])


def load_model() -> rt.InferenceSession:
    path = huggingface_hub.hf_hub_download(
        CFG.REPO, "model.onnx", use_auth_token=CFG.HF_TOKEN
    )
    wfr()
    CFG.MODEL = rt.InferenceSession(path, providers=["CUDAExecutionProvider"])


def core():
    if cuda_check() == "FAIL":
        return 0
    list_gen()
    load_labels()
    load_model()
    tag_proc()
    iptc_proc()


def statbar(tot: int, desc: str):
    l_bar = r"{desc}: {percentage:3.0f}%|"
    r_bar = r"| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}]"
    bar = r"{rate_fmt}{postfix}]"
    status_bar = tqdm(
        total=tot,
        desc=desc,
        bar_format=f"{l_bar}{bar}{r_bar}{chr(10)}",
        maxinterval=0.2,
        mininterval=0.1,
    )
    return status_bar


def f_split(f: str) -> list:
    return [
        # base directory
        f[: len(f) - (f[::-1].find("/")) :].lower(),
        # filename
        f[len(f) - (f[::-1].find("/")) : (len(f)) - 1 - len(f[-(f[::-1].find(".")) :])],
        # file extension
        f[-(f[::-1].find(".")) :].lower(),
    ]


def txt_write(t_name: str, tag_list: dict):
    tag_list = list(
        sorted(
            list(x for x in tag_list.keys()), key=lambda x: tag_list[x], reverse=True
        )
    )
    tags = [
        x
        for x in list(
            str(x).replace("_", chr(32))
            for x in map(
                lambda x: str(x).strip().lower().replace(x, CFG.REPLACE_LIST[x])
                if len(x) > 0 and str(x).strip().lower() in CFG.REPLACE_LIST.keys()
                else x,
                tag_list,
            )
        )
    ]
    l_w = (
        lambda x: str(x).capitalize()
        if not set(x).issubset(sw)
        else str(x).capitalize()
    )
    tag_write = str(
        ", ".join(
            t
            for t in [
                " ".join(l_w(s) for s in str(w).split(chr(32)))
                for w in np.unique(np.array(tags)).tolist()
            ]
        )
    )
    with open(t_name, "wt") as fi:
        fi.write(tag_write)
        fi.close()
    return


def iptc_tag_proc(file: str) -> None:
    t_name = file[: (len(file)) - 1 - len(file[-(file[::-1].find(".")) :])] + ".txt"
    if os.path.isfile(t_name):
        with open(t_name, "rt") as fi:
            txt_data = fi.read()
        txt_data = bytes(txt_data, "utf-8")
        txt_data = codecs.decode(
            unicodedata.normalize("NFKD", codecs.decode(txt_data)).encode(
                "ascii", "ignore"
            )
        )
        txt_list = str(txt_data).split(",")
        tags = [
            x
            for x in list(
                str(x).replace("_", chr(32))
                for x in map(
                    lambda x: str(x).strip().lower().replace(x, CFG.REPLACE_LIST[x])
                    if len(x) > 0 and str(x).strip().lower() in CFG.REPLACE_LIST.keys()
                    else x,
                    txt_list,
                )
            )
        ]
        l_w = (
            lambda x: str(x).capitalize()
            if not set(x).issubset(sw)
            else str(x).capitalize()
        )
        tag_write = str(
            ",".join(
                t
                for t in [
                    " ".join(l_w(s) for s in str(w).split(chr(32)))
                    for w in np.array(
                        [
                            y[np.argsort(z)].astype("str").flatten().tolist()
                            for y, z in [np.unique(np.array(tags), return_index=True)]
                        ]
                    )[0]
                ]
            )
        )
        iptc_tags = [
            codecs.encode(x.strip(), encoding="utf-8") for x in tag_write.split(",")
        ]
        iptc_info = IPTCInfo(file, force=True)
        iptc_info["keywords"] = iptc_tags
        iptc_info.save()
        with open(t_name, "wt") as fi:
            fi.write(tag_write)
            fi.close()


def img_proc(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    size = max(img.shape[0:2])
    pad_x = size - img.shape[1]
    pad_y = size - img.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    img = np.pad(
        img,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    interp = cv2.INTER_CUBIC
    img = cv2.resize(img, (448, 448), interpolation=interp)
    image = np.expand_dims(img, 0)
    image = image.astype(np.float32)
    return image


def list_gen():
    CFG.FILE_LIST = []
    if CFG.LIST_GEN == "WALK":
        for r, d, f in os.walk(CFG.FILE_PATH[:-1:]):
            for file in f:
                if ((file[-(file[::-1].find(".")) :]).lower()) in CFG.IMG_TYPES:
                    CFG.FILE_LIST.append(os.path.join(r, file))
    elif CFG.LIST_GEN == "LIST":
        CFG.FILE_LIST = [
            CFG.FILE_PATH + f
            for f in os.listdir(CFG.FILE_PATH[:-1:])
            if os.path.isfile(CFG.FILE_PATH + f)
            and f[-(f[::-1].find(".")) :] in CFG.IMG_TYPES
        ]
    print("Number of Image Files: ", len(CFG.FILE_LIST))


def get_hf_key() -> str:
    CFG.HF_TOKEN = keyring.get_password("hf", "hf_key")
    return CFG.HF_TOKEN


def set_key(s: str) -> None:
    keyring.set_password("hf", "hf_key", s)
    CFG.HF_TOKEN = keyring.get_password("hf", "hf_key")
    print(f"Keyring updated ('hf','hf_key'): {keyring.get_password('hf','hf_key')}")


def cuda_check() -> str:
    try:
        assert CFG.CUDA_CHECK == "GPU"
    except AssertionError:
        return "FAIL"
    finally:
        return "PASS"


if __name__ == "__main__":
    VALIDATION = False
    sg.set_options(text_justification="right")
    MODEL_NAME_LIST = CFG.MODEL_NAME_LIST
    params = [
        [
            sg.Text(
                "Image Directory",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 12",
                size=(30, 1),
            )
        ],
        [
            sg.InputText(
                CFG.FILE_PATH,
                background_color="#000000",
                text_color="#FF8C00",
                key="-FOLDER-",
                font="Consolas 12",
                size=(89, 1),
            ),
            sg.FolderBrowse(font="Consolas 12", enable_events=True),
        ],
        [
            sg.Text(
                "List Method:  Walk = Recursive,  List = Directory Only",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 12",
                size=(89, 1),
            )
        ],
        [
            sg.Drop(
                values=("LIST", "WALK"),
                background_color="#000000",
                text_color="#FF8C00",
                font="Consolas 12",
                key="-LIST_GEN-",
                default_value="LIST",
            )
        ],
        [
            sg.Text(
                "Accuracy Threshold",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 12",
                size=(30, 1),
            )
        ],
        [
            sg.Input(
                default_text=str(CFG.GEN_THRESH),
                background_color="#000000",
                text_color="#FF8C00",
                key="-THRESH-",
                font="Consolas 12",
                size=(10, 1),
            )
        ],
        [
            sg.Text(
                "Multithreading",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 12",
                size=(30, 1),
            )
        ],
        [
            sg.Input(
                default_text=str(CFG.THREAD_NUM),
                background_color="#000000",
                text_color="#FF8C00",
                key="-THREADS-",
                font="Consolas 12",
                size=(8, 1),
            )
        ],
        [
            sg.Text(
                "HuggingFace API Token",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 12",
                size=(30, 1),
            )
        ],
        [
            sg.InputText(
                get_hf_key(),
                background_color="#000000",
                text_color="#FF8C00",
                key="-HF_TOKEN-",
                font="Consolas 12",
                size=(40, 1),
                password_char="*",
            ),
            sg.Button("SET_TOKEN"),
            sg.Text(
                'Keyring: "hf","hf_key"',
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 12",
            ),
        ],
    ]

    models = [
        [
            sg.Rad(
                "MOAT",
                "model",
                size=(10, 1),
                background_color="#000000",
                font="Consolas 12",
                text_color="#FF8C00",
            )
        ],
        [
            sg.Rad(
                "SwinV2",
                "model",
                size=(10, 1),
                background_color="#000000",
                font="Consolas 12",
                text_color="#FF8C00",
            )
        ],
        [
            sg.Rad(
                "ConvNext",
                "model",
                size=(10, 1),
                background_color="#000000",
                font="Consolas 12",
                text_color="#FF8C00",
            )
        ],
        [
            sg.Rad(
                "ConvNextV2",
                "model",
                size=(10, 1),
                background_color="#000000",
                font="Consolas 12",
                text_color="#FF8C00",
            )
        ],
        [
            sg.Rad(
                "ViT",
                "model",
                default=True,
                size=(10, 1),
                background_color="#000000",
                font="Consolas 12",
                text_color="#FF8C00",
            )
        ],
    ]

    tagging_layout = [
        [
            sg.Frame(
                "Parameters",
                params,
                background_color="#000000",
                font="Consolas 12",
                title_color="#ffffff",
            )
        ],
        [
            sg.Frame(
                "Models",
                models,
                background_color="#000000",
                font="Consolas 12",
                title_color="#ffffff",
            ),
            sg.Output(
                echo_stdout_stderr=True,
                size=(122, 15),
                text_color="#FF8C00",
                key="-OUTPUT-",
                background_color="#000000",
                font="Consolas 8",
            ),
        ],
        [sg.Button("Start"), sg.Button("Exit")],
    ]

    layout = [
        [
            sg.TabGroup(
                [
                    [
                        sg.Tab("Image Tagging", tagging_layout),
                    ]
                ]
            )
        ]
    ]

    sg.set_options(text_justification="left")

    window = sg.Window(
        "Image Tagger",
        layout,
        font="Consolas 12",
        background_color="#000000",
        use_custom_titlebar=False,
        titlebar_background_color="#000000",
        titlebar_text_color="#ffffff",
        keep_on_top=True,
    )

    CFG.WINDOW = window

    while True:
        VALIDATION = True
        event, values = window.read()
        window["-OUTPUT-"].Update("")
        if event in (sg.WIN_CLOSED, "EXIT"):
            break
        if event == "SET_TOKEN":
            _ = str(values["-HF_TOKEN-"])
            if (
                _[:3].lower() == "hf_"
                or _[:4].lower() == "api_"
                and len(_) > 4
                and _ != "None"
            ):
                set_key(_)
            else:
                print("This does not appear to be a valid HF API Access Token")
                continue
        elif event == "Start":
            CFG.LIST_GEN = values["-LIST_GEN-"]
            CFG.MODEL_NAME = CFG.MODEL_NAME_LIST[
                list(
                    map(lambda x: values[x] if values[x] == True else None, range(5))
                ).index(True)
            ]
            CFG.REPO = CFG.REPO_LIST[
                list(
                    map(lambda x: values[x] if values[x] == True else None, range(5))
                ).index(True)
            ]
            print(CFG.MODEL_NAME)
            if os.path.isdir(values["-FOLDER-"]):
                print("Folder Check: PASS")
                CFG.FILE_PATH = f"{values['-FOLDER-']}{chr(47)}"
            elif not os.path.isdir(values["-FOLDER-"]):
                print("Folder Check: FAIL")
                print("Please select a valid folder")
                CFG.FILE_PATH = ""
                VALIDATION = False
                continue
            try:
                type(float(values["-THRESH-"]))
                CFG.GEN_THRESH = float(values["-THRESH-"])
                if CFG.GEN_THRESH < 1.0 and CFG.GEN_THRESH > 0.0:
                    print("Accuracy Threshold Check: PASS")
                else:
                    print("Accuracy Threshold Check: FAIL")
                    CFG.GEN_THRESH = 0.35
                    VALIDATION = False
            except ValueError as e:
                print(e)
                CFG.GEN_THRESH = 0.35
                VALIDATION = False
                continue
            try:
                type(int(values["-THREADS-"]))
                tn = int(values["-THREADS-"])
                CFG.THREAD_NUM = tn
            except ValueError as e:
                print("Thread Number Check: FAIL")
                print(e)
                print("Please select a valid integer")
                CFG.THREAD_NUM = 1
                VALIDATION = False
                continue
            if VALIDATION:
                print("FIELD VALIDATION: PASS")
                window.refresh()
                ret = core()
                if ret == 0:
                    print("Halted.")
                    continue
                elif ret == 1:
                    print("Run Complete")
                    continue
            elif not VALIDATION:
                print("FIELD VALIDATION: FAIL")
                continue
        elif event == "Exit":
            break
        elif event == "Browse":
            print("wat")
    window.close()

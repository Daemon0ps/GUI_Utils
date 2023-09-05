from __future__ import annotations
import os
import re
import os
import sys
import cv2
import time
import nltk
import piexif
import codecs
import shutil
import hashlib
import threading
import numpy as np
import pillow_heif
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm
from typing import List
import PySimpleGUI as sg
from piexif import TYPES
import PySimpleGUI as sg
from datetime import datetime
from PIL.ExifTags import TAGS
from iptcinfo3 import IPTCInfo
from unidecode import unidecode
from nltk.corpus import stopwords
from dataclasses import dataclass
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageFile, ImageOps
from traceback_with_variables import activate_by_import
from concurrent.futures import ThreadPoolExecutor, as_completed


nltk.download("stopwords")
Image.MAX_IMAGE_PIXELS = None
sw = set(stopwords.words("english"))
ImageFile.LOAD_TRUNCATED_IMAGES = True


# pip install -U git+https://github.com/Daemon0ps/iptcinfo3.git@master
# pip install -e git+https://github.com/hMatoba/Piexif.git@6135c61db5179dd0e48fc6a8cbb03acad46e7461#egg=Piexif


# todo

# pnginfo dict mapping
# exif dict mapping - I hate Piexif, but done?'ish?
# HEIC->JPG
# auto os.symlinks (ckpt/pt/safetensors)
# civit.ai -> keywords lookup
# bing download
# img_crop
# sprite_rip


@dataclass(frozen=True)
class win:
    dow: sg.Window = None

    def __post_init__(self):
        self.dow = win.dow


@dataclass(frozen=False)
class CFG:
    SP_CHG: str = ""
    CONV_TYPE: str = ""
    TQ_DESC: str = ""
    RUN_OPT: str = ""
    RESIZE_W: str = ""
    RESIZE_H: str = ""
    FILE_PATH: str = ""
    SAVE_PATH: str = ""
    VALIDATION: str = ""
    BROWSE_FILE_PATH: str = ""
    BROWSE_SAVE_PATH: str = ""
    A_TO_Z: bool = None
    BACKUP: bool = None
    RECURSE: bool = None
    DELCONV: bool = None
    EXIF_TRANS: bool = None
    THREAD_NUM: int = 8
    WINDOW: sg.Window = None
    FUNC: object = None
    NOW = lambda: datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S")
    STD_IMG_TYPES = [
        "blp",
        "bmp",
        "dib",
        "bufr",
        "cur",
        "pcx",
        "dcx",
        "dds",
        "ps",
        "eps",
        "fit",
        "fits",
        "fli",
        "flc",
        "ftc",
        "ftu",
        "gbr",
        "gif",
        "grib",
        "h5",
        "hdf",
        "png",
        "apng",
        "jp2",
        "j2k",
        "jpc",
        "jpf",
        "jpx",
        "j2c",
        "icns",
        "ico",
        "im",
        "iim",
        "tif",
        "tiff",
        "jfif",
        "jpe",
        "jpg",
        "jpeg",
        "mpg",
        "mpeg",
        "mpo",
        "msp",
        "palm",
        "pcd",
        "pdf",
        "pxr",
        "pbm",
        "pgm",
        "ppm",
        "pnm",
        "psd",
        "bw",
        "rgb",
        "rgba",
        "sgi",
        "ras",
        "tga",
        "icb",
        "vda",
        "vst",
        "webp",
        "wmf",
        "emf",
        "xbm",
        "xpm",
    ]
    STR_NOW = NOW()
    IMG_TYPES = []
    RUN_OPT_LIST = []
    OPT_VALS = [
        "OPT_IPTC_TXT",
        "OPT_TXT_IPTC",
        "OPT_REN_MD5",
        "OPT_EXIF_TXT",
        "OPT_TXT_EXIF",
        "OPT_CONVERT",
        "OPT_PNGINFO_TXT",
        "OPT_TXT_PNGINFO",
        "OPT_STRIP",
        "OPT_RESIZE",
        "OPT_HEIC",
        "OPT_RM_PAD",
        "OPT_SQ_PAD",
    ]
    FILE_LIST = []
    FILE_LIST = []
    THREAD_DICT = {}

    def __post_init__(self):
        self.NOW = CFG.NOW
        self.FUNC = CFG.FUNC
        self.SP_CHG = CFG.SP_CHG
        self.CONV_TYPE = CFG.CONV_TYPE
        self.TYPE_W = CFG.TYPE_W
        self.A_TO_Z = CFG.A_TO_Z
        self.RECURSE = CFG.RECURSE
        self.STR_NOW = CFG.STR_NOW
        self.TQ_DESC = CFG.TQ_DESC
        self.RUN_OPT = CFG.RUN_OPT
        self.BACKUP = CFG.BACKUP
        self.RESIZE_W = CFG.RESIZE_W
        self.RESIZE_H = CFG.RESIZE_H
        self.DELCONV = CFG.DELCONV_N
        self.IMG_TYPES = CFG.IMG_TYPES
        self.STD_IMG_TYPES = CFG.STD_IMG_TYPES
        self.FILE_PATH = CFG.FILE_PATH
        self.SAVE_PATH = CFG.SAVE_PATH
        self.RECURSE = CFG.RECURSE
        self.FILE_LIST = CFG.FILE_LIST
        self.VALIDATION = CFG.VALIDATION
        self.THREAD_NUM = CFG.THREAD_NUM
        self.THREAD_DICT = CFG.THREAD_DICT
        self.RUN_OPT_LIST = CFG.RUN_OPT_LIST
        self.OPT_VALS = CFG.OPT_VALS
        self.EXIF_TRANS = CFG.EXIF_TRANS
        self.BROWSE_FILE_PATH = CFG.BROWSE_FILE_PATH
        self.BROWSE_SAVE_PATH = CFG.BROWSE_SAVE_PATH
        super().__setattr__("attr_name", self)


pillow_heif.register_heif_opener()


def wfr() -> sg.Window:
    return win.dow.refresh()


def pulse(window: sg.Window) -> None:
    i = 0
    while True:
        time.sleep(1)
        window.write_event_value("-THREAD-", (threading.current_thread().name, i))
        i += 1


def statbar(tot: int, desc: str) -> tqdm:
    l_bar = "{desc}: {percentage:3.0f}%|"
    r_bar = "| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] "
    bar = "{rate_fmt}{postfix}]"
    status_bar = tqdm(total=tot, desc=desc, bar_format=f"{l_bar}{bar}{r_bar}{chr(10)}")
    return status_bar


def f_split(f: str) -> list[str]:
    return [
        f[: len(f) - (f[::-1].find("/")) :].lower(),  # base directory
        f[
            len(f) - (f[::-1].find("/")) : (len(f)) - 1 - len(f[-(f[::-1].find(".")) :])
        ],  # filename
        f[-(f[::-1].find(".")) :].lower(),  # file extension
    ]


def opt_iptc_txt(file: str) -> None:
    tags = []
    txt_write = ""
    f = f_split(file)
    iptc_info = IPTCInfo(file, force=True)
    tags = [
        codecs.decode(x, encoding="utf-8").strip().lower()
        for x in iptc_info["keywords"]
    ]
    l_w = (
        lambda x: str(x).capitalize()
        if not set(x).issubset(sw)
        else str(x).capitalize()
    )
    if CFG.A_TO_Z:
        txt_write = str(
            ", ".join(
                t
                for t in [
                    " ".join(l_w(s) for s in str(w).split(chr(32)))
                    for w in np.unique(np.array(tags)).tolist()
                ]
            )
        )
    elif not CFG.A_TO_Z:
        txt_write = str(
            ", ".join(
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
    with open(f"{CFG.SAVE_PATH}{f[1]}.txt", "wt", encoding="utf-8") as fi:
        fi.write(txt_write)


def iptc_write(file: str, tags: list[bytes]):
    iptc_info = IPTCInfo(file, force=True)
    iptc_info["keywords"] = tags
    iptc_info.save(options=["overwrite"])


def opt_txt_iptc(file):
    txt_data = ""
    txt_list = []
    txt_kw = []
    tags = []
    f = f_split(file)
    with open(f"{f[0]}{f[1]}.txt", "rt") as fi:
        txt_data = fi.read()
    txt_list = txt_data.split(",")
    l_w = (
        lambda x: str(x).capitalize()
        if not set(x).issubset(sw)
        else str(x).capitalize()
    )
    if CFG.A_TO_Z:
        txt_kw = [
            t.strip()
            for t in [
                " ".join(l_w(s) for s in str(w).split(chr(32)))
                for w in np.unique(np.array(txt_list)).tolist()
            ]
        ]
    if not CFG.A_TO_Z:
        txt_kw = [
            t.strip()
            for t in [
                " ".join(l_w(s) for s in str(w).split(chr(32)))
                for w in np.array(
                    [
                        y[np.argsort(z)].astype("str").flatten().tolist()
                        for y, z in [np.unique(np.array(txt_list), return_index=True)]
                    ]
                )[0]
            ]
        ]
    tags = [codecs.encode(str(x), encoding="utf-8") for x in txt_kw]
    return iptc_write(file, tags)


def opt_ren_md5(file: str) -> None:
    f = f_split(file)
    with open(file, "rb") as fi:
        file_bytes = fi.read()
    md5_calc = str(hashlib.md5(file_bytes).hexdigest()).lower()
    os.rename(
        src=str(f"{f[0]}/{f[1]}.{f[2]}"),
        dst=str(f"{CFG.SAVE_PATH}{md5_calc}.{f[2]}"),
    )
    if os.path.isfile(f"{f[0]}{f[1]}.txt"):
        os.rename(
            src=str(f"{f[0]}/{f[1]}.txt"),
            dst=str(f"{CFG.SAVE_PATH}{md5_calc}.txt"),
        )


def opt_exif_txt(file: str):
    f = f_split(file)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = Image.fromarray(file)
    exif_info = img.getexif()
    exif_tags = {TAGS.get(k, k): v for k, v in exif_info.items()}
    with open(f"{f[0]}{f[1]}.txt", "wt") as fi:
        fi.write(chr(10).join(f"{k}:{v}" for k, v in exif_tags.items()))
    img.close()


def t_ex(exl: list[str], pe_T: str):
    def _b(b: str) -> bytes:
        return bytes(b, encoding="utf-8")

    def _i(i: str) -> int:
        i = i.split(".")[0]
        i = i.split(",")[0]
        i = "".join(
            c for c in i if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        )
        return int(i)

    def _f(f: str) -> float:
        f = "".join(
            c for c in f if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]
        )
        if f.find(".") == -1:
            f = f + ".0"
            return float(f)

    def _t(t: str) -> tuple:
        if len(t.split()) == 1:
            print(t)
            t = t.split(".")[0]
            t = t.split(",")[0]
            t = "".join(
                c for c in t if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
            )
            t = int(t)
            return t
        else:
            xl = []
            for x in t.split():
                x = x.split(".")[0]
                x = x.split(",")[0]
                x = "".join(
                    c
                    for c in x
                    if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
                )
                x = int(x)
                xl.append(x)
                ret = tuple(int(x) for x in xl)
            return ret

    def _s(s: str) -> str:
        return unidecode(s)

    lambda_dict = {
        1: lambda x: _i(x),
        2: lambda x: _s(x),
        3: lambda x: _i(x),
        4: lambda x: _i(x),
        5: lambda x: _t(x),
        6: lambda x: _i(x),
        7: lambda x: _t(x),
        8: lambda x: _i(x),
        9: lambda x: _i(x),
        10: lambda x: _t(x),
        11: lambda x: _f(x),
        12: lambda x: _f(x),
    }
    if pe_T == "I0":
        ex_k = piexif.ImageIFD.__dict__.get(exl[0])
        ex_v = lambda_dict[
            piexif._exif.TAGS["Image"][piexif.ImageIFD.__dict__.get(exl[0])]["type"]
        ](exl[1])
        return (ex_k, ex_v)
    if pe_T == "EI":
        ex_k = piexif.ImageIFD.__dict__.get(exl[0])
        ex_v = lambda_dict[
            piexif._exif.TAGS["Exif"][piexif.ImageIFD.__dict__.get(exl[0])]["type"]
        ](exl[1])
        return (ex_k, ex_v)
    if pe_T == "GPS":
        ex_k = piexif.ImageIFD.__dict__.get(exl[0])
        ex_v = lambda_dict[
            piexif._exif.TAGS["GPS"][piexif.ImageIFD.__dict__.get(exl[0])]["type"]
        ](exl[1])
        return (ex_k, ex_v)
    if pe_T == "IOp":
        ex_k = piexif.ImageIFD.__dict__.get(exl[0])
        ex_v = lambda_dict[
            piexif._exif.TAGS["Interop"][piexif.ImageIFD.__dict__.get(exl[0])]["type"]
        ](exl[1])
        return (ex_k, ex_v)


def opt_txt_exif(file: str) -> None:
    txt_data = ""
    txt_list = []
    txt_kw = []
    tags = []
    exif_info: Image.Exif
    f = f_split(file)
    if not os.path.isfile(f"{f[0]}{f[1]}.txt"):
        return
    with open(f"{f[0]}{f[1]}.txt", "rt") as fi:
        txt_data = fi.read()
    if len(txt_data) == 0:
        return
    if list(txt_data).count(",") > list(txt_data).count(":"):
        txt_list = txt_data.split(",")
        l_w = (
            lambda x: str(x).capitalize()
            if not set(x).issubset(sw)
            else str(x).capitalize()
        )
        if CFG.A_TO_Z:
            txt_kw = [
                t.strip()
                for t in [
                    " ".join(l_w(s) for s in str(w).split(chr(32)))
                    for w in np.unique(np.array(txt_list)).tolist()
                ]
            ]
        if not CFG.A_TO_Z:
            txt_kw = [
                t.strip()
                for t in [
                    " ".join(l_w(s) for s in str(w).split(chr(32)))
                    for w in np.array(
                        [
                            y[np.argsort(z)].astype("str").flatten().tolist()
                            for y, z in [
                                np.unique(np.array(txt_list), return_index=True)
                            ]
                        ]
                    )[0]
                ]
            ]
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = Image.fromarray(img)
        exif_info = img.getexif()
        XPKeywords = 0x9C9E
        UserComment = 0x9286
        tags = codecs.encode(",".join(str(x) for x in txt_kw), encoding="utf-16")
        exif_info[XPKeywords] = tags
        exif_info[UserComment] = tags
        img.save(file, exif=exif_info)
        img.close()
    elif list(txt_data).count(":") > list(txt_data).count(","):
        txt_data.replace(chr(13) + chr(10), chr(10))
        txt_list = [
            [x[: x.find(":")], x[-(len(x) - len(x[: x.find(":")])) + 1 :]]
            for x in txt_data.split(chr(10))
        ]
        for x in txt_list:
            print(x)
        pe_I0_dict = {
            k: v
            for k, v in [
                t_ex(x, "I0")
                for x in txt_list
                if x[0] in piexif.ImageIFD.__dict__.keys()
            ]
        }
        pe_EI_dict = {
            k: v
            for k, v in [
                t_ex(x, "EI")
                for x in txt_list
                if x[0] in piexif.ExifIFD.__dict__.keys()
            ]
        }
        pe_GPS_dict = {
            k: v
            for k, v in [
                t_ex(x, "GPS")
                for x in txt_list
                if x[0] in piexif.GPSIFD.__dict__.keys()
            ]
        }
        pe_IOp_dict = {
            k: v
            for k, v in [
                t_ex(x, "IOp")
                for x in txt_list
                if x[0] in piexif.InteropIFD.__dict__.keys()
            ]
        }
        exif_dict = {}
        if len(pe_I0_dict) > 0:
            exif_dict["0th"] = pe_I0_dict
        if len(pe_EI_dict) > 0:
            exif_dict["Exif"] = pe_EI_dict
        if len(pe_GPS_dict) > 0:
            exif_dict["GPS"] = pe_GPS_dict
        if len(pe_IOp_dict) > 0:
            exif_dict["1st"] = pe_IOp_dict
        print(exif_dict)
        exif_bytes = piexif.dump(exif_dict, strict=False)
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = Image.fromarray(img)
        img.save("./test.jpg", exif=exif_bytes)
        img.close()

    return


def opt_pnginfo_txt(file: str) -> None:
    f = f_split(file)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = Image.fromarray(img)
    if len(img.text) < 1:
        return
    with open(f"{f[0]}{f[1]}.txt", "wt") as fi:
        fi.write(chr(10).join(f"{k}:{v}" for k, v in img.text.items()))
    return


def opt_txt_pnginfo(file: str):
    txt_data = ""
    txt_list = []
    txt_kw = []
    f = f_split(file)
    if not os.path.isfile(f"{f[0]}{f[1]}.txt"):
        return
    with open(f"{f[0]}{f[1]}.txt", "rt") as fi:
        txt_data = fi.read()
    if len(txt_data) == 0:
        return
    txt_list = txt_data.split(",")
    l_w = (
        lambda x: str(x).capitalize()
        if not set(x).issubset(sw)
        else str(x).capitalize()
    )
    if CFG.A_TO_Z:
        txt_kw = [
            t.strip()
            for t in [
                " ".join(l_w(s) for s in str(w).split(chr(32)))
                for w in np.unique(np.array(txt_list)).tolist()
            ]
        ]
    if not CFG.A_TO_Z:
        txt_kw = [
            t.strip()
            for t in [
                " ".join(l_w(s) for s in str(w).split(chr(32)))
                for w in np.array(
                    [
                        y[np.argsort(z)].astype("str").flatten().tolist()
                        for y, z in [np.unique(np.array(txt_list), return_index=True)]
                    ]
                )[0]
            ]
        ]
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = Image.fromarray(img)
    exif_info = img.getexif()
    png_info = PngInfo()
    png_info.add_text("Tags", ",".join(str(x) for x in txt_kw))
    img.save(file, pnginfo=png_info, exif=exif_info)
    img.close()
    return


def img_info_save(f: list[str], img: Image) -> None:
    tags = []
    exif_info: Image.Exif = None
    iptc_info: list[bytes] = []
    png_info: dict = {}
    ext = f[2] if CFG.CONV_TYPE == "" else CFG.CONV_TYPE
    if f[2].lower() in [
        "jpg",
        "jpeg",
        "tiff",
        "png",
        "jp2",
        "pgf",
        "miff",
        "hdp",
        "psp",
        "xcf",
    ]:
        exif_info = img.getexif()
        if len(exif_info.items()) > 0:
            e = True
        else:
            e = False
    if f[2].lower() in [
        "jpg",
        "jpeg",
        "tiff",
        "png",
        "miff",
        "ps",
        "pdf",
        "psd",
        "xcf",
        "dng",
    ]:
        iptc_info = IPTCInfo(f"{f[0]}{f[1]}.{ext}", force=True)
        tags = [x for x in iptc_info["keywords"]]
        del iptc_info
        if len(tags) > 0:
            i = True
        else:
            i = False
    if f[2].lower() == "png":
        if len(img.text) > 0:
            p = True
            png_info = PngInfo()
            for k, v in img.text.items():
                png_info.add_text(str(k), str(v))
    if e and p:
        img.save(f"{f[0]}{f[1]}.{ext}", format=ext, exif=exif_info, pnginfo=png_info)
        img.close()
    elif e and not p:
        img.save(f"{f[0]}{f[1]}.{ext}", format=ext, exif=exif_info)
        img.close()
    elif p and not e:
        img.save(f"{f[0]}{f[1]}.{ext}", format=ext, pnginfo=png_info)
        img.close()
    else:
        img.save(f"{f[0]}{f[1]}.{ext}", format=ext)
        img.close()
    if i:
        iptc_info = IPTCInfo(f"{f[0]}{f[1]}.{ext}", force=True)
        iptc_info["keywords"] = tags


def opt_convert(file: str, chg_ext: str) -> tuple(list[str], Image):
    f = f_split(file)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = Image.fromarray(img)
    return img_info_save(f, img)


def opt_strip(file: str) -> None:
    img = cv2.imread(file)
    cv2.imwrite(file, img)
    return


def opt_resize(file: str, w: int, h: int) -> None:
    f = f_split(file)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = ImageOps.contain(Image.fromarray(img), (w, h), Image.LANCZOS)
    return img_info_save(f, img)


def bk_zip() -> str:
    def D_rm_spec(s) -> str:
        return (
            re.sub("\s+", "_", (re.sub("[^a-zA-z0-9\s]", "_", unidecode(s))))
            .strip()
            .lower()
        )

    print("CurDir: ", os.path.curdir + chr(47))
    r = os.path.curdir
    print("cd ", CFG.FILE_PATH)
    os.chdir(CFG.FILE_PATH)
    s = shutil.make_archive(
        base_name=f"{D_rm_spec(CFG.FILE_PATH)}{chr(95)}{CFG.STR_NOW}",
        base_dir=CFG.FILE_PATH,
        root_dir=CFG.FILE_PATH,
        format="zip",
        verbose=True,
    )
    print("Saved as: ", s)
    print("cd ", r + chr(47))
    os.chdir(r)
    print("CurDir: ", os.path.curdir + chr(47))
    return s


def window_main() -> sg.Window:
    src_dir_pn = [
        sg.Text(
            "File Path:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(10, 1),
        ),
        sg.InputText(
            "",
            background_color="#121212",
            text_color="#FF8C00",
            key="-FILE_PATH-",
            font="Consolas 10",
            size=(50, 1),
            justification="left",
        ),
        sg.FolderBrowse(
            font="Consolas 10",
            enable_events=True,
            button_color="#FF8C00 on #121212",
            key="-BROWSE_FILE_PATH-",
        ),
    ]

    save_dir_pn = [
        sg.Text(
            "Save Path: ",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(10, 1),
        ),
        sg.InputText(
            "Leave this or keep empty to use File Path",
            background_color="#121212",
            text_color="#FF8C00",
            key="-SAVE_PATH-",
            font="Consolas 10",
            size=(50, 1),
            justification="left",
            enable_events=True,
        ),
        sg.FolderBrowse(
            font="Consolas 10",
            enable_events=True,
            button_color="#FF8C00 on #121212",
            key="-BROWSE_SAVE_PATH-",
        ),
    ]

    recurse_pn = [
        sg.Text(
            "Recurse SubDirs",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(15, 1),
            justification="left",
            pad=((0, 0), (4, 4)),
        ),
        sg.Rad(
            "Y",
            "Recurse",
            size=(1, 1),
            background_color="#000000",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-RECURSE_Y-",
        ),
        sg.Rad(
            "N",
            "Recurse",
            default=True,
            size=(1, 1),
            background_color="#000000",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-RECURSE_N-",
        ),
    ]

    dir_bk_pn = [
        sg.Text(
            "Back-Up Files before Action",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(38, 1),
            justification="right",
            pad=0,
        ),
        sg.Rad(
            "Y",
            "Back-Up",
            size=(1, 1),
            background_color="#000000",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-BACKUP_Y-",
        ),
        sg.Rad(
            "N",
            "Back-Up",
            default=True,
            size=(1, 1),
            background_color="#000000",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-BACKUP_N-",
        ),
    ]

    alphabet_pn = [
        sg.Text(
            "Alphabetize Tags:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(16, 1),
        ),
        sg.Rad(
            "Y",
            "Alphabetize",
            default=True,
            size=(1, 1),
            background_color="#000000",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-A_TO_Z_Y-",
        ),
        sg.Rad(
            "N",
            "Alphabetize",
            size=(1, 1),
            background_color="#000000",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-A_TO_Z_N-",
        ),
    ]

    exif_trans_pn = [
        sg.Text(
            "exif_transpose?",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(16, 1),
        ),
        sg.Rad(
            "Y",
            "exif_trans",
            default=True,
            size=(1, 1),
            background_color="#010101",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-EXIF_TRANS_Y-",
        ),
        sg.Rad(
            "N",
            "exif_trans",
            size=(1, 1),
            background_color="#010101",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-EXIF_TRANS_N-",
        ),
    ]

    save_type_pn = [
        sg.Text(
            "Save as Type:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(16, 1),
        ),
        sg.Drop(
            values=("JPG", "PNG", "BMP", "WEBP"),
            default_value="",
            background_color="#121212",
            text_color="#FF8C00",
            font="Consolas 10",
            key="-TYPE_D-",
        ),
        sg.Text(
            "or:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(2, 1),
        ),
        sg.Input(
            "",
            background_color="#121212",
            text_color="#FF8C00",
            font="Consolas 10",
            key="-TYPE_W-",
            size=(4, 1),
            enable_events=True,
        ),
    ]

    type_to_type = [
        sg.Text(
            "Convert",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(8, 1),
            justification="left",
        ),
        sg.Text(
            "From:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(8, 1),
            justification="right",
            pad=0,
        ),
        sg.InputText(
            "",
            background_color="#121212",
            text_color="#FF8C00",
            key="-CONV_FR-",
            font="Consolas 10",
            size=(5, 1),
            pad=0,
        ),
        sg.Text(
            "To:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(6, 1),
            justification="right",
            pad=0,
        ),
        sg.InputText(
            "",
            background_color="#121212",
            text_color="#FF8C00",
            key="-CONV_TO-",
            font="Consolas 10",
            size=(5, 1),
            pad=0,
        ),
    ]

    conv_del_param = [
        sg.Text(
            "DEL previous after Conv.",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(25, 1),
        ),
        sg.Rad(
            "Y",
            "opt_del_conv",
            size=(1, 1),
            background_color="#010101",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-DELCONV_Y-",
        ),
        sg.Rad(
            "N",
            "opt_del_conv",
            default=True,
            size=(1, 1),
            background_color="#010101",
            font="Consolas 10",
            text_color="#FF8C00",
            pad=0,
            key="-DELCONV_N-",
        ),
    ]

    resize_sz = [
        sg.Text(
            "Resize to:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(16, 1),
        ),
        sg.Text(
            "W:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(4, 1),
            justification="right",
            pad=0,
        ),
        sg.InputText(
            "",
            background_color="#121212",
            text_color="#FF8C00",
            key="-RESIZE_W-",
            font="Consolas 10",
            size=(5, 1),
            pad=0,
        ),
        sg.Text(
            "H:",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(4, 1),
            justification="right",
            pad=0,
        ),
        sg.InputText(
            "",
            background_color="#121212",
            text_color="#FF8C00",
            key="-RESIZE_H-",
            font="Consolas 10",
            size=(5, 1),
            pad=0,
        ),
    ]

    thread_nums = [
        sg.Text(
            "No. of Threads",
            text_color="#FF8C00",
            background_color="#000000",
            font="Consolas 10",
            size=(16, 1),
        ),
        sg.Input(
            default_text=str(CFG.THREAD_NUM),
            background_color="#121212",
            text_color="#FF8C00",
            key="-THREAD_NUM-",
            font="Consolas 10",
            size=(3, 1),
            justification="center",
        ),
    ]

    iptc_txt_pn = sg.Rad(
        "IPTC -> .txt",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    txt_iptc_pn = sg.Rad(
        ".txt -> IPTC",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    exif_txt_pn = sg.Rad(
        "EXIF -> .txt",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    txt_exif_pn = sg.Rad(
        ".txt -> EXIF",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    pnginfo_txt_pn = sg.Rad(
        "PNGInfo ->.txt",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    txt_pnginfo_pn = sg.Rad(
        ".txt->PNGInfo",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    strip_pn = sg.Rad(
        "Strip Metadata",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    convert_pn = sg.Rad(
        "Convert Types",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    resize_pn = sg.Rad(
        "Resize",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    heic_pn = sg.Rad(
        "HEIC -> *",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    ren_md5_pn = sg.Rad(
        "Rename to MD5",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    rm_pad_pn = sg.Rad(
        "Remove Padding",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    sq_pad_pn = sg.Rad(
        "Pad to Square",
        "img_util",
        size=(15, 1),
        background_color="#010101",
        font="Consolas 10",
        text_color="#FF8C00",
        enable_events=True,
    )

    f_params = [
        [
            src_dir_pn[0],
            src_dir_pn[1],
            src_dir_pn[2],
        ],
        [
            sg.Text(
                "",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 6",
                size=(1, 1),
                justification="left",
                pad=0,
            ),
        ],
        [
            save_dir_pn[0],
            save_dir_pn[1],
            save_dir_pn[2],
        ],
        [
            sg.Text(
                "",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 6",
                size=(1, 1),
                justification="left",
                pad=0,
            ),
        ],
        [
            recurse_pn[0],
            recurse_pn[1],
            recurse_pn[2],
            dir_bk_pn[0],
            dir_bk_pn[1],
            dir_bk_pn[2],
        ],
    ]

    params = [
        [
            alphabet_pn[0],
            alphabet_pn[1],
            alphabet_pn[2],
        ],
        [
            sg.Text(
                "",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 6",
                size=(1, 1),
                justification="left",
                pad=0,
            ),
        ],
        [
            exif_trans_pn[0],
            exif_trans_pn[1],
            exif_trans_pn[2],
        ],
        [
            sg.Text(
                "",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 6",
                size=(1, 1),
                justification="left",
                pad=0,
            ),
        ],
        [
            save_type_pn[0],
            save_type_pn[1],
            save_type_pn[2],
            save_type_pn[3],
        ],
        [
            sg.Text(
                "",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 6",
                size=(1, 1),
                justification="left",
                pad=0,
            ),
        ],
        [
            type_to_type[0],
            type_to_type[1],
            type_to_type[2],
            type_to_type[3],
            type_to_type[4],
        ],
        [
            sg.Text(
                "",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 6",
                size=(1, 1),
                justification="left",
                pad=0,
            ),
        ],
        [
            conv_del_param[0],
            conv_del_param[1],
            conv_del_param[2],
        ],
        [
            sg.Text(
                "",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 6",
                size=(1, 1),
                justification="left",
                pad=0,
            ),
        ],
        [
            resize_sz[0],
            resize_sz[1],
            resize_sz[2],
            resize_sz[3],
            resize_sz[4],
        ],
        [
            sg.Text(
                "",
                text_color="#FF8C00",
                background_color="#000000",
                font="Consolas 6",
                size=(1, 1),
                justification="left",
                pad=0,
            ),
        ],
        [
            thread_nums[0],
            thread_nums[1],
        ],
    ]

    opts = [
        [
            iptc_txt_pn,
            txt_iptc_pn,
            ren_md5_pn,
        ],
        [
            exif_txt_pn,
            txt_exif_pn,
            convert_pn,
        ],
        [
            pnginfo_txt_pn,
            txt_pnginfo_pn,
            strip_pn,
        ],
        [
            resize_pn,
            heic_pn,
            rm_pad_pn,
        ],
        [
            sq_pad_pn,
        ],
    ]

    console = [
        [
            sg.Output(
                echo_stdout_stderr=True,
                size=(108, 19),
                text_color="#00EE00",
                key="-OUTPUT-",
                background_color="#000000",
                font="Consolas 8",
            )
        ],
    ]

    control = [
        [
            sg.Button(
                "Start",
                button_color="#00FF00 on #00008B",
                font="Consolas 12 bold",
            ),
            sg.Text(
                "",
                size=(53, 1),
                background_color="#000000",
            ),
            sg.Button(
                "Help",
                button_color="#FF8C00 on #121212",
                font="Consolas 12 bold",
            ),
            sg.Text(
                "",
                size=(53, 1),
                background_color="#000000",
            ),
            sg.Button(
                "Exit",
                button_color="#FFFFFF on #8B0000",
                font="Consolas 12 bold",
            ),
        ]
    ]

    img_util_layout = [
        [
            sg.Frame(
                "Actions",
                opts,
                background_color="#000000",
                font="Consolas 10",
                title_color="#3264FF",
                vertical_alignment="top",
            ),
            sg.Frame(
                "File Parameters",
                f_params,
                background_color="#000000",
                font="Consolas 10",
                title_color="#3264FF",
                vertical_alignment="top",
                size=(515, 165),
            ),
        ],
        [
            sg.Frame(
                "Parameters",
                params,
                background_color="#000000",
                font="Consolas 10",
                title_color="#3264FF",
            ),
            sg.Frame(
                "Console",
                console,
                background_color="#000000",
                font="Consolas 10",
                title_color="#3264FF",
                vertical_alignment="top",
            ),
        ],
        [
            sg.Frame(
                "",
                control,
                background_color="#000000",
                font="Consolas 10",
                title_color="#3264FF",
                vertical_alignment="bottom",
            )
        ],
    ]

    sg.set_options(text_justification="left")

    window = sg.Window(
        "Image Tagger",
        img_util_layout,
        font="Consolas 10",
        background_color="#000000",
        use_custom_titlebar=False,
        titlebar_background_color="#000000",
        titlebar_text_color="#ffffff",
        keep_on_top=True,
        finalize=True,
    )

    window["-FILE_PATH-"].Widget.config(insertbackground="red")
    window["-SAVE_PATH-"].Widget.config(insertbackground="red")
    window["-TYPE_W-"].Widget.config(insertbackground="red")
    window["-CONV_FR-"].Widget.config(insertbackground="red")
    window["-CONV_TO-"].Widget.config(insertbackground="red")
    window["-RESIZE_W-"].Widget.config(insertbackground="red")
    window["-RESIZE_H-"].Widget.config(insertbackground="red")
    window["-THREAD_NUM-"].Widget.config(insertbackground="red")

    return window


if __name__ == "__main__":

    def list_gen():
        CFG.FILE_LIST: list[str] = []
        if CFG.RECURSE:
            for r, _, f in os.walk(CFG.FILE_PATH[:-1:]):
                for file in f:
                    if ((file[-(file[::-1].find(".")) :]).lower()) in CFG.IMG_TYPES:
                        CFG.FILE_LIST.append(os.path.join(r, file))
        elif not CFG.RECURSE:
            CFG.FILE_LIST = [
                CFG.FILE_PATH + f
                for f in os.listdir(CFG.FILE_PATH[:-1:])
                if os.path.isfile(CFG.FILE_PATH + f)
                and f[-(f[::-1].find(".")) :].lower() in CFG.IMG_TYPES
            ]

    win.dow = window_main()
    CFG.SP_CHG = False

    while True:
        try:
            CFG.VALIDATION = True
            event, values = win.dow.read()
            win.dow.refresh()
            if event in (sg.WIN_CLOSED, "EXIT"):
                break
            if event == "-TYPE_W-":
                win.dow["-TYPE_D-"].Update("")
                win.dow.refresh()
            if not CFG.SP_CHG and event == "-SAVE_PATH-":
                if (
                    len(
                        set(str(values["-SAVE_PATH-"]).split()).intersection(
                            "Leave this or keep empty to use File Path".split()
                        )
                    )
                    > 5
                ):
                    win.dow["-SAVE_PATH-"].Update("")
                else:
                    continue
            if event == "Help":
                sg.popup(
                    "PySimpleGUI Demo All Elements",
                    "Right click anywhere to see right click menu",
                    "Visit each of the tabs to see available elements",
                    "Output of event and values can be see in Output tab",
                    "The event and values dictionary is printed after every event",
                    keep_on_top=True,
                )
            elif event == "Start":
                win.dow["-OUTPUT-"].Update("")
                print(values)
                CFG.RUN_OPT = CFG.OPT_VALS[
                    list(
                        map(
                            lambda x: values[x] if values[x] == True else None,
                            range(0, 13),
                        )
                    ).index(True)
                ]
                print(CFG.RUN_OPT)
                CFG.FILE_PATH = values["-FILE_PATH-"]
                if CFG.FILE_PATH[:-1:] != chr(92) or CFG.FILE_PATH[:-1:] != chr(47):
                    CFG.FILE_PATH = str(CFG.FILE_PATH + chr(47)).replace(
                        chr(92), chr(47)
                    )
                CFG.SAVE_PATH = str(values["-SAVE_PATH-"])
                if CFG.SAVE_PATH == "Leave this or keep empty to use File Path":
                    CFG.SAVE_PATH = CFG.FILE_PATH
                if os.path.isdir(CFG.FILE_PATH):
                    if CFG.FILE_PATH[:-1:] == chr(92) or CFG.FILE_PATH[:-1:] == chr(47):
                        CFG.FILE_PATH = CFG.FILE_PATH[:-1:]
                    print("File Path Folder Check: PASS")
                    CFG.FILE_PATH = CFG.FILE_PATH + chr(47)
                elif not os.path.isdir(CFG.FILE_PATH):
                    print("File Path Folder Check: FAIL")
                    print("Please select a valid folder")
                    CFG.FILE_PATH = ""
                    CFG.VALIDATION = False
                    continue
                if os.path.isdir(CFG.SAVE_PATH):
                    if CFG.SAVE_PATH[:-1:] == chr(92) or CFG.SAVE_PATH[:-1:] == chr(47):
                        CFG.SAVE_PATH = CFG.SAVE_PATH[:-1:]
                    print("Save Path Folder Check: PASS")
                    CFG.SAVE_PATH = CFG.SAVE_PATH + chr(47)
                elif not os.path.isdir(CFG.SAVE_PATH):
                    print("Save Path Folder Check: FAIL")
                    print("Please select a valid folder")
                    CFG.SAVE_PATH = ""
                    CFG.VALIDATION = False
                    continue

                if CFG.RUN_OPT == "OPT_IPTC_TXT":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = "IPTC->TXT"
                    CFG.IMG_TYPES = [
                        "jpg",
                        "jpeg",
                        "tiff",
                        "png",
                        "miff",
                        "ps",
                        "pdf",
                        "psd",
                        "xcf",
                        "dng",
                    ]
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    continue

                if CFG.RUN_OPT == "OPT_TXT_IPTC":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = "IPTC->TXT"
                    CFG.IMG_TYPES = [
                        "jpg",
                        "jpeg",
                        "tiff",
                        "png",
                        "miff",
                        "ps",
                        "pdf",
                        "psd",
                        "xcf",
                        "dng",
                    ]
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    continue

                if CFG.RUN_OPT == "OPT_EXIF_TXT":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = "EXIF->TXT"
                    CFG.IMG_TYPES = [
                        "jpg",
                        "jpeg",
                        "tiff",
                        "png",
                        "jp2",
                        "pgf",
                        "miff",
                        "hdp",
                        "psp",
                        "xcf",
                    ]
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    continue

                if CFG.RUN_OPT == "OPT_TXT_EXIF":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = "TXT->EXIF"
                    CFG.IMG_TYPES = [
                        "jpg",
                        "jpeg",
                        "tiff",
                        "png",
                        "jp2",
                        "pgf",
                        "miff",
                        "hdp",
                        "psp",
                        "xcf",
                    ]
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    continue

                if CFG.RUN_OPT == "OPT_REN_MD5":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = "MD5 RENAME"
                    CFG.IMG_TYPES = CFG.STD_IMG_TYPES
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    continue

                if CFG.RUN_OPT == "OPT_PNGINFO_TXT":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = "PNGInfo->TXT"
                    CFG.IMG_TYPES = ["png"]
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    continue

                if CFG.RUN_OPT == "OPT_TXT_PNGINFO":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = "TXT->PNGInfo"
                    CFG.IMG_TYPES = ["png"]
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    continue

                if CFG.RUN_OPT == "OPT_CONVERT":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.CONV_TYPE = (
                        str(values["-TYPE_D-"]).lower()
                        if len(values["-TYPE_D-"]) > 0
                        else str(values["-TYPE_W-"]).lower()
                    )
                    CFG.TQ_DESC = f"Convert to {CFG.CONV_TYPE}"
                    CFG.IMG_TYPES = CFG.STD_IMG_TYPES
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [
                            executor.submit(CFG.FUNC, x, CFG.CONV_TYPE)
                            for x in CFG.FILE_LIST
                        ]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    CFG.CONV_TYPE = ""
                    continue

                if CFG.RUN_OPT == "OPT_STRIP":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = f"Stripping Metadata"
                    CFG.IMG_TYPES = CFG.STD_IMG_TYPES
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    continue

                if CFG.RUN_OPT == "OPT_RESIZE":
                    CFG.BACKUP = values["-BACKUP_Y-"] if values["-BACKUP_Y-"] else False
                    if CFG.BACKUP:
                        bk_zip()
                    CFG.A_TO_Z = values["-A_TO_Z_Y-"] if values["-A_TO_Z_Y-"] else False
                    CFG.RECURSE = (
                        values["-RECURSE_N-"]
                        if values["-RECURSE_N-"] == True
                        else values["-RECURSE_Y-"]
                    )
                    CFG.RESIZE_W = values["-RESIZE_W-"]
                    try:
                        CFG.RESIZE_W = values["-RESIZE_W-"]
                        print("Width: ", int(CFG.RESIZE_W))
                    except ValueError as e:
                        print(e)
                        print("Width Validation: FAIL")
                        continue
                    try:
                        CFG.RESIZE_H = values["-RESIZE_H-"]
                        print("Height: ", int(CFG.RESIZE_W))
                    except ValueError as e:
                        print(e)
                        print("Width Validation: FAIL")
                        continue
                    CFG.THREAD_NUM = int(values["-THREAD_NUM-"])
                    CFG.FUNC = locals()[CFG.RUN_OPT.lower()]
                    CFG.TQ_DESC = f"Resizing to: ({CFG.RESIZE_W},{CFG.RESIZE_H})"
                    CFG.IMG_TYPES = CFG.STD_IMG_TYPES
                    list_gen()
                    print("Number of Files to Process: ", len(CFG.FILE_LIST))
                    status_bar = statbar(len(CFG.FILE_LIST), CFG.TQ_DESC)
                    with ThreadPoolExecutor(CFG.THREAD_NUM) as executor:
                        futures = [executor.submit(CFG.FUNC, x) for x in CFG.FILE_LIST]
                        for _ in as_completed(futures):
                            status_bar.update(n=1)
                            win.dow.refresh()
                        status_bar.close()
                    CFG.CONV_TYPE = ""
                    continue

                # if CFG.RUN_OPT == "OPT_HEIC":

                # if CFG.RUN_OPT == "OPT_RM_PAD":

                # if CFG.RUN_OPT == "OPT_SQ_PAD":

            elif event == "Exit":
                win.dow.close()
                sys.exit()
            elif event == "Browse":
                print("wat")
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(e)
            continue

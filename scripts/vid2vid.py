# Author: Filarius
# Credits: orcist1
# https://github.com/Filarius

import json
import os,sys
import shutil
import string
import pathlib
from subprocess import Popen, PIPE
import numpy as np
from PIL import Image

import gradio as gr
from modules import scripts
from modules.images import save_image
from modules.sd_samplers import sample_to_image
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.processing import Processed, process_images

class Script(scripts.Script):
    # script title to show in ui
    def title(self):
        return 'Video2video'

    def show(self, is_img2img):
        #return scripts.AlwaysVisible
        return is_img2img


    # ui components
    def ui(self, is_visible):
            with gr.Row():
                file = gr.File(label="Upload Video", file_types = ['.*;'], live=True, file_count = "single")
                tmp_path = gr.Textbox(label='Or path to file', lines=1, value='')


            with gr.Row():
                fps = gr.Slider(
                    label="FPS change",
                    minimum=1,
                    maximum=60,
                    step=1,
                    value=24,
                )
            return [tmp_path, fps, file]



    def run(self, p, file_path, fps, file_obj, *args):
                # return_images, all_prompts, infotexts, inter_images = [], [], [], []
            # state.job_count = inp_gif.n_frames * p.n_iter
            if not os.path.isfile(file_path):
                file_path = file_obj.name
            p.do_not_save_grid = True
            p.do_not_save_samples = True

            initial_seed = p.seed
            p.do_not_save_grid = True
            p.do_not_save_samples = True
            p.batch_count = 1

            start_time = "00:00:00"
            end_time = "00:00:00"
            start_time = start_time.strip()
            end_time = end_time.strip()

            if start_time == "" or start_time == "hh:mm:ss":
                start_time = "00:00:00"
            if end_time == "00:00:00" or end_time == "hh:mm:ss":
                end_time = ""

            time_interval = (
                f"-ss {start_time}" + f" -to {end_time}" if len(end_time) else ""
            )

            import modules

            path = modules.paths.script_path
            save_dir = "outputs/img2img-video/"
            ffmpeg.install(path, save_dir)

            input_file = os.path.normpath(file_path.strip())
            decoder = ffmpeg(
                " ".join(
                    [
                        "ffmpeg/ffmpeg -y -loglevel panic",
                        f'{time_interval} -i "{input_file}"',
                        f"-s:v {p.width}x{p.height} -r {fps}",
                        "-f image2pipe -pix_fmt rgb24",
                        "-vcodec rawvideo -",
                    ]
                ),
                use_stdout=True,
            )
            decoder.start()

            output_file = input_file.split("\\")[-1]
            encoder = ffmpeg(
                " ".join(
                    [
                        "ffmpeg/ffmpeg -y -loglevel panic",
                        "-f rawvideo -pix_fmt rgb24",
                        f"-s:v {p.width}x{p.height} -r {fps}",
                        "-i - -c:v libx264 -preset medium",
                        f'-crf 22 "{save_dir}/{output_file}"',
                    ]
                ),
                use_stdin=True,
            )
            encoder.start()

            pull_count = p.width * p.height * 3
            batch = []

            while True:
                raw_image = decoder.readout(pull_count)
                if len(raw_image) == 0:
                    raw_image = None
                if (raw_image is None) and len(batch)==0:
                    break

                if (raw_image is None):
                   p.batch_size = len(batch)
                else:
                    image_PIL = Image.fromarray(
                        np.uint8(raw_image).reshape((p.height, p.width, 3)), mode="RGB"
                    )
                    batch.append(image_PIL)

                if len(batch) == p.batch_size:
                    p.seed = initial_seed
                    p.init_images = batch
                    batch = []
                    try:
                        proc = process_images(p)
                    except:
                        break

                    for output in proc.images:
                        if output.mode != "RGB":
                            output = output.convert("RGB")
                        encoder.write(np.asarray(output))

            encoder.write_eof()

            return Processed(p, [], p.seed, proc.info)

class ffmpeg:
    def __init__(
        self,
        cmdln,
        use_stdin=False,
        use_stdout=False,
        use_stderr=False,
        print_to_console=True,
    ):
        self._process = None
        self._cmdln = cmdln
        self._stdin = None

        if use_stdin:
            self._stdin = PIPE

        self._stdout = None
        self._stderr = None

        if print_to_console:
            self._stderr = sys.stdout
            self._stdout = sys.stdout

        if use_stdout:
            self._stdout = PIPE

        if use_stderr:
            self._stderr = PIPE

        self._process = None

    def start(self):
        self._process = Popen(
            self._cmdln, stdin=self._stdin, stdout=self._stdout, stderr=self._stderr
        )

    def readout(self, cnt=None):
        if cnt is None:
            buf = self._process.stdout.read()
        else:
            buf = self._process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)

        return arr

    def readerr(self, cnt):
        buf = self._process.stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self._process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self._process.stdin.close()

    def is_running(self):
        return self._process.poll() is None

    @staticmethod
    def install(path, save_dir):
        from basicsr.utils.download_util import load_file_from_url
        from zipfile import ZipFile

        ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/5.1.1/ffmpeg-5.1.1-full_build.zip"
        ffmpeg_dir = os.path.join(path, "ffmpeg")

        ckpt_path = load_file_from_url(url=ffmpeg_url, model_dir=ffmpeg_dir)

        if not os.path.exists(os.path.abspath(os.path.join(ffmpeg_dir, "ffmpeg.exe"))):
            with ZipFile(ckpt_path, "r") as zipObj:
                listOfFileNames = zipObj.namelist()
                for fileName in listOfFileNames:
                    if "/bin/" in fileName:
                        zipObj.extract(fileName, ffmpeg_dir)
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffmpeg.exe"),
                os.path.join(ffmpeg_dir, "ffmpeg.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffplay.exe"),
                os.path.join(ffmpeg_dir, "ffplay.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffprobe.exe"),
                os.path.join(ffmpeg_dir, "ffprobe.exe"),
            )

            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin"))
            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1]))
        os.makedirs(save_dir, exist_ok=True)
        return

    @staticmethod
    def seconds(input="00:00:00"):
        [hours, minutes, seconds] = [int(pair) for pair in input.split(":")]
        return hours * 3600 + minutes * 60 + seconds

from pathlib import Path
import requests
import os

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

BASE_DIR = Path(__file__).resolve().parent.parent
RUNPOD_VOLUME_DIR = '/runpod-volume'
mdxnet_models_dir = RUNPOD_VOLUME_DIR / 'mdxnet_models'
rvc_models_dir = RUNPOD_VOLUME_DIR / 'rvc_models'


def dl_model(link, model_name, dir_name):
    model_path = dir_name / model_name
    
    if model_path.exists():
        #print(f"Model '{model_name}' already exists in directory '{dir_name}'. Skipping download.")
        return

    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == '__main__':
    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        print(f'Downloading {model}...')
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    rvc_model_names = ['hubert_base.pt', 'rmvpe.pt']
    for model in rvc_model_names:
        print(f'Downloading {model}...')
        dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)

    print('All models downloaded!')

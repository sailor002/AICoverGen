from pathlib import Path
import requests
import os

MDX_DOWNLOAD_LINK = 'https://ai-music-application.s3.us-east-2.amazonaws.com/base_models/'
RVC_DOWNLOAD_LINK = 'https://ai-music-application.s3.us-east-2.amazonaws.com/base_models/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = f'{BASE_DIR}/mdxnet_models'
rvc_models_dir = f'{BASE_DIR}/rvc_models'


def dl_model(link, model_name, dir_name):
    model_path = Path(os.path.join(dir_name, model_name))
    #model_path = dir_name / model_name
    
    if model_path.exists():
        print(f"Model '{model_name}' already exists in directory '{dir_name}'. Skipping download.")
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

# !pip install img2dataset
# !pip install datasets

from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from img2dataset import download
import os
import matplotlib.pyplot as plt
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def clean_df(df):

    df["pixels"] = df["HEIGHT"]*df["WIDTH"]
    df["ratio"] = df["HEIGHT"]/df["WIDTH"]
    x = df["similarity"]
    x[x<1].hist(bins=100)
    plt.show()
    df["ratio"].quantile(np.arange(0, 1, 0.01)).plot()
    plt.show()
    df["pixels"].quantile(np.arange(0, 1, 0.01)).plot()
    plt.show()

    df = df[df["similarity"] >= 0.3]
    print(df.shape)

    #### only pick the first URL - there are a lot of duplicate images:
    df = df.groupby("URL").first().reset_index()

    df = df.drop_duplicates(subset = ["TEXT", "WIDTH", "HEIGHT"])
    print(df.shape)

    #remove huge images for faster download:
    df = df[df["pixels"] <= 1024*1024]
    print(df.shape)

    # remove images that aren't close to being square - otherwise faces get cropped.
    df = df[df["ratio"] > 0.3]
    print(df.shape)
    df = df[df["ratio"] < 2]
    print(df.shape)

    df = df[df["AESTHETIC_SCORE"] > 5.5]
    print(df.shape)

    vectorizer = CountVectorizer(min_df=25, stop_words="english")
    text = df["TEXT"]
    vectorizer.fit(text)

    word_counts = vectorizer.transform(text)
    x = word_counts.sum(1)
    df["word_counts"] = pd.DataFrame(x)[0].values

    df["word_counts"].value_counts().sort_index()[:30].plot(kind="bar")

    df = df[df["word_counts"] >= 1.0]
    print(df.shape)
    df = df[df["word_counts"] <= 35.0]
    print(df.shape)

    return df

def download_data(url_path="df"):
        download(
        processes_count=8,
        thread_count=16,
        url_list=url_path,
        image_size=64,
        output_folder=output_dir,
        output_format="files",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        number_sample_per_shard=10000,
        distributor="multiprocessing",
        resize_mode="center_crop",
        )


def get_imgs_and_captions():
    #TODO: This is sketchy...and error prone:
    imgs_files = glob.glob("/content/output_dir/*/*.jpg")
    imgs_files.sort()
    text_files = glob.glob("/content/output_dir/*/*.txt")
    text_files.sort()

    caption_list = []
    for txt_path in tqdm(text_files):
        with open(txt_path) as f:
            lines = f.readlines()
            caption_list.append(lines[0])

    img_list = []
    for img_path in tqdm(imgs_files):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

    img_array = np.array(img_list)
    del img_list
    return img_array, caption_list

def get_dataset_from_huggingface(hugging_face_dataset_name):
    dataset = load_dataset(hugging_face_dataset_name, split="train")
    df = dataset.to_pandas()

    return df

def build_dataset_and_save(hugging_face_dataset_name):
    sample_size = 120000
    seed = 1
    df = get_dataset_from_huggingface(hugging_face_dataset_name)

    df = clean_df(df)
    np.random.seed(seed)
    df.sample(sample_size).to_parquet("df") #note that this keeps the original indexes from the data.
    output_dir = os.path.abspath("output_dir")
    download_data(url_path="df")
    train_data, caption_list = get_imgs_and_captions()
    pd.Series(caption_list).to_csv("/content/diffusion_model_aesthetic_keras/captions.csv", index=None)
    np.save("/content/diffusion_model_aesthetic_keras/imgs.npy", train_data)
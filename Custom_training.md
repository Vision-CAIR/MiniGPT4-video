# Customizing MiniGPT4-video for your own Video-text dataset

## Add your own video dataloader 
Construct your own dataloader here `minigpt4/datasets/datasets/video_datasets.py` based on the existing dataloaders.<br>
Copy Video_loader_template class and edit it according to you data nature.

## Create config file for your dataloader
Here `minigpt4/configs/datasets/dataset_name/default.yaml` creates your yaml file that includes paths to your dataset.<br>
Copy the template file `minigpt4/configs/datasets/template/default.yaml` and edit the paths to your dataset.


## Register your dataloader
In the `minigpt4/datasets/builders/image_text_pair_builder.py` file
Import your data loader class from the `minigpt4/datasets/datasets/video_datasets.py` file <br>
Copy and edit the VideoTemplateBuilder class.<br>
put the train_dataset_cls = YourVideoLoaderClass that you imported from `minigpt4/datasets/datasets/video_datasets.py` file.

## Edit training config file 
Add your dataset to the datasets in the yml file as shown below:
```yaml
datasets:
  dataset_name: # change this to your dataset name
    batch_size: 4  # change this to your desired batch size
    vis_processor:
      train:
        name: "blip2_image_train"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"
    sample_ratio: 200 # if you including joint training with other datasets, you can set the sample ratio here
```


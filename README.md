# DogBreedAI (Still in progress)
First frontend project of a minor web application to predict different dog breeds based on photos from the gallery/right from the camera. 

<hr>

Initially using pretrained [ResNeXt](https://arxiv.org/abs/1611.05431) as classification network, but also created a customized ResNeXt aswell as an ensmble network of different ResNet and ResNeXt architectures to be trained. The dataset was collected from [kaggle](https://www.kaggle.com/c/dog-breed-identification/data) consisting of 120 different dog breeds with a total size of 10222 images. As the official amount of total breeds, according to FCI, is [360](https://www.psychologytoday.com/us/blog/canine-corner/201305/how-many-breeds-dogs-are-there-in-the-world), the model covers a third of the entire species.
<br>
<br>
The image below shows the current status of the web application in its early stage

![Current status](https://github.com/olof98johansson/DogBreedAI/blob/main/status/first.PNG?raw=true)

<br>
which can input an image from gallery, upload it, and make a valid prediction. The following list illustrates what has been done and what will be done.

Checkpoints | Dataset | Create web app | Implement model | Preprocessing functions | Initial ML pipeline | Phone scalability | Improve performance | "Fun fact" feature | More aesthetics
:------------ | :-------------| :-------------| :-------------| :-------------| :-------------| :-------------| :-------------| :-------------
Status | :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: | :white_check_mark: | :heavy_check_mark:| :heavy_check_mark:| :heavy_check_mark:



| Task                                              | Status |
| ------------------------------------------------- | ----   |
| :black_circle: <input type="checkbox" disabled checked /> Dataset  |      |
| :black_circle: <input type="checkbox" disabled  checked/> Create web app |      |
| <input type="checkbox" disabled  checked/> Implement model |      |
| <input type="checkbox" disabled  checked/> Preprocessing functions |      |
| <input type="checkbox" disabled  checked/> Initial ML pipeline |      |
| <input type="checkbox" disabled  checked/> Phone scalability |      |
| <input type="checkbox" disabled  checked/> Improve performance |      |
| <input type="checkbox" disabled  checked/> "Fun fact" feature |      |
| <input type="checkbox" disabled  checked/> "Fun fact" feature |      |
| <input type="checkbox" disabled  checked/> Aesthetics & design |      |

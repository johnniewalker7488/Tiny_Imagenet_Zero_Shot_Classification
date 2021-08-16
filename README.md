# Zero-Shot Classification on Tiny Imagenet dataset
Zero-shot classification methods tested on Tiny Imagenet dataset

## What is Zero-Shot Learning?
As soon as in classic image recongition tasks we need to label images to be able to train a model at some moment we encounter a problem of scalability. 
We are unable to permanently increase the number of classes as it becomes too expensive to label more and more of them.
However, it is not necessary for humans to _see_ an object in advance to be able to recognize it.
Let say, a child can recognize a zebra on a picture even if he has never seen a zebra before. It is enough for him to have seen a horse and also _learn somewhere_ that zebra looks like a horse and has black and white stripes.
In case of computer vision this is what's called zero-shot learning.

## How Zero-Shot Learning Works
In a zero-shot learning pipeline we get feature embeddings from images and also some semantic embeddings as an auxiliary dataset. As for image feature embeddings it is an easy task for a convolutional neural network while semantic embeddings can be obtained from a pretrained word2vec model. After that we are to minimize the distance between the two distributions we obtained and then use some KNN-based algorithm to find the closest word embedding to the embedding of the input image.

<img width="394" alt="zero_shot_pipeline (1)" src="https://user-images.githubusercontent.com/44619521/127662474-27f73ccf-0c08-459c-a4c6-6a9e2c5f8700.PNG">

## Implementation and Results
This is a simple example of zero-shot learning approach using a custom ResNet as the CNN feature extractor and the pretrained Google News word2vec vectors as semantic embeddings. Using cosine embedding loss in addition to the cross-entropy loss gives __25.85% top-5 accuracy__ on 50 __unseen__ classes with random guessing of 10%. 

# Humans vs everything
The objective of this project was to build an image classifier to classify whether the image contains a human or not. Seems straight forward given the fact that there are a lot of code snippets available to build an image classifer but there's a catch most of the dataset readily available are for cats, dogs, digits and other stuff seemingly random stuff.
On first glance human detection dataset seems like an obvious choice but its not, since theres a fundamental difference in how image detection works as opposed to image classification using neural nets are concerned.
Neural nets can be considered as a cake of filters, just like a cake has layers, each layer in neural network contains (or builds) a number of filters targeting a very specific feature. In image detection these features are marked and the purpose of the Neural net is to learn from those marked features. However in case of image classification, the end objective is to determine whether an image falls under a certain class or not and for this very purpose the network not only has to learn what (features) makes a human being but also what features does not.
Many challenges for this project were
  - Finding the appropriate dataset.
  - Determining the architecture.
  - Hardware constrains.

# Finding the appropriate dataset
Data is everything for a model, since an algorithm is transformed into a model based on the data that is fed to it so choosing the right one was crucial.
The dataset which was suitable for this purpose was the GRAZ-01 dataset.However the size of the dataset was abysmally small, containing only 420 images out of which the latter 200 were supposed to be testing data ie they were not clear in depicting a human being.
# Determining the architecture
Since the training data on hand was clean but was really small, training a neural net from scratch was out of question. So the next choice was to find a model that satisfies my need.
##### Enter transfer learning
   Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.
https://en.wikipedia.org/wiki/Transfer_learning

The description above encompasses the core essence of transfer learning.So the next step was to find a pretrained model which was capable of doing either 2 things:
  - classify whether a human is in the image.
  - classify whether an entity other than human is in the image.

There are really impressive models released by researchers for free that are able to classify more that 1000 classes simulatneously. The model ive chosen for this purpose was VGG16 Net. This is really simple in architecture but is 14 layers deep and has around 160 million parameters to train. We wont be dealing with the training of hidden layers so thats a relief.
Now why did I choose this model over other networks is due to accuracy and efficiency. VGG16 is made of only convolutional and max pooling blocks which are relatively less straining operations when hardware is one of the biggest constraints.
# Hardware constraints
Having a GTX 1050ti with 4 GB of VRAM cramped into a laptop chassis is not cool (pun intended). 4 GB of VRAM is not enough when dealing with coloured images of atleast 224 x 224 in size. Theramal issues do arise when dealing with laptop GPU since the thermal solution can be stressed easily in such tight spaces.
The next issue with hardware is the limited amount of RAM (8 GB) running Windows OS is not ideal for such applications the reason being the OS it self consumes 2 - 2.5 GB RAM while idling. That leaves with 6 GB of system memory which can be easily consumed due to tensor operations coupled with python's unsatifactory garbage collection routines. Most of the tensor operations returns a new tensor which if not stored in a variable is stored in temporary memory which is system memory and garbage collector does not interfere with system memory operations thus, one ends up consuming the enitre system memory when dealing with tensor operations in python environmet. Same is also true while using pandas for data wrangling.

Thus in conclusion to architecture selection:
  - Dataset dictated the way to transfer learning.
  - Hardware constraints dictated the architecture for transfer learning.

# Working
VGG16 Net is a simple yet powerful Deep Convolutional Neural Network. 
VGG16 is trained on ImageNet dataset which means that the weights are trained to activate if a human is present in an image or not.
The top layer of VGG16 is an input layer which expects an input having a dimension of 224 x 224. Since the input layer has set the input shape the weights of the convolutional layers below it are already trained to accept tesnors of such dimensions.Therefore down scaling the 256 x 256 image to 224 x 224 size was a better option than rebuilding this massive network.
There last layer which is the predictions layer of VGG16 was removed since the aim is to classify whether a human is present or not ie a binary classification challenge.Thus the last layer was removed and replaced with a dense layer with a dropout of 30% followed by another dense layer with softmax activation.
Another thing to note is that the VGG16 architecture is sequential in nature but due to keras the network is downloaded as a functional network thus the network was transformed into a sequential network and then modified accordingly.
# Results
Trained on 647 images (350 for humans and 297 for non humans) required an epoch time of 18s - 47s per epoch.
Total of 15 epochs were used with a batch size of 25 with ADAM optimizer which yielded an 80% accuracy on test set of 20 images (10 for humans and 10 non humans shuffled randomly).
# Infernece
The model performed remarkably well for being modified to suit the purpose of the objective statement while being trained on such a small dataset.
The epoch timings were relatively short for the reason being:
  - The simple yet deep architecture meant basic multiplicative tensor operations were taking place which is memory intensive but computationally efficient.
 
# Conclusion
In conclusion the environmental constraints played a major role in deciding the architecture, which had to be navigated thoughtfully.
Extending the architecture would have yielded better accuracy but given the timeframe was not possible however the model produced meets the required objective statement.



<a name="readme-top"></a>




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="">
    <img src="images/HiRes-7.jpeg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Data Science in Health project (Oral Cancer image classification)</h3>

  <p align="center">
    This project is about classify oral cancer by using image classification with various techniques and implementing our own CNN model.
    <br />
    
  </p>
</div>





<!-- ABOUT THE PROJECT -->
## About The Project

### Oral cancer is causes from tobacco and alcohol consumption. Other risk factors can include: human papillomavirus (HPV) epstein-barr virus (EBV). It can occur in all part of your mouth. There is a cell called squamous cell carcinoma(SCC) which is a type of skin cancer that can grow large or spread to other parts of the body. Below will be an example images of normal cell and SCC cell:
<table>
  <tr>
    <td align="center">
      <img src="Oral Cancer/Normal/oral_normal_0008.jpg" width="300" height="300"><br>
      Normal cell
    </td>
    <td align="center">
      <img src="Oral Cancer/Squamous Cell Carcinoma/oral_scc_0024.jpg" width="300" height="300"><br>
      Squamous Cell Carcinoma
    </td>
  </tr>
</table>


From the images, we can see that the normal sample shows a small size of epithelium, whereas the OSCC sample shows large size of squamous epithelium.



<!-- Walk through the code  -->
## Walk through the code
We firstly splitted into 70% of train set and 30% of train set for both classes.
```! pip install python_splitter
import python_splitter
python_splitter.split_from_folder("/tmp/Oral Cancer", train=0.7, test=0.3)
   ```

Then we create a function for transforming all of images by resizing to 224x224 pixels, and converting image to pytorch tensor.
```img_dim = 224
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_dim, img_dim)), # Resize the image to 224x224
    torchvision.transforms.ToTensor() # Convert the image to a pytorch tensor
])
```
After that we check the number of samples in the train and test dataset which there are 7000 images in the train set and 3002 images in the test set.
<img src="images/2nd.png">

---

Now we splitted data into train and and validation set randomly from 70% train set that we already splitted. We will have 4900 images in the new train set, 2100 images in the validation set, and 3002 images in the test set.

<img src="images/2nd2.png">

---

<img src="images/2nd3.png" width="800" height="400">


<img src="images/third.png">
This will be an example of the images that applied transforming into them.

<img src="images/fifth.png">
This is our test CNN model that was applied for training.

<img src="images/sixth.png">
Now we calculate validation loss by using cross-entropy function, SGD as optimizer, and lr scheduler for reducing learning rate. In our project, we got the best validation loss as 0.589 and train loss as 0.3039.

<img src="images/seventh.png">
<img src="images/seventh2.png">

After we got the result of the loss for train and validation set, we now can plot the loss and the learning rate with log scale.


<img src="images/output.png">
Then finally we used test images to predict the correct class which we got accuracy of the network on the test images at 68%.

This confusion matrix display the amount of images that the model can predict their classes correct and incorrect, we can see that both class have half way more than incorrect prediction which could consider as a good result since the model was crafted from scratch.

<img src="images/output2.png">
<img src="images/output3.png">
<img src="images/output4.png">

Lastly, we displayed the images for each class that was predicted correctly and incorrectly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/dgNathiRocha/datascienceHealth.git
   ```
2. Install all libraries
 - Matplotlib
 - Torch
 - Time
 - Numpy
 - SkLearn
 - Torvhvision

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

This project aim to classify accurate as much as possible but in the end, the crucial part is about understanding the process of the workflow.

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




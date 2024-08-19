<h1> Image Processing Project </h1>
<ol>
  <li> The aim of the project is to analyse the rotational speed of different rings from a video of rotating rings densely packed in a box. <br> </li>
  <li> This repo contains python scripts to apply image transformations,cross- correlation method and calculation  rotational velocity for all 180 rings at all time instances and data analysis </li>
  <li> Velocity vectors are superimposed on original frames to visually compare ring velocities with each other. </li>
  <li> Tech Stack: Python, Numpy, Pandas, Matplotlib, Scipy, OpenCV </li>
  
</ol>


<h2> Video of rotating rings </h2> <br>

![RingsVideoGIF](https://github.com/user-attachments/assets/5986030e-81c2-45a8-8459-336490fb39af)

<br>

<h2> Calculation of ring's bounding box using Trigonometry </h2> <br>

![IMG_20210604_231539](https://github.com/user-attachments/assets/6321d36c-3d14-4e29-9b2d-515441818f04)

<br>

![IMG_20210604_231550](https://github.com/user-attachments/assets/5e52fbfd-aeb5-4df2-afc9-0bc0ce7fa5fa)

<br>

<h2> Detection of all bounding box </h2> <br>

![IMG_20210606_173731](https://github.com/user-attachments/assets/0a7fcdc5-6880-48ff-92e0-4af0916d9962)

<br>

<h2> Isolating each ring and aligning it to x-axis</h2> <br>

![IMG_20210606_221910](https://github.com/user-attachments/assets/7df1a8f6-4aa5-4687-a7b1-110c87e090b6)
![IMG_20210613_100005](https://github.com/user-attachments/assets/de2b73ca-7131-4635-8d14-899ddc724da1)
![IMG_20210612_191153](https://github.com/user-attachments/assets/798b8034-1dc2-4d46-a3ca-3f524b26fbb1)
<br>

<h2> Saving of roi (Region of interest) locally for each ring and at all time instances </h2> <br>

![IMG_20210613_100638](https://github.com/user-attachments/assets/24a64ee1-3bb7-400e-8623-1d181f647636)

<br>

<h2> Cross Correlation method to find the displacement of pixels and its usage for retrieval of rings displacement in between frames with fixed difference  </h2> <br>

![IMG_20210627_230832(1)](https://github.com/user-attachments/assets/c151d87e-e789-41ba-acb2-89b09750b906)
![IMG_20210627_231840(1)](https://github.com/user-attachments/assets/3ac0c9bf-dba9-4421-9f02-735babab4a79)

<br>

<h2> Arrowhead scaling on the basis of ring's rotational speed   </h2> <br>

![IMG_20210715_111426(1)](https://github.com/user-attachments/assets/ab8a7ffe-f7ee-4b5a-8636-bb60a9408642)
![IMG_20210715_111430(1)](https://github.com/user-attachments/assets/42b55f16-31b8-4e98-b651-aebbda2802f7)

<br>




---
tags: documentation
last-updated: 2023-9-29
---

--------------------------------------------------------------------------------------------
## 2023-9-29: Documentation for DermaML Progress

### Goal

* The goal of this project is to develop machine learning software that detects hand age from image data collected by the University of Iowa faculty and staff. An emphasis was placed on feature engineering and understanding the mechanisms behind which the machine learning model(s) obtained results.

* Long term goals:
    - The model would be able to detect hand age as accurately as a doctor
    - The model could help us discover features important to determining hand age that doctors have not found yet

### Data Preprocessing

* REMBG
    - A Python library for removing the background of images, which is based on the REMBG neural network algorithm
    - We utilized this library to remove the green screen from hand images, however a green tint was left around the hand outline (this could be a point to consider later on - make sure green tint does not affect feature extraction)
    - Was able to separate green color from hand image

* Mediapipe
    - Mediapipe Hands utilizes an ML pipeline consisting of (1) a palm detection model that operates on the full image and returns an oriented hand bounding box, and (2) a hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints. 
    - Resource: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
    - We used Mediapipe to extract 20 landmarks on the hand and crop the portion of the palm within the surrounding landmarks
    - Landmarks can be used to extract different parts of the hand -> knuckles, fingertips, etc.
    - Returns palm extractions with irregular side lengths -> we cropped palm extractions to a rectangle when using GLCM to extract features
        - Cropping process of palm extraction is not automated, rectangle size is set/ does not transform for different palm sizes
        - If hands in images are resized to a set size, non-automated cropping could work
    - Mediapipe does not have a 100% hand detection rate -> struggles with older, heavily wrinkled hands
    - Use Mediapipe as baseline for dataset -> however many images Mediapipe detects will also be the amount of REMBG images that will be tested

### Feature Extraction

* Thresholding for Image Segmentation: Experimented with (1) Simple thresholding, (2) Adaptive thresholding, and (3) Otsu thresholding for defining hand texture (specifically wrinkles on back of hand)
    - Simple thresholding was not able to detect the hand outline 
    - Otsu thresholding was largely able to detect pigmentation, not wrinkle outlines (this could be useful if looking at pigmentation as a feature)
    - Adaptive thresholing was able to detect detailed wrinkles on hand with more success than using the naked eye. Utilizing LBP to extract features from these images could be useful for defining wrinkles.

* Hand Outline
    - Made progress obtaining hand outline by filling in hand and background with different colors (the more obscure folds and lines in the hand were defined, and wrinkles were no longer a feature)
    - Could be useful if looking at hand shape as feature
    - For future work, the hand outline could be pulled out instead of entire filled-in hand

* Edge Detection for Wrinkles
    - Utilized edge detection to pinpoint the place of wrinkles on the hand

### Methods

* Gray Level Co-occurence Matrix (GLCM)
    - Utilized gray level co-occurence matrix to pull out values of energy, homogeneity, contrast, and correlation
        - Energy: provides the sum of squared elements in the GLCM
        - Homogeneity: measures the closeness of distribution of elements in the GLCM to the GLCM diagonal
        - Contrast: measures the local variations in the GLCM
        - Correlation: measures the joint probabilty occurence of the specified pixel pairs
    - Resource: https://www.mathworks.com/help/images/texture-analysis-using-the-gray-level-co-occurrence-matrix-glcm.html

    - Testing GLCM
        - Function to test GLCM on images with pre-defined values for energy, homogeneity, contrast, and correlation has a long runtime and may produce inaccurate images based on user-input values
        - Function needs additional work, alternative is getting images from the internet with pre-defined feature values


* Local Binary Patterns (LBP)
    - LBP computes the local representation of texture, which is constructed by comparing each pixel with its surrounding neighborhood of pixels
        - Local features focus on analyzing specific regions within an image/ capture information about texture in smaller image patches, whereas global features analyze the entire image and consider the overall content of the image
    - We used LBP to extract texture information from hand images

### AutoML Experiments (with 11khands dataset)

* General Considerations
    - 11khands images are blurry -> could affect features for GLCM and LBP when analyzing texture
    - UofI dataset contains clear, well-defined hand images that include accurate textures (experimentation with this dataset may lead to better hand age prediction results overall)
    - Mediapipe dataset contained 47 images

* Experiments using GLCM
    - (1) REMBG (removed background/ no cropping): 
        - Extra Trees Regressor, Light Gradient Boosting Machine, and Dummy Regressor performed the best

    - (2) Mediapipe (irrgular cropping):
        - Huber regressor, Random Forest, and Linear Regression perform the best
        - Dummy Regressor 5th best
        - used cropped hand images that have irregular side lengths -> could lead to less accurate results

* Experiments using LBP
    - (3) REMBG (removed background/ no cropping):
        - AdaBoost Regressor, Extra Trees Regressor, and Gradient Boosting Machine performed the best
        - Dummy Regressor did not perform well

    - (4) Mediapipe (with single rectangular palm cropping):
        - Huber Regressor, Passive Aggressive Regressor, and Gradient Boosting Regressor performed the best
        - Dummy Regressor 6th best

    - (5) Mediapipe (with rectangular multi-area cropping): 
        - Two cropped (not automated) sections per hand - (Landmark 9 and Landmark 10)
        - Linear Regressor, Bayesian Ridge, and Light Gradient Boosting Machine performed the best
        - Dummy Regressor did not perform as well compared to experiments with LBP features from single cropped palm -> more lbp features per hand leads to better results with automl

### Future Directions

* Continue testing on different features (hand shape/ finger shape, knuckle texture, etc.)
* Automate cropping functions for palm regions so rectangle size automatically adjusts according to palm size
* Other possible data preprocessing functions to consider:
    - Image augmentation (blur, zoom, flip, rotate): avoid overfitting and help model generalize during training process
    - Image resizing: for consistency and better computation downstream in ML
    - Grayscale Pixel Values as Features
    - Mean Pixel Value of Channels

### Questions to Consider for the Future

* Why is Dummy Regressor performing so well when using features extracted from GLCM? Could it have some correlation to using blurry hands from 11khands datset?
* In what order is AutoML sorting the best models in? What specific metric is it using?
* What is different between LBP and GLCM in the way they extract their features that might cause one to be better suited for this project?



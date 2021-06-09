# LINDA Project
# Bangkit Team 42

Versions:
0.0.3

## To-dos of each team:
### Android Development (Ichsan and Yunda):
- a Design UI/UX
- b Build the apps
- c Run and debug

### Cloud Computing (Kia and Aldi):
- Make Kubernetes Server
- Make VM Instance
-

### Machine Learning (Wika and Daffa):
This is a model based on the paper by UC San Diego and Adobe research titled [Visually-Aware Fashion Recommendation and Design with Generative Image Models](http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm17.pdf). This model is run locally using a local machine for a total of 3 days, counting the time where the machine fails and suddenly crashes.

The difference between Linda's implementation and the paper is that we didn't put the final model, Preference Maximization, which further generates images that matches the user's taste as time goes on.

The dataset for the model can be downloaded in the dataset.sh by using the code:
`bash dataset.sh`
The dataset itself is based on the AmazonFashion dataset, that is made from the journal [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](http://cseweb.ucsd.edu/~jmcauley/pdfs/aaai16.pdf) that can be found [here](http://jmcauley.ucsd.edu/data/amazon/).

During the production of the model, we changed journals numerous times due to several limitations, such as time constraints when running the models, model complexity, and basic understanding. The previous models can be seen in the folder 'Machine Learning Unused Model'. Most are incomplete.

From the paper, we changed several codes in the DVBPR model, whereas for the image generation with GAN, we mainly use codes from [this source](https://github.com/Newmu/dcgan_code).

As for the tflite implementation, we lack the time as we made a mistake in the code and we only realized it after running.

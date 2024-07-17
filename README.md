# Streamlit-based Recommender System
#### EXPLORE Data Science Academy Unsupervised Predict

![Movie_Recommendations](resources/imgs/Image_header.png)

In today’s technology driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. 

One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options

## Getting Started

Clone the repository
```
https://github.com/monicafar147/unsupervised-predict-streamlit-template.git
```

### Prerequisites

The following packages need to be installed:

```
pip install -U streamlit numpy pandas scikit-learn
conda install -c conda-forge scikit-surprise
```

### Installing

The main product is a streamlit app.

To run the app locally:

```
cd unsupervised-predict-streamlit-template/
streamlit run edsa_recommender.py
```
The app contains 5 main pages:
1. Recommender System
  - content based approach
  - collaborative based approach
2. Solution Overview
3. Data analysis and plots
4. Meet the team
5. Slide deck

## Running the tests

- Go to the Recommender System page
- choose an algorithm that you want to test
- choose 3 movies from each dropdown
- click the recommend button

### Break down into recommender output

The output will be the top ten movies recommended.

The output from the content based model is based on
- genres
- actors

The output from the collaborative based model is based on
- user to user ratings 

## Deployment

The streamlit app can be deployed on an AWS EC2 instance.

The app is memory intensive and a minimum of 16gb RAM is recommended.

On the instance:
```
pip install -U streamlit numpy pandas scikit-learn
conda install -c conda-forge scikit-surprise
tmux
streamlit run --server.port 5000 edsa_recommender.py
```
To keep the app running continuously in the background:
Detach from the Tmux window by pressing ```ctrl + b``` and then ```d```. 

This should return you to the view of your terminal before you opened the Tmux window.

To go back to the Tmux window at any time (even if you've left your ssh session and then return):
Simply type ```tmux attach-session.```

## Built With

* [EXPLORE](https://github.com/Explore-AI/unsupervised-predict-streamlit-template) - The template we used
* [Streamlit](https://www.streamlit.io/)- App framework 

## 7. Team Members<a class="anchor" id="team-members"></a>
| Name                                                                                        |  Email              
|---------------------------------------------------------------------------------------------|--------------------             
| [Tumelo Matamane](https://github.com/MetaXide)                                                      |  tumelomatamane1@gmail.com
| [Nelisiwe Bezana](https://github.com/NelisiweBezana)                                                                                  | nelisiwebezana@gmail.com
| [Abel Masotla](https://github.com/Masotlaabel)                                                   | masotlaabel@gmail.com
| [Nolwazi Vezi](https://github.com/Lwazikayise)                                                | nolwazinvd@gmail.com
| [Khuphukani Maluleke](https://github.com/khupukani)                                         | khupukanimaluleke@gmail.com
| [Slindile Ndlela](https://github.com/SleeNdlela)                                                 | slindilendlela11@gmail.com


## Acknowledgments

* Xolisa Mzini (supervisor)
* Towards data science blog posts
* Medium blog posts
* Explore Data Science Academy

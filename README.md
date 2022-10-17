
# Neuroinformatics LSTM Synthesizer based on ICD (Impulse Control Disorder).

Computer-aided detection and diagnosis (CAD) systems are increasingly being used as an aid by clinicians 
for detection and interpretation of diseases medical analysis and identification. Let's say a person stays in a remote area far from a 
healthcare facility, or doesn't have the financial means to pay their clinic bill, or don't have the time to take sick leave from their jobs. In such a case, disease prediction using excessive-cease state-of-the-art equipment can be really beneficial especially when it comes to decision-making. Two distinct research findings are also addressed to highlight the need of having a thorough 
understanding of procedures when diagnosing a condition. Deep learning requires the use of large neural networks with densely interconnected, each of which can adjust its hyper-parameters in response to incoming input. It is because of this technology that computer architectures are enabled to examine things without the need for particular programming from humans. Here are the most recent tendencies and breakthroughs in the deep learning of the field in this lookup, which might have a huge impact on the effective identification and diagnosis of a variety of ailments. The aim is to look into the use of deep learning in the accurate evaluation of positive disease risk indicators, 
with the goal of supporting health workers in their decision-making.


## Introduction
Impulse control disorders are a type of mental problem marked by an failure to control an aggressive act or action that could be damaging to one or both. In neurological conditions, dopaminergic medicines, particularly reward agonists, are linked to issues with impulsive behavior. In addition to Parkinson's disease, restless legs dysfunction and variant Parkinson's conditions have been linked to poor impulse control. Young adults, masculine gender, higher novelty seeking, impulsivity, anxiety, & pre-morbid impulse control problems are the most frequent risk factors, based on the most recent epidemiological data. Frequently, the behaviors violate the rights of others or violate societal norms and laws. As per the 4th version of the Diagnostic and Statistical Manual of Mental Disorders (DSM-IV), 10.5 percent of citizens suffers from impulse control problem.

![fmri](https://user-images.githubusercontent.com/86592569/196053806-d45f356b-1804-4657-b9ca-2a13545010ce.jpg)
|:--:| 
|*fMRI of a subject at different time stamps*|

Future prospective results can also be forecasted thanks to advances in data analysis, which will assist the user in avoiding injury to a certain body component. As a result, wearable sensors and gadgets are becoming more common.
Males are significantly more likely than females to develop impulse control disorders, and these illnesses frequently coexist with other psychiatric problems or the use of drugs. Because impulse control problems are often missed or misdiagnosed, many people who suffer from them may not receive the care they require. A greater knowledge of the disease can assist bridge the treatment gap and ensure that people receive the care they need to improve their symptoms. Behavioral therapy is commonly used to treat impulse control issues, although drugs may also be helpful.
Despite the fact that dopamine and associated incentives have long been the research focus, researchers have recently begun to go outside such mechanisms for new insights into the neurological basis of ICD and possibly new therapy strategies.

## Using Deep Learning capabilities for the diagnosis of the disease
Convolutional neural network (CNN): A method of using deep learning that can be used to predict illness. It is a form of artificial neural network which has at minimum 3 layers: convolutional, pooling, and fully connected layers. CNN provides the ability in diagnosing illness with the help of creating a model to extract specific disease function. It can also be used to predict disease and drug discovery in clinical pictures. Examples of disease predictors that concern CNN use include illness category, disease classification, and disease identity. CNN uses images to predict the disease. Examples of diseases for which CNNs may be used include pores and skin cancer, breast cancer, and heart disease.
Recurrent Neural community (RNN) / LSTM: An in-depth knowledge of this is widely used in the processing of language applications, which includes the techniques of translating the device. It contains a small number of stacks of long and fast memory units (LSTMs), which may be neural community cells. RNN may be used to diagnose illness using disease-specific language models, including disease symptoms or patient information. Examples of diseases RNNs can develop disease predictor models that include Alzheimer's disease, Parkinson's disease, among others.

## Methodology and system description
The methodologies used in this process can be bifurcated into 2 major categories: Collection and preprocessing of the data and second is Extraction of meaningful data and applying Deep Learning techniques to get required results
FMRI pictures are 4D matrices that reflect each voxel's activity level in three-dimensional space and time. There are a variety of approaches for masking fMRI data, which are primarily influenced by the analysis' purpose. This project focuses on using resting-state networks to distinguish ADHD patients from healthy pastients. As a result, Smith's rs-fMRI components atlas (Smith et al.,2009) are used, shown in Table . The Smith atlas represents seventy resting-state networks (RSN) gathered from thousands of healthy patients using independent component analysis (ICA), can be seen in Figure \ref{fig4}. Because it prevents double-dipping, which can lead to over fitting, the Smith atlas has been preferred rather than dataset-specific ICA components.
However, a portion of this data frequently depicts the relevant information. For example, the main interest is in resting-state networks in this case. Masks have been used to hide the info that isn't important. Masks are simple filters that allow only a subset of data to pass while rejecting the rest. Masks effectively set the activation value of unwanted vowels to 0.  Overall procedure that has been followed in the project can be seen in the given figure

![methodology](https://user-images.githubusercontent.com/86592569/196053839-b1f0033a-498b-4a86-b8b8-dd5ff5979890.png)
|:--:| 
| *Project outline* |

## Data preparation and results based orientation

The train/test split paradigm has been used to make sure the model is tested on completely new data. The function below randomly splits the data into train and test and reshapes each part according to the model’s requirements. Details of the functions are given in Table
There are certain advantages to studying fMRI data with
long short term memory (LSTM) model. The major factor
behind this is that, unlike other machine learning or deep
learning algorithms, they can retain the inputs’ contextual in-
formation, allowing them to absorb data from prior sections of
the input sequence while processing the present one. However,
being extremely contextual isn’t necessarily a good thing. In
some circumstances, LSTMs aren’t the ideal option; being
contextual can lead to over-interpretation of data. LSTMs may
take longer to execute than a simple neural network and may
have many more parameters to change than a simple neural
network. It is critical to explore a variety of choices and
select the most appropriate and relevant scenario. Because
of the contextual nature of fMRI, illustration of the capacity
of LSTMs in analyzing them in this case has been done.
Because fMRI data depicts dynamic brain activity over time,
using LSTMs allows for the analysis of functional connectivity
to take advantage of the temporal information (which would
otherwise be lost). The problem with LSTMs is that they can
easily over fit training data, which reduces their prediction
ability. Regularization, which prevents the model from over-
fitting, solves this problem. Depending on the hardware setting
specified, the training time will be approximately 4 hours for
200 most epoch LSTMs and early action will be taken whether
the false positive rate does not rise for 5 epochs in a row.

![Details of functions used while processing the](https://user-images.githubusercontent.com/86592569/196053950-9eb3527f-a054-4f37-826b-023e8b77e232.png)
|:--:| 
| *Table 1 : Details of functions used while processing the fMRI images* |

Various experiments with extraordinary layer configurations
were done to provide the information about the the most ap-
propriate model setup. Table II lists the various characteristics
of the layers used for feature extraction, as well as how they
are altered over time. For this LSTM, several layers were
tried, consisting of 3 to 10 convolutional (return) layers with
different variations of units. Then, each layer was observed
with a percent loss to reduce the over fitting values and a linked
dense layer was created. It depends on the final convolution

![Table 2](https://user-images.githubusercontent.com/86592569/196054041-0a42a57c-5d26-42f8-a2c1-b297e7e7cc52.png)
|:--:| 
| *Table 2 : Distinct features in the fMRI image file of a subject* |


## Header of the FMRI images

ICA (Independent Component Analysis) is an effective
method for identifying different sources in fMRI data. As a
result, ICA and comparable algorithms can be used to identify
regions or networks with similar BOLD (blood-oxygen-level-
dependent) signals throughout time. To arrive at agreed com-
ponents, the CanICA takes into account information from both
inside and between subjects. CanICA is a readily available
object that may be used with multi-subject Nifti data, such as
filenames, to execute a multi-subject ICA breakdown by using
CanICA model. Its parameters were specified at build time

and then shaped it into inputs, as is done with every factor
in nilearn. Here are some examples of this system: Having
knowledge of the use of ICA and Dictionary, gives the ability
to create contour charts with the use of organizational fMRI
data. Individual parts of an ICA element will be imported as a
4D Nifti element with the help of the additives image feature
after inserting it into an fMRI dataset. Commonly requested
features in the waveform are extracted using ICA and related
strategies, which are visible in Figure 3. Accordingly, it no
longer only detects practical networks, but also non-neuronal
modes of activity, i.e. unified messages. Inside the view charts,
each is prominent. All features of the fMRI are shown in Table 3

![table 3](https://user-images.githubusercontent.com/86592569/196054079-b2599266-56eb-45b4-b599-db2012a7874d.png)
|:--:| 
| *Table 3 : The Finest Deep Learning Architectures’ Setup Descriptions* |

Various techniques are available to obtain spatial maps or
networks from group fMRI information. The techniques ex-
tract dedicated areas of the mind that exhibit comparably bold
fluctuations. Decomposition techniques allow the simultaneous
generation of many unbiased maps without the need to submit
a record of concerns (e.g. seeds or predecessors)

## Baseline models

The initial baseline model for the ICD and other diseases
was the binomial logistic regression model that was con-
structed using simple ML libraries. Deep learning models
were employed to male the models more robust, which were
trained using various DL libraries and as a result the model
independently predicted the accuracy. Various setups were
explored, with ”the number of iterations” being adjusted from
5 to 100. There was just one note per admission because
only discharge summaries notes were used. 

![ica](https://user-images.githubusercontent.com/86592569/196054381-b794ae57-2285-4703-b74d-8f5cbc690747.png)
|:--:| 
| *Independent Component Analysis Map* |

This classifier uses
features taken from this note as inputs. Features were directly
used as input features for the model. CNNs were used as a
starting point for multi-label classification problems requiring
algorithm customization. The identical input characteristics
and train-test data split were used as before. The LSTM cell
gate was utilised in all hidden layers, whereas the sigmoid
function was employed in the output layer. 

![smithrsn](https://user-images.githubusercontent.com/86592569/196054398-05207653-600e-4080-8f16-3020be3beddf.png)
|:--:| 
| *Smith Altas showing resting state Neural Networks* |

The algorithm was
batch gradient descent, and even the loss function became
binary cross entropy. A variety of neural network models
with 1 to 3 hidden layers were also tested. LSTM models
were implemented for each hidden layer, with neuron sizes ranging from 50 to 100.

![table 4](https://user-images.githubusercontent.com/86592569/196054448-55d68225-8728-4191-8828-8ba037ab0bc3.png)
|:--:| 
| *Table 4 : A Straightforward example to demonstrate extraction of areas via Smith's atlas resting state pathways* |

## Results

![roc](https://user-images.githubusercontent.com/86592569/196054455-6eeb6bf8-f739-4d42-af04-5fdc6272cefa.png)
|:--:| 
| *ROC Curve of the LSTM model with hyperparameter tuning* |

The ROC curve shows the balance between sensitivity and
specificity. Basically, Classifiers that provide curves in the
upper left corner direction perform better.A random classifier
(FPR = TPR) is predicted to present the points lying along
the diagonal. The closer the curve, the more it contains the
forty-five degree diagonal, the less accurate the test is. It is
clear from the graph that the AUC for the LSTM model ROC
curve as shown in Figure 5 is sustainable and gives a thin area
of 0.84. In conclusion, it can be said that the LSTM version
does the task of effectively classifying the high-quality class
in the datasets better, so it can help to predict the chances of
having ICD. Table V shows the different accuracy achieved
by using different techniques.

![table 5](https://user-images.githubusercontent.com/86592569/196054511-f30145db-370b-4f86-87de-e1d8cb0d790b.png)
|:--:| 
| *Table 5 : Different models and tuned architecture with their AUC and Accuracy* |

## Conclusion

The Deep learning based model (LSTM) and feature extrac-
tion approach are evaluated in this work. It can be concluded
that the dataset is not full, and that further feature vectors
are still missing. Attempts were done to generalise a subspace
of the real input space where the additional dimensions were
unknown, and so the classifiers could only do better than
70.10 percent up till now (LSTM), as seen in Figure 6 and
Figure 7. More feature vectors will need to be produced in
the future if similar studies are undertaken to build the dataset
utilised in this study so that the classifiers can form a better
understanding.

![modelacc](https://user-images.githubusercontent.com/86592569/196054549-bfef57df-19ea-4266-8a39-3d94c511e4d5.png)
|:--:| 
| *Model Accuracy : 70.10% (Hyper parameter tuned architecture)* |


![modelloss1](https://user-images.githubusercontent.com/86592569/196054755-3fad142a-5577-4177-8042-70420b053247.png)
|:--:| 
| *Model Loss : 0.64-0.89 (Hyper parameter tuned architecture)* |


![modacc](https://user-images.githubusercontent.com/86592569/196054539-1a7a0b0e-1fa8-4c80-a1d1-6bd998fe1f19.png)
|:--:| 
| *Model Accuracy : 71.10% (LSTM cells with default architecture)* |

![modloss](https://user-images.githubusercontent.com/86592569/196054622-50b9db65-3573-4b10-b3ec-ac6bb46f9042.png)
|:--:| 
| *Model Loss : 0.84 - 1.24 (LSTM cells with default architecture)* |

-------------------------------------------

Here is the Link to our research [paper](https://drive.google.com/file/d/1-K0-Js2PAsuhTjB9THY7m-BE_hwR8iLg/view)




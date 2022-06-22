# PPG2ABP using GAN
This repository aims to research on PPG2ABP using GAN.

# Terminology
|Abbreviation|Word|Meaning|
|------------|----|-------|
|BP     |Blood pressure     |       |
|ECG    |Electrocardiogram  |ECG is the process of producing an electrocardiogram, a recording of the heart's electrical activity.|
|PPG    |Photoplethysmogram |PPG can be used to detect blood volume changes in the microvascular bed of tissue.|
|ABP    |Arterial Blood Pressure|ABP is defined as the force that is exerted by the blood on the arterial wall.|
|PTT    |Pulse Transit Time |PPT is the time taken for the arterial pulse pressure wave to travel from the aortic valve(대동맥 판막) to a peripheral site(말초).|


<br>

# Paper-Review 
|Read_Date|Paper|Authors|Links|Data|Github|Tag|
|------|---|---|---|---|---|---|
|`06/15/2022`|PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks|Ibtehaz.,<br/> Rahman.|[[Paper]](https://www.semanticscholar.org/paper/PPG2ABP%3A-Translating-Photoplethysmogram-%28PPG%29-to-Ibtehaz-Rahman/26238aa1d8ec51788f1b5e22aeb6ea88cac0c41f)<br/>[[Review]](paper-review/PPG2ABP_T2TGAN.md)|[[MIMIC II dataset]](https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation)||`CNN` `PPG2ABP`|
|`06/16/2022`|Estimation of Continuous Blood Pressure from PPG via a Federated Learning Approach|Brophy.,<br/> Vos.|[[Paper]](https://arxiv.org/abs/2102.12245)<br/>[[Review]](paper-review/Estimation_of_Continuous_Blood_Pressure_from_PPG_via_a_Federated_Learning_Approach.md)|[[Train]](https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation)<br/>[[Test]](https://outbox.eait.uq.edu.au/uqdliu3/uqvitalsignsdataset/index.html)|[[Github]](https://github.com/Brophy-E/T2TGAN)|`T2TGAN` `PPG2ABP`|
|`06/17/2022`|Can Photoplethysmography Replace Arterial Blood Pressure in the Assessment of Blood Pressure?|Martínez., <br/> Howard.|[[Paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6209968/)<br/>[[Review]](paper-review/PPGABP.md)|[[MIMIC II dataset]](https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation)||`PPG2ABP`|
|`06/20/2022`|Generative Adversarial Nets|Goodfellow., |[[Paper]](https://arxiv.org/pdf/1406.2661.pdf)<br/>[[Review]](paper-review/GAN.md)||[[Github](https://github.com/goodfeli/adversarial)]|`GAN`|
|`06/21/2022`|DCGAN - Unsupervised Representation Learning with Deep Convolutional Gnerative Adversarial Networks|Radford., <br/> Metz. <br/> Chintala. |[[Paper]](https://arxiv.org/abs/1511.06434)<br/>[[Review]](paper-review/DCGAN.md)|||`DCGAN`|



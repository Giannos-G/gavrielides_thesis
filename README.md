# This is the GitHub Repository of my Thesis during my Undergaduate Student years at the National Tecnhical University of Athens
## Title: A software analysis tool for Energy and Time-aware function placement on the Edge 
## Introduction

In this thesis, we present a tool which proposes a way in which the individual functions of a monolithic code could run in a serverless environment as individual and independent application packaged containers, so that given a maximum runtime threshold by the user, the minimum possible power consumption in our cluster is achieved. Our serverless infrastructure (cluster) consists of Aarch64 Edge devices which are orchestrated by Kubernetes. Finally, using OpenFaaS serverless platform and Docker we managed to package our applications into containers and run them in our serverless environment.

Our tool analyzes (profiles) the code file given in terms of memory requirements and the execution time of each of its individual functions on our testing device (personal computer) and then using machine learning techniques (Linear Regression and Decision Tree Regression models) predicts the energy consumption and the execution time of each function on each of our devices. The utilization of this information and the decision for the proposed solution is achieved through our developed algorithm which is inspired by the well-known * simulated annealing algorithm *. Furthermore, using Openfaas, Kubernetes and Docker we show that our work is quite promising and to be continued.

# Logistic_Regreesion_W



This web application visualizes a logistic regression model with one feature and its corresponding loss landscape. Users can interact with the model by adjusting weight **w** using a widget, and observe how changes affect the loss function and help reach the global minima.


## Features

* Plot of logistic Regression Model: Visualizes how the model fits a given dataset with one feature.

* Loss Landscape Plot: Shows the Binary Cross Entropy Loss (BCE Loss) landscape for different values of weight.

* Interactive Widgets: Adjust weight (w) in real-time and see the effect on the loss function and  sigmoid curve.

* Equations Display: Displays the equations used in the plots:
      sigmoid: Used in the first plot.
      Binary Cross Entropy (BCE) Equation: Used in the second plot.




## Installation

To run this application, you need to have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository:

    `git clone https://github.com/Hawar-Dzaee/Logistic_Regreesion_W.git`


2. Install the required packages:

    `pip install -r requirements.txt`

## Usage

To run the application, use the following command:

  `streamlit run main.py`



## Files

  main.py: Contains the code for the web application.
  LO_W.ipynb : a notebook that follows main.py (to some extend)
  requirements.txt: Lists the required Python packages.
  LICENSE : MIT License



## Example

Once the application is running, you will see two plots:

1. sigmoid curve fitting the data :

  * Shows the S-shaped curve fitting the dataset with one feature(two classes 0 & 1).


2. Loss Landscape Plot:

  * Displays Binary Cross Entropy (BCE) landscape.


![Alt text](<Screen Shot 2024-06-06 at 6.47.02 PM.png>)

## Requirements 

The application requires the following Python packages:

* Streamlit
* Torch
* Numpy
* Plotly



## License

This project is licensed under the MIT License. See the **LICENSE** file for more details.


Contact
For any questions or suggestions, please open an issue or contact [hawardizayee@gmail.com].





import torch 
import torch.nn as nn 
import streamlit as st
import numpy as np 
import plotly.graph_objects as go 

# --------------------------------

lower_0,upper_0 = -1,0
sample_size_0= 6
noise_0=0.25

lower_1,upper_1 = 0,1
sample_size_1 = 5
noise_1 = 0.25



x0 = torch.linspace(lower_0,upper_0,sample_size_0) + torch.tensor([noise_0])
x1 = torch.linspace(lower_1,upper_1,sample_size_1) - torch.tensor([noise_1])

X = torch.cat((x0,x1),dim=0)

y = torch.cat( (torch.zeros(len(x0)),torch.ones(len(x1))), dim = 0)

# colors = ['orange' if i>0 else 'purple' for i in y]



# --------------------------------
# Generating Data 

def generate_plot(w):


# plot the Dataset
  scatter_class_0 = go.Scatter(
    x = x0,
    y = torch.zeros(len(x0)),
    mode = 'markers',
    marker = dict(color='purple'),
    name = 'class 0'
    )
  
  scatter_class_1 = go.Scatter(
    x = x1,
    y = torch.ones(len(x1)),
    mode = 'markers',
    marker = dict(color='orange'),
    name = 'class 1'
    )


  # nonlinear Transformation
  z = w*X
  y_hat = torch.sigmoid(z)


  # sigmoid 
  non_linear_line = go.Scatter(
      x = torch.linspace(-3,3,1000),
      y = torch.sigmoid(w*torch.linspace(-3,3,1000)),
      mode = 'lines',
      line = {'color' : 'rgb(27,158,119)'},
      name = 'model'
  )

  layout = go.Layout (
              xaxis = dict(
                range = [-1.5,1.5],
                title = 'X',
                zeroline = True,
                zerolinewidth = 2,
                zerolinecolor = 'rgba(205, 200, 193, 0.7)'
              ),
              yaxis = dict(
                range = [-0.5,1.5],
                title = 'Y',
                zeroline = True,
                zerolinewidth = 2,
                zerolinecolor = 'rgba(205, 200, 193, 0.7)'
              ),
              height = 500,
              width = 2600
            )



  figure = go.Figure(data=[scatter_class_0,scatter_class_1, non_linear_line ],layout=layout)

  return figure
# ------------------------------------------------------
# Calculating Loss Function Landscape 
possible_weights = torch.linspace(-5,25,100)
L= []
loss_fn = nn.BCEWithLogitsLoss()



for w in possible_weights:
  z = w * X
  loss = loss_fn(z,y)
  L.append(loss)
  #-----------

L = torch.as_tensor(L)
secret_weight = possible_weights[torch.argmin(L)]


# ------------------------------------------------------
# Ploting Loss function Landscape

def loss_landscape(w):

    loss_landscape = go.Scatter(
            x = possible_weights,
            y = L,
            mode = 'lines',
            line = dict(color='pink'),
            name ='Loss function landscape'
        )

    Global_minima = go.Scatter(
        x = (secret_weight,),
        y = (torch.min(L),),
        mode = 'markers',
        marker = dict(color='yellow',size=10,symbol='diamond'),
        name = 'Global minima'
    )

    z = w*X
    loss = loss_fn(z,y)

    ball = go.Scatter(
    x = (w,),
    y = (loss,),
    mode = 'markers',
    marker= dict(color='red'),
    name = 'loss'
    )

    layout = go.Layout(
            xaxis = dict(title='w',
                         range = [-8,25],
                         zeroline = True,
                        zerolinewidth = 2,
                        zerolinecolor = 'rgba(205, 200, 193, 0.7)'),

            yaxis = dict(title='L',
                         range=[0,1.6],
                        zeroline = True,
                        zerolinewidth = 2,
                        zerolinecolor = 'rgba(205, 200, 193, 0.7)')
        )


    figure = go.Figure(data = [loss_landscape,Global_minima,ball],layout=layout)
    
    return figure

# ------------------------------------------------------
#streamlit 

st.set_page_config(layout='wide')


st.title("Logistic Regression")
st.write('By : Hawar Dzaee')


with st.sidebar:
    st.subheader("Adjust the parameters to minimize the loss")
    w_val = st.slider("weight (w):", min_value=-4.0, max_value=18.0, step=0.1, value= -3.5)



container = st.container()


with container:
 

    # Create two columns with different widths
    col1, col2 = st.columns([3,3])

    # Plot figure_1 in the first column
    with col1:
        figure_1 = generate_plot(w_val)
        st.plotly_chart(figure_1, use_container_width=True)  # Change aspect ratio to 1.0
        st.latex(r'''\hat{{y}} = \frac{1}{1 + e^{-(\color{green}w\color{black}X)}}''')
        st.latex(fr'''\hat{{y}} = \frac{{1}}{{1 + e^{{-(\color{{green}}{{{w_val}}}\color{{black}}X)}}}}''')



    z = w_val*X
    loss = loss_fn(z,y)

    loss_class_0 = torch.mean(-torch.log(1-torch.sigmoid(w_val*x0)))
    loss_class_1 = torch.mean(-torch.log(torch.sigmoid(w_val*x1)))



    with col2:
       figure_2 = loss_landscape(w_val)
       st.plotly_chart(figure_2,use_container_width=True)
       st.write('     ')
       st.write('     ')
       st.latex(r"""L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]""")
       st.latex(rf"""L_{{\text{{class 0}}}} = \textcolor{{purple}}{{{loss_class_0:.4f}}}  \qquad L_{{\text{{class 1}}}} = \textcolor{{orange}}{{{loss_class_1:.4f}}}""")
       st.latex(rf"""L_{{\text{{total}}}} = \textcolor{{red}}{{{loss:.4f}}}""")
       




  
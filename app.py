import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('churn_dl.sav', 'rb') as model_file:
    churn_model = pickle.load(model_file)

def main():
    st.title('Multiplayer Gaming Churn Prediction')

    # Likert scale input
    st.write('Rate Your Experience - Strongly Disagree(1) - Strongly Agree(5)')
    highping = st.slider('**High Ping**', 1, 5)
    gamedifficulty = st.slider('**Game Difficulty**', 1, 5)
    gameprog = st.slider('**Game Progression**', 1, 5)
    gamecon = st.slider('**Game Content**', 1, 5)
    lacksocial = st.slider('**Lack of Social Features**', 1, 5)
    socialint = st.slider('**Social Interactions**', 1, 5)
    newgame = st.slider('**New Games**', 1, 5)
    techissue = st.slider('**Technical Issues**', 1, 5)
    internetacc = st.slider('**Limited Internet Access**', 1, 5)
    devicesost = st.slider('**Device Cost**', 1, 5)
    powerout = st.slider('**Power Outages**', 1, 5)
    paywall = st.slider('**Paywalls**', 1, 5)
    friendinter = st.slider('**Interaction with Friends**', 1, 5)

    # Button to make prediction
    churn_predict = ''
    if st.button('Predict Churn'):
        # Perform prediction
        input_data ={'High Ping ':highping,'Game Difficulty ':gamedifficulty, 'Game Progression':gameprog, 'Game Content':gamecon, 'Lack of Social Features':lacksocial, 'Social Interactions':socialint, 'New Games':newgame, 'Technical Issues':techissue, 'Limited Internet Access':internetacc, 'Device Cost':devicesost, 'Power Outages':powerout, 'Paywalls':paywall, 'Low Social Interactions with Friends':friendinter}
        input_df =pd.DataFrame([input_data])
        churn_out = churn_model.predict(input_df)
        churn_bin = (churn_out> 0.5).astype(int)
        if churn_bin == 1:
            churn_predict = 'The Player is likely to Churn'
        else:
            churn_predict = 'The Player will Retain'
    st.success(churn_predict)

if __name__ == '__main__':
    main()

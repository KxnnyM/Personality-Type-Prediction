import streamlit as st
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle
import time


st.header("MBTI Personality Prediction")
st.text("Give us a text so that we can predict your personality type.")
x = st.text_input("Enter text below:",placeholder="Type here...",help="Please press the enter button on your keyboard to process your text.")
if x:
    
    with st.expander("Click here to view your text."):
        st.info('You can scroll horizontally to view through the entire text you have put in.',icon="ℹ️")
        st.text(x)
if x:
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    st.success('Here is your personality type!')
sample_test = [x]


loaded_model_1 = pickle.load(open("C:/Users/kenny/Documents/random/capstone/trained_model_1.sav","rb"))
loaded_model_2 = pickle.load(open("C:/Users/kenny/Documents/random/capstone/trained_model_2.sav","rb"))
loaded_model_3 = pickle.load(open("C:/Users/kenny/Documents/random/capstone/trained_model_3.sav","rb"))
loaded_model_4 = pickle.load(open("C:/Users/kenny/Documents/random/capstone/trained_model_4.sav","rb"))

postdf=pd.read_csv("post.csv")

def pred(sample_test):
    vector = CountVectorizer(stop_words='english', max_features=1500)
    vector = vector.fit(postdf.post_list)
    features = vector.transform(sample_test)
    transform = TfidfTransformer()
    finalfeatures = transform.fit_transform(features).toarray()
    sample_input = transform.fit_transform(features).toarray()
    vect = vector.transform(sample_test).toarray()
    loaded_model_1.predict(vect)
    if loaded_model_1.predict(vect) == 1:
        a = "I"
        A="Introversion"
    else:
        a = "E"
        A="Extraversion"
    vect = vector.transform(sample_test).toarray()
    loaded_model_2.predict(vect)
    if loaded_model_2.predict(vect) == 1:
        b = "N"
        B="Intuition"
    else:
        b = "S"
        B="Sensing"
    vect = vector.transform(sample_test).toarray()
    loaded_model_3.predict(vect)
    if loaded_model_3.predict(vect) == 1:
        c = "T"
        C="Thinking"
    else:
        c = "F"
        C="Feeling"
    vect = vector.transform(sample_test).toarray()
    loaded_model_4.predict(vect)
    if loaded_model_4.predict(vect) == 1:
        d = "J"
        D="Judging"
    else:
        d = "P"
        D="Perceiving"
    st.subheader(a+b+c+d+" ("+A+"-"+B+"-"+C+"-"+D+")")

    if a+b+c+d=="ISTJ":
        st.write("The Inspector:")
        st.caption("Reserved and practical, they tend to be loyal, orderly, and traditional.")
        st.caption("Characteristics: Responsible, sincere, analytical, reserved, realistic, systematic. Hardworking and trustworthy with sound practical judgment.")
    elif a+b+c+d=="ISTP":
        st.write("The Crafter:")
        st.caption("Highly independent, they enjoy new experiences that provide first-hand learning.")
        st.caption("Characteristics: Action-oriented, logical, analytical, spontaneous, reserved, independent. Enjoy adventure, skilled at understanding how mechanical things work.")
    elif a+b+c+d =="ISFJ":
        st.write("The Protector:")
        st.caption("Warm-hearted and dedicated, they are always ready to protect the people they care about.")
        st.caption("Characteristics: Warm, considerate, gentle, responsible, pragmatic, thorough. Devoted caretakers who enjoy being helpful to others.")
    elif a+b+c+d =="ISFP":
        st.write("The Artist:")
        st.caption("Easy-going and flexible, they tend to be reserved and artistic.")
        st.caption("Characteristics: Gentle, sensitive, nurturing, helpful, flexible, realistic. Seek to create a personal environment that is both beautiful and practical.")
    elif a+b+c+d=="INFJ":
        st.write("The Advocate:")
        st.caption("Creative and analytical, they are considered one of the rarest Myers-Briggs types.")
        st.caption("Characteristics: Idealistic, organized, insightful, dependable, compassionate, gentle. Seek harmony and cooperation, enjoy intellectual stimulation.") 
    elif a+b+c+d=="INFP":
        st.write("The Mediator:")
        st.caption("Idealistic with high values, they strive to make the world a better place.")
        st.caption("Characteristics: Sensitive, creative, idealistic, perceptive, caring, loyal. Value inner harmony and personal growth, focus on dreams and possibilities.")
    elif a+b+c+d=="INTJ":
        st.write("The Architect:")
        st.caption("High logical, they are both very creative and analytical.")
        st.caption("Characteristics: Innovative, independent, strategic, logical, reserved, insightful. Driven by their own original ideas to achieve improvements.")
    elif a+b+c+d=="INTP":
        st.write("The Thinker:")
        st.caption("Quiet and introverted, they are known for having a rich inner world.")
        st.caption("Characteristics: Intellectual, logical, precise, reserved, flexible, imaginative. Original thinkers who enjoy speculation and creative problem solving.")
    elif a+b+c+d=="ESTP":
        st.write("The Persuader:")
        st.caption("Out-going and dramatic, they enjoy spending time with others and focusing on the here-and-now.")
        st.caption("Characteristics: Outgoing, realistic, action-oriented, curious, versatile, spontaneous. Pragmatic problem solvers and skillful negotiators.")
    elif a+b+c+d=="ESTJ":
        st.write("The Director:")
        st.caption("Assertive and rule-oriented, they have high principles and a tendency to take charge.")
        st.caption("Characteristics: Efficient, outgoing, analytical, systematic, dependable, realistic. Like to run the show and get things done in an orderly fashion.")
    elif a+b+c+d=="ESFP":
        st.write("The Performer:")
        st.caption("Outgoing and spontaneous, they enjoy taking center stage.")
        st.caption("Characteristics: Playful, enthusiastic, friendly, spontaneous, tactful, flexible. Have strong common sense, enjoy helping people in tangible ways.")
    elif a+b+c+d=="ESFJ":
        st.write("The Caregiver:")
        st.caption("Soft-hearted and outgoing, they tend to believe the best about other people.")
        st.caption("Characteristics: Friendly, outgoing, reliable, conscientious, organized, practical. Seek to be helpful and please others, enjoy being active and productive.")
    elif a+b+c+d=="ENFP":
        st.write("The Champion:")
        st.caption("Charismatic and energetic, they enjoy situations where they can put their creativity to work.")
        st.caption("Characteristics: Enthusiastic, creative, spontaneous, optimistic, supportive, playful, Value inspiration, enjoy starting new projects, see potential in others.")
    elif a+b+c+d=="ENFJ":
        st.write("The Giver:")
        st.caption("Loyal and sensitive, they are known for being understanding and generous.")
        st.caption("Characteristics: Caring, enthusiastic, idealistic, organized, diplomatic, responsible. Skilled communicators who value connection with people.")
    elif a+b+c+d=="ENTP":
        st.write("The Debater:")
        st.caption("Loyal and sensitive, they are known for being understanding and generous.")
        st.caption("Characteristics: Inventive, enthusiastic, strategic, enterprising, inquisitive, versatile. Enjoy new ideas and challenges, value inspiration.")
    elif a+b+c+d=="ENTJ":
        st.write("The Commander:")
        st.caption("Outspoken and confident, they are great at making plans and organizing projects.")
        st.caption("Characteristics: Strategic, logical, efficient, outgoing, ambitious, independent. Effective organizers of people and long-range planners.")



if x:
    vector = CountVectorizer(stop_words='english', max_features=1500)
    features = vector.fit_transform(sample_test)

    transform = TfidfTransformer()
    sample_input = transform.fit_transform(features).toarray()

    pred(sample_test)
    my_bar = st.progress(100)
#streamlit run "C:\Users\kenny\Documents\random\capstone\deploy.py"
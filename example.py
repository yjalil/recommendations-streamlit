import streamlit as st
import pandas as pd
from streamlit_image_select import image_select

df = pd.read_csv('products.csv')
prodcuts = df.head(4)
images = prodcuts['Image'].to_list()


def display_image(index):
    st.image(images[index])

if 'images' not in st.session_state:
    st.session_state['image'] = 0

def increase_rows():
    st.session_state['image'] += 1

st.button('Add person', on_click=increase_rows)

img = image_select(
    label="Select a product",
    images=images,
    return_value="original"
)
for i in range(st.session_state['images']):
    display_image(i)

# Show the results
st.subheader('People')
for i in range(st.session_state['images']):
    st.write(
        f'Person {i+1}:',
        st.session_state[f'first_{i}'],
        st.session_state[f'middle_{i}'],
        st.session_state[f'last_{i}']
    )

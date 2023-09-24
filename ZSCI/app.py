from ZSIC import ZeroShotImageClassification
from PIL import Image
import random
import os
import streamlit as st

# hide hamburger and customize footer
hide_menu = """
<style>

#MainMenu {
    visibility:hidden;
}

footer{
    visibility:visible;
}

footer:after{
    content: 'With 🫶️ from Shubham Shankar.';
    display:block;
    position:relative;
    color:grey;
    padding;5px;
    top:3px;
}
</style>

"""
# Styling ----------------------------------------------------------------------

st.image("icon.jpg", width=85)
st.title("ZSCI")
st.subheader("Zero Shot Image Classifier")
st.markdown(hide_menu, unsafe_allow_html=True)

# Intro ----------------------------------------------------------------------

st.write(
    """

    Hi 👋, I'm **:red[Shubham Shankar]**, and welcome to my **:green[Zero Shot Image Classification Application]**! :rocket: This program makes use of **:blue[ZSCI]** and **:orange[CLIP]** model, 
    To perform Classification on Vehicle dataset obtained from **:violet[Roboflow]** .  ✨

    """
)

st.markdown('---')

st.write(
    """
    ### App Interface!!

    :dog: The web app has an easy-to-use interface. 

    1] **:green[Select Image]**: Select an Image randomly from Dataset on click of a button.

    2] **:violet[Classification - Label]**: Performs Classification using ZSCI and CLIP model.

    """
)

st.markdown('---')

st.subheader('CLIP')

st.write(
    """
    Contrastive Language-Image Pre-training (CLIP forshort) is a state-of-the-art model introduced by OpenAl.
    
    CLIP can find whether a given image and textual description match without being trained for a specificdomain.

    """
)

st.subheader('Architecture')

st.write(
    """
    * The text encoder's backbone is a transformer model, and the base size uses 63 millions- parameters,12 layers, and a 512-wide model containing 8 attention heads.
    * The image encoder, on the other hand, uses both a Vision Transformer (ViT) and a ResNet50 as its backbone, responsible for generating the feature representation of the image.
    """
)

st.image('CLIP.png')

st.markdown('---')

st.error(
    """
    Connect with me on [**Github**](https://github.com/RATHOD-SHUBHAM) and [**LinkedIn**](https://www.linkedin.com/in/shubhamshankar/). ✨
    """,
    icon="🧟‍♂️",
)

st.markdown('---')

# Get Model
# default: 'ViT-B/32', en
zsic = ZeroShotImageClassification()

st.info("Click to Select a Random Vehicle")

if st.button('Select'):
    # Get the path to the folder
    folder_path = "images"

    # Get a list of all the files in the folder
    filenames = os.listdir(folder_path)

    # Select a random file from the list
    random_filename = random.choice(filenames)

    # Print the random filename
    print(random_filename)

    image_path = os.path.join(folder_path, random_filename)

    img = Image.open(image_path)

    st.text("Selected Image")
    st.image(img)

    class_names = ["sedan", "pickup", "suv", "coupe", "van"]

    preds = zsic(image=img,
                 candidate_labels=class_names,
                 hypothesis_template="A photo of {}",
                 )

    predictions = preds['labels'][preds['scores'].index(max(preds['scores']))]

    print(predictions)

    print(type(predictions))

    st.success(predictions, icon="✅")

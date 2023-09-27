from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import matplotlib.pyplot as plt
import streamlit as st

# Get the base directory
BASE_DIR = Path(__file__).resolve(strict=True).parent

# hide hamburger and customize footer
hide_menu= """
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

icon_path = BASE_DIR/"icon.jpg"

st.image(str(icon_path), width=85)
st.title("T2I")
st.subheader("Text To Image Search")
st.markdown(hide_menu, unsafe_allow_html=True)

# Intro ----------------------------------------------------------------------

st.write(
    """

    Hi 👋, I'm **:red[Shubham Shankar]**, and welcome to my **:green[Image Search Application]**! :rocket: This program makes use of OpenAI **:blue[CLIP]** and Hugging Face **:orange[Sentence-Transformers]** model, 
    which can be used for image search and for zero-shot image classification.  ✨

    """
)

st.markdown('---')

st.write(
    """
    ### App Interface!!

    :dog: The web app has an easy-to-use interface. 

    1] **:green[Query]**: Enter the type of Image to be searched for.
    """
)

st.markdown('---')

st.info(
    """
    Visit this page to learn more about [Image Search sentence-transformers](https://www.sbert.net/examples/applications/image-search/README.html).
    """,
    icon="👾",
)

st.markdown('---')

st.subheader('Image Search')

pipeline = BASE_DIR/'ImageSearch.png'
st.image(str(pipeline))

st.markdown('---')

st.error(
    """
    Connect with me on [**Github**](https://github.com/RATHOD-SHUBHAM) and [**LinkedIn**](https://www.linkedin.com/in/shubhamshankar/). ✨
    """,
    icon="🧟‍♂️",
)

st.markdown('---')



# Get Input Directory

input_data_path = "animals"

data_path = BASE_DIR/input_data_path

data_path = str(data_path) + '/'


# Model
model = SentenceTransformer('clip-ViT-B-32')

# """# Image embeddings."""

img_names = list(glob.glob(f'{data_path}*.jpg'))


img_emb = model.encode(
                            [Image.open(filepath) for filepath in img_names],
                            batch_size=128,
                            convert_to_tensor=True,
                            show_progress_bar=True
                            )


def plot_images(images, query, n_row=2, n_col=2):
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()

    for img, ax in zip(images, axs):
        ax.set_title(query)
        ax.imshow(img)

    plt.savefig('foo.png')


def search(query, k=4):
    query_emb = model.encode([query],
                             convert_to_tensor=True,
                             show_progress_bar=False)

    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]


    matched_images = []
    for hit in hits:
        matched_images.append(Image.open(img_names[hit['corpus_id']]))

    plot_images(matched_images, query)


q = st.text_input('Query', 'Image of Human holding Animals')

if st.button('RUN'):
    search(query = q)


    op_image = BASE_DIR/'foo.png'
    st.image(str(op_image))

